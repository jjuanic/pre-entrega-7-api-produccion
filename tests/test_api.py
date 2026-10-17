"""API + Redis + worker de punta a punta, con fakeredis y un LLM guionado (sin red)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import fakeredis
import httpx
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import compilar
from app.hitl import PublicadorRedis
from app.jobs import JobStore
from app.main import app
from app.worker import Worker
from conftest import analisis, decidir, guion, investigacion_ok

TAREA = "¿Cuántos hilos usa ThreadPoolExecutor por defecto en una máquina de 8 núcleos?"


def llm_flujo_completo():
    return guion(
        decidir("investigador", instruccion="Buscá el max_workers por defecto."),
        *investigacion_ok(),
        decidir("analista", instruccion="Calculá min(32, 8 + 4)."),
        *analisis(),
        decidir("FINISH", razon="Todo válido."),
        AIMessage(content="Usa 12 hilos (concurrent_futures.md)."),
    )


class Entorno:
    """App FastAPI conectada a un Redis falso + un Worker que procesamos a mano."""

    def __init__(self, llm: Any = None, timeout: float = 30) -> None:
        self.redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        self.store = JobStore(self.redis, cola="cola:test")
        app.state.redis = self.redis
        app.state.store = self.store
        app.state.checkpointer = InMemorySaver()
        app.state.observabilidad = "none"
        grafo = compilar(llm or llm_flujo_completo(), checkpointer=app.state.checkpointer, publicador=PublicadorRedis(self.redis))
        self.worker = Worker(grafo, self.store, nombre="w-test", timeout=timeout)
        self.cliente = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def procesar_siguiente(self) -> dict[str, Any]:
        crudo = await self.store.tomar("w-test", timeout=0.1)
        assert crudo is not None, "la cola estaba vacía"
        await self.worker.procesar(json.loads(crudo))
        await self.store.confirmar("w-test", crudo)
        return json.loads(crudo)


def test_post_encola_y_devuelve_job_id_sin_ejecutar_el_agente() -> None:
    async def caso() -> None:
        e = Entorno()
        inicio = time.perf_counter()
        r = await e.cliente.post("/tasks", json={"tarea": TAREA})
        assert time.perf_counter() - inicio < 0.5  # responde enseguida: nadie corrió el grafo
        assert r.status_code == 202
        cuerpo = r.json()
        assert cuerpo["status"] == "PENDING" and len(cuerpo["job_id"]) == 32

        # El estado PENDING ya está en Redis y el mensaje en la cola.
        estado = await e.cliente.get(f"/tasks/{cuerpo['job_id']}")
        assert estado.status_code == 200 and estado.json()["status"] == "PENDING"
        assert await e.redis.llen("cola:test") == 1

    asyncio.run(caso())


def test_validacion_y_404() -> None:
    async def caso() -> None:
        e = Entorno()
        assert (await e.cliente.post("/tasks", json={"tarea": "corta"})).status_code == 422
        assert (await e.cliente.get("/tasks/no-existe")).status_code == 404
        assert (await e.cliente.post("/tasks/no-existe/approve", json={"aprobado": True})).status_code == 404

    asyncio.run(caso())


def test_flujo_completo_sin_hitl_termina_done() -> None:
    async def caso() -> None:
        e = Entorno()
        job_id = (await e.cliente.post("/tasks", json={"tarea": TAREA})).json()["job_id"]
        await e.procesar_siguiente()
        job = (await e.cliente.get(f"/tasks/{job_id}")).json()
        assert job["status"] == "DONE"
        assert job["resultado"]["respuesta_final"].startswith("Usa 12 hilos")
        assert job["resultado"]["riesgo"]["critica"] is False
        assert job["ejecuciones"] == 1 and job["finished_at"]
        # Los checkpoints del hilo quedaron guardados.
        cps = (await e.cliente.get(f"/tasks/{job_id}/checkpoints")).json()
        assert cps["total"] > 5 and cps["thread_id"] == job_id

    asyncio.run(caso())


def test_hitl_por_api_pausa_aprueba_y_publica() -> None:
    async def caso() -> None:
        e = Entorno()
        job_id = (await e.cliente.post("/tasks", json={"tarea": TAREA, "publicar": True})).json()["job_id"]

        # Aprobar antes de que el worker llegue a la pausa: conflicto.
        assert (await e.cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})).status_code == 409

        await e.procesar_siguiente()
        job = (await e.cliente.get(f"/tasks/{job_id}")).json()
        assert job["status"] == "AWAITING_APPROVAL"
        assert job["aprobacion_pendiente"]["accion"] == "publicar_informe"
        assert await e.redis.xlen("informes:publicados") == 0  # todavía no se publicó nada

        r = await e.cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "ana"})
        assert r.status_code == 202 and r.json()["status"] == "PENDING"
        # Una segunda aprobación no puede reanudar el mismo grafo otra vez.
        assert (await e.cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})).status_code == 409

        mensaje = await e.procesar_siguiente()
        assert mensaje["tipo"] == "resume"
        job = (await e.cliente.get(f"/tasks/{job_id}")).json()
        assert job["status"] == "DONE" and job["ejecuciones"] == 2
        assert job["aprobacion"]["revisor"] == "ana"
        assert job["resultado"]["publicacion"]["destino"] == "redis-stream:informes:publicados"
        informes = (await e.cliente.get("/informes")).json()
        assert len(informes) == 1 and informes[0]["job_id"] == job_id

    asyncio.run(caso())


def test_hitl_rechazado_queda_rejected() -> None:
    async def caso() -> None:
        e = Entorno()
        job_id = (await e.cliente.post("/tasks", json={"tarea": TAREA, "publicar": True})).json()["job_id"]
        await e.procesar_siguiente()
        await e.cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "comentario": "no"})
        await e.procesar_siguiente()
        job = (await e.cliente.get(f"/tasks/{job_id}")).json()
        assert job["status"] == "REJECTED"
        assert await e.redis.xlen("informes:publicados") == 0

    asyncio.run(caso())


def test_excepcion_en_el_worker_marca_failed() -> None:
    async def caso() -> None:
        e = Entorno()
        job_id = (await e.cliente.post("/tasks", json={"tarea": TAREA, "simular_falla": "excepcion"})).json()["job_id"]
        await e.procesar_siguiente()  # no propaga la excepción
        job = (await e.cliente.get(f"/tasks/{job_id}")).json()
        assert job["status"] == "FAILED"
        assert "FallaSimulada" in job["error"]
        assert job["finished_at"]

    asyncio.run(caso())


def test_error_dentro_del_grafo_marca_failed() -> None:
    async def caso() -> None:
        # Guion vacío: la síntesis no tiene respuesta -> el LLM lanza una excepción no controlada.
        llm = guion(decidir("FINISH"), decidir("investigador"), *investigacion_ok(), decidir("FINISH"))
        e = Entorno(llm=llm)
        job_id = (await e.cliente.post("/tasks", json={"tarea": TAREA})).json()["job_id"]
        await e.procesar_siguiente()
        job = (await e.cliente.get(f"/tasks/{job_id}")).json()
        assert job["status"] == "FAILED" and job["error"]

    asyncio.run(caso())


def test_timeout_marca_failed() -> None:
    async def caso() -> None:
        e = Entorno(timeout=0.01)

        async def lento(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(1)

        e.worker.grafo.ainvoke = lento  # type: ignore[method-assign]
        job_id = (await e.cliente.post("/tasks", json={"tarea": TAREA})).json()["job_id"]
        await e.procesar_siguiente()
        job = (await e.cliente.get(f"/tasks/{job_id}")).json()
        assert job["status"] == "FAILED" and "Timeout" in job["error"]

    asyncio.run(caso())


def test_cola_confiable_recupera_mensajes_de_un_worker_caido() -> None:
    async def caso() -> None:
        e = Entorno()
        await e.cliente.post("/tasks", json={"tarea": TAREA})
        assert await e.store.tomar("w-test", timeout=0.1)  # el worker lo tomó y "se murió"
        assert await e.redis.llen("cola:test") == 0
        assert await e.store.recuperar_huerfanos("w-test") == 1
        assert await e.redis.llen("cola:test") == 1

    asyncio.run(caso())
