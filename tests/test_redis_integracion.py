"""Integración con un Redis REAL (Redis 8 / Stack). Se saltea si no hay uno en REDIS_URL.

Reproduce lo que pasa en producción: un worker corre el grafo hasta la pausa
HITL y OTRO worker (otro grafo compilado, otra conexión al checkpointer)
lo reanuda a partir del checkpoint guardado en Redis.

    docker compose up -d redis
    python -m pytest tests/test_redis_integracion.py -v
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

import config
from app.graph import checkpointer_redis, compilar, config_job
from app.hitl import PublicadorEnMemoria
from app.worker import resumen_resultado
from conftest import analisis, decidir, guion, investigacion_ok
from state import estado_inicial


def _redis_disponible() -> bool:
    async def ping() -> bool:
        from redis.asyncio import Redis

        cliente = Redis.from_url(config.REDIS_URL)
        try:
            modulos = {m[b"name"] for m in await cliente.module_list()}
            return b"search" in modulos or b"ReJSON" in modulos
        finally:
            await cliente.aclose()

    try:
        return asyncio.run(asyncio.wait_for(ping(), timeout=2))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _redis_disponible(), reason=f"No hay Redis 8/Stack en {config.REDIS_URL}")

TAREA = "¿Cuántos hilos usa ThreadPoolExecutor por defecto en una máquina de 8 núcleos?"


def test_pausa_en_un_worker_y_reanuda_en_otro_desde_redis() -> None:
    async def caso() -> None:
        job_id = f"test-{uuid.uuid4().hex}"
        cfg = config_job(job_id)
        llm = guion(
            decidir("investigador", instruccion="Buscá el max_workers por defecto."),
            *investigacion_ok(),
            decidir("analista", instruccion="Calculá min(32, 8 + 4)."),
            *analisis(),
            decidir("FINISH", razon="Todo válido."),
            AIMessage(content="Usa 12 hilos (concurrent_futures.md)."),
        )
        # "Worker A": corre hasta la pausa y se va.
        async with checkpointer_redis() as saver_a:
            grafo_a = compilar(llm, checkpointer=saver_a)
            await grafo_a.ainvoke(estado_inicial(TAREA, publicar=True), cfg)
            snap = await grafo_a.aget_state(cfg)
            assert snap.next == ("aprobacion_humana",)

        # "Worker B": proceso nuevo, sin nada en memoria; solo tiene Redis.
        publicador = PublicadorEnMemoria()
        async with checkpointer_redis() as saver_b:
            grafo_b = compilar(guion(), checkpointer=saver_b, publicador=publicador)  # LLM vacío: no debe llamarse
            await grafo_b.ainvoke(Command(resume={"aprobado": True, "revisor": "test"}), cfg)
            final = await grafo_b.aget_state(cfg)
            assert final.next == ()
            assert publicador.publicados[job_id]["aprobado_por"] == "test"
            # Los valores leídos del checkpoint (dicts en JSON) se pueden resumir igual.
            resumen = resumen_resultado(final.values)
            assert resumen["riesgo"]["critica"] is True
            assert resumen["publicacion"]["id_publicacion"] == job_id
            assert resumen["recorrido"][-1] == "sintesis"
            historial = [c async for c in saver_b.alist(cfg)]
            assert len(historial) > 8

    asyncio.run(caso())
