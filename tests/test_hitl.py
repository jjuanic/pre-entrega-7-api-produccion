"""Human-in-the-loop: pausa con interrupt(), reanudación, rechazo y criterios de criticidad."""

from __future__ import annotations

import asyncio

import fakeredis
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import compilar, config_job
from app.hitl import PublicadorEnMemoria, PublicadorRedis, crear_nodo_evaluar_riesgo
from app.observability import USO_ACTUAL, UsoTokens
from conftest import analisis, decidir, guion, investigacion_ok
from state import estado_inicial

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


def test_publicar_pausa_hasta_aprobacion_y_despues_publica() -> None:
    async def caso() -> None:
        publicador = PublicadorEnMemoria()
        app = compilar(llm_flujo_completo(), checkpointer=InMemorySaver(), publicador=publicador)
        cfg = config_job("job-1")

        resultado = await app.ainvoke(estado_inicial(TAREA, publicar=True), cfg)
        # El grafo se detuvo ANTES del efecto secundario.
        assert "__interrupt__" in resultado
        snap = await app.aget_state(cfg)
        assert snap.next == ("aprobacion_humana",)
        pendiente = snap.interrupts[0].value
        assert pendiente["accion"] == "publicar_informe"
        assert any("EFECTO SECUNDARIO" in m for m in pendiente["motivos"])
        assert pendiente["borrador"].startswith("Usa 12 hilos")
        assert publicador.publicados == {}

        # Señal externa: aprobado. Se reanuda desde el checkpoint (sin volver a llamar al LLM).
        await app.ainvoke(Command(resume={"aprobado": True, "revisor": "ana", "comentario": "ok"}), cfg)
        final = await app.aget_state(cfg)
        assert final.next == ()
        assert final.values["aprobacion"].aprobado
        assert final.values["publicacion"].id_publicacion == "job-1"
        assert publicador.publicados["job-1"]["aprobado_por"] == "ana"

    asyncio.run(caso())


def test_rechazo_humano_no_ejecuta_el_efecto_secundario() -> None:
    async def caso() -> None:
        publicador = PublicadorEnMemoria()
        app = compilar(llm_flujo_completo(), checkpointer=InMemorySaver(), publicador=publicador)
        cfg = config_job("job-2")
        await app.ainvoke(estado_inicial(TAREA, publicar=True), cfg)
        await app.ainvoke(Command(resume={"aprobado": False, "revisor": "ana", "comentario": "falta contexto"}), cfg)
        final = await app.aget_state(cfg)
        assert final.next == ()
        assert not final.values["aprobacion"].aprobado
        assert final.values["publicacion"] is None
        assert publicador.publicados == {}

    asyncio.run(caso())


def test_tarea_no_critica_termina_sin_pausa() -> None:
    async def caso() -> None:
        app = compilar(llm_flujo_completo(), checkpointer=InMemorySaver())
        cfg = config_job("job-3")
        resultado = await app.ainvoke(estado_inicial(TAREA), cfg)
        assert "__interrupt__" not in resultado
        assert resultado["riesgo"].critica is False
        assert resultado["respuesta_final"].startswith("Usa 12 hilos")

    asyncio.run(caso())


def test_escalamiento_del_supervisor_es_critico() -> None:
    nodo = crear_nodo_evaluar_riesgo()
    estado = estado_inicial(TAREA) | {"requiere_revision_humana": True, "motivo_escalamiento": "analista agotó sus 3 intentos"}
    riesgo = asyncio.run(nodo(estado))["riesgo"]
    assert riesgo.critica and riesgo.accion == "entregar_respuesta"
    assert "ESCALAMIENTO" in riesgo.motivos[0]


def test_costo_alto_es_critico() -> None:
    uso = UsoTokens()
    uso.handler.usage_metadata = {"modelo-x": {"input_tokens": 9_000, "output_tokens": 2_000, "total_tokens": 11_000}}
    token = USO_ACTUAL.set(uso)
    try:
        riesgo = asyncio.run(crear_nodo_evaluar_riesgo(max_tokens=10_000, max_costo_usd=1.0)(estado_inicial(TAREA)))["riesgo"]
    finally:
        USO_ACTUAL.reset(token)
    assert riesgo.critica and riesgo.tokens_totales == 11_000
    assert any("COSTO ALTO" in m and "11000" in m for m in riesgo.motivos)


def test_publicador_redis_es_idempotente() -> None:
    async def caso() -> None:
        redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        publicador = PublicadorRedis(redis, stream="informes:test")
        primera = await publicador("job-9", {"informe": "texto"})
        segunda = await publicador("job-9", {"informe": "texto"})  # p. ej. un reintento
        assert not primera.duplicada and segunda.duplicada
        assert segunda.id_publicacion == primera.id_publicacion
        assert await redis.xlen("informes:test") == 1

    asyncio.run(caso())
