"""
API REST del orquestador multi-agente (FastAPI).

El endpoint NUNCA ejecuta el agente: registra el job en Redis, lo encola y
devuelve el `job_id` en milisegundos (202 Accepted). Un worker lo ejecuta en
segundo plano y el cliente consulta el estado por polling.

    POST /tasks                    -> 202 {job_id, status: PENDING}
    GET  /tasks/{job_id}           -> estado, resultado, aprobación pendiente, tokens/costo, error
    POST /tasks/{job_id}/approve   -> reanuda un job pausado por el HITL (ver app/hitl.py)
    GET  /tasks                    -> últimos jobs (filtro opcional por estado)
    GET  /tasks/{job_id}/checkpoints -> historial de checkpoints de LangGraph guardados en Redis
    GET  /informes                 -> informes publicados (el efecto secundario aprobado)
    GET  /health                   -> Redis, cola y observabilidad

Uso:
    uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, status
from redis.asyncio import Redis

import config
from app import hitl
from app.jobs import JobStore, ahora
from app.observability import apagar_observabilidad, configurar_observabilidad
from app.schemas import EstadoJob, Job, TareaEncolada, TareaRequest

logger = logging.getLogger("orquestador.api")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.observabilidad = configurar_observabilidad("api")
    redis = Redis.from_url(config.REDIS_URL, decode_responses=True)
    try:
        await redis.ping()
    except Exception as exc:
        raise RuntimeError(f"No se pudo conectar a Redis en {config.REDIS_URL}: {exc}. ¿Está levantado?") from exc
    app.state.redis = redis
    app.state.store = JobStore(redis)

    async with contextlib.AsyncExitStack() as pila:
        # El checkpointer en la API se usa solo para LEER el historial de los jobs.
        from app.graph import checkpointer_redis

        app.state.checkpointer = await pila.enter_async_context(checkpointer_redis())
        tarea_worker = None
        if config.WORKER_EMBEBIDO:
            tarea_worker = await _arrancar_worker_embebido(app, redis)
        try:
            yield
        finally:
            if tarea_worker is not None:
                app.state.worker.parar()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait_for(tarea_worker, timeout=10)
            await redis.aclose()
            apagar_observabilidad()


async def _arrancar_worker_embebido(app: FastAPI, redis: Redis) -> asyncio.Task[None]:
    """Modo simple: los consumidores corren como tareas asyncio dentro de la API."""
    from app.graph import compilar
    from app.worker import Worker
    from llm import crear_llm, crear_llms_por_rol
    from rag.backend import obtener_buscador

    await asyncio.to_thread(obtener_buscador)
    grafo = compilar(
        crear_llm(), checkpointer=app.state.checkpointer, publicador=hitl.PublicadorRedis(redis), llms_por_rol=crear_llms_por_rol()
    )
    app.state.worker = Worker(grafo, app.state.store, nombre="api-embebido")
    logger.info("Worker embebido en la API (%d consumidores).", config.WORKER_CONCURRENCIA)
    return asyncio.create_task(app.state.worker.correr())


app = FastAPI(
    title="Orquestador multi-agente — API de producción",
    version="1.0.0",
    description="Pre-entrega 7: FastAPI asíncrona + Redis (estado y checkpoints) + Phoenix + Human-in-the-loop.",
    lifespan=lifespan,
)
app.include_router(hitl.router)


def _store(request: Request) -> JobStore:
    return request.app.state.store


@app.post("/tasks", status_code=status.HTTP_202_ACCEPTED, response_model=TareaEncolada, tags=["tareas"])
async def crear_tarea(cuerpo: TareaRequest, request: Request) -> TareaEncolada:
    """Registra el job (PENDING) y lo encola. No espera al agente."""
    if cuerpo.simular_falla and not config.PERMITIR_FALLAS_SIMULADAS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Las fallas simuladas están deshabilitadas.")
    momento = ahora()
    job = Job(
        job_id=uuid.uuid4().hex,
        status=EstadoJob.PENDING,
        tarea=cuerpo.tarea,
        publicar=cuerpo.publicar,
        simular_falla=cuerpo.simular_falla,
        created_at=momento,
        updated_at=momento,
    )
    await _store(request).crear_y_encolar(job)
    return TareaEncolada(
        job_id=job.job_id,
        status=job.status,
        links={"estado": f"/tasks/{job.job_id}", "aprobar": f"/tasks/{job.job_id}/approve"},
    )


@app.get("/tasks/{job_id}", response_model=Job, tags=["tareas"])
async def estado_tarea(job_id: str, request: Request) -> Job:
    job = await _store(request).obtener(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}.")
    return job


@app.get("/tasks", response_model=list[Job], tags=["tareas"])
async def listar_tareas(
    request: Request,
    estado: EstadoJob | None = Query(default=None, alias="status"),
    limite: int = Query(default=20, ge=1, le=200),
) -> list[Job]:
    return await _store(request).listar(limite=limite, estado=estado)


@app.get("/tasks/{job_id}/checkpoints", tags=["persistencia"])
async def checkpoints_tarea(job_id: str, request: Request, limite: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    """Historial de checkpoints de LangGraph del hilo `thread_id=job_id`, leído de Redis.

    Muestra el paso, qué nodo escribió y cuáles iban a correr después: es lo
    que permite reanudar un job pausado desde cualquier worker.
    """
    if await _store(request).obtener(job_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}.")
    saver = request.app.state.checkpointer
    items = []
    async for tupla in saver.alist({"configurable": {"thread_id": job_id}}, limit=limite):
        meta = tupla.metadata or {}
        valores = tupla.checkpoint.get("channel_values", {})
        items.append({
            "checkpoint_id": tupla.checkpoint["id"],
            "ts": tupla.checkpoint.get("ts"),
            "paso": meta.get("step"),
            "origen": meta.get("source"),
            "canales_actualizados": sorted(tupla.checkpoint.get("updated_channels") or []),
            "escrituras_pendientes": len(tupla.pending_writes or []),
            "respuesta_final_lista": bool(valores.get("respuesta_final")),
        })
    return {"job_id": job_id, "thread_id": job_id, "total": len(items), "checkpoints": items}


@app.get("/informes", tags=["human-in-the-loop"])
async def informes_publicados(request: Request, limite: int = Query(default=10, ge=1, le=100)) -> list[dict[str, Any]]:
    """Informes que un humano aprobó publicar (Stream de Redis)."""
    entradas = await request.app.state.redis.xrevrange(config.STREAM_PUBLICACIONES, count=limite)
    return [{"id": id_, **campos} for id_, campos in entradas]


@app.get("/health", tags=["operación"])
async def salud(request: Request) -> dict[str, Any]:
    try:
        redis_ok = bool(await request.app.state.redis.ping())
    except Exception:
        redis_ok = False
    cola = await _store(request).largo_cola() if redis_ok else {}
    return {
        "status": "ok" if redis_ok else "degradado",
        "redis": redis_ok,
        "cola": cola,
        "observabilidad": request.app.state.observabilidad,
        "worker_embebido": config.WORKER_EMBEBIDO,
    }
