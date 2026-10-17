"""
Worker: saca tareas de la cola de Redis, ejecuta el grafo y actualiza el estado.

    PENDING ──► RUNNING ──┬──► DONE
                          ├──► AWAITING_APPROVAL ──(POST /approve)──► PENDING ──► RUNNING ──► DONE | REJECTED
                          └──► FAILED   (cualquier excepción o timeout)

- Proceso separado de la API (`python -m app.worker`) con N consumidores
  asyncio (`WORKER_CONCURRENCIA`). También se puede embeber en la API con
  `WORKER_EMBEBIDO=true` para desarrollo.
- Todo es async (Redis, LLM, checkpointer). Las herramientas síncronas de
  los agentes (búsqueda en Pinecone/BM25) LangChain las corre en un thread
  pool, así que tampoco frenan el loop.
- Nunca deja un job colgado: cualquier excepción del grafo (o un timeout)
  se captura y el job pasa a FAILED con el error, para que el polling del
  cliente termine.
- La pausa HITL no ocupa al worker: el grafo se detiene en `interrupt()`, el
  checkpoint queda en Redis y el consumidor sigue con la próxima tarea. La
  reanudación puede tomarla cualquier worker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import sys
import time
import traceback
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

import config
from agents.supervisor import PROMPT_VERSION
from app.graph import config_job
from app.jobs import JobStore, ahora
from app.observability import USO_ACTUAL, UsoTokens, contexto_job, trazado
from app.schemas import EstadoJob, Job, UsoJob
from state import AgentState, Contribucion, DecisionHumana, EvaluacionRiesgo, como, estado_inicial

logger = logging.getLogger("orquestador.worker")


class FallaSimulada(RuntimeError):
    """Excepción no controlada a propósito (campo `simular_falla` del request)."""


def _serializar(valor: Any) -> Any:
    if hasattr(valor, "model_dump"):
        return valor.model_dump(mode="json")
    if isinstance(valor, dict):
        return {k: _serializar(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [_serializar(v) for v in valor]
    return valor


def resumen_resultado(estado: dict[str, Any]) -> dict[str, Any]:
    """Lo que la API devuelve del estado final (sin los mensajes internos)."""
    return {
        "respuesta_final": estado.get("respuesta_final", ""),
        "task_completed": estado.get("task_completed", False),
        "requiere_revision_humana": estado.get("requiere_revision_humana", False),
        "motivo_escalamiento": estado.get("motivo_escalamiento"),
        "delegaciones": estado.get("pasos", 0),
        "intentos": estado.get("intentos", {}),
        "decisiones": _serializar(estado.get("decisiones", [])),
        "validaciones": _serializar(estado.get("validaciones", {})),
        "riesgo": _serializar(estado.get("riesgo")),
        "publicacion": _serializar(estado.get("publicacion")),
        "recorrido": [como(Contribucion, c).agente for c in estado.get("contribuciones", [])],
    }


class Worker:
    def __init__(
        self,
        grafo: CompiledStateGraph[AgentState],
        store: JobStore,
        concurrencia: int = config.WORKER_CONCURRENCIA,
        nombre: str | None = None,
        timeout: float = config.JOB_TIMEOUT_SEGUNDOS,
    ) -> None:
        self.grafo = grafo
        self.store = store
        self.concurrencia = concurrencia
        self.nombre = nombre or os.getenv("WORKER_NOMBRE") or socket.gethostname()
        self.timeout = timeout
        self._parar = asyncio.Event()

    def parar(self) -> None:
        self._parar.set()

    async def correr(self) -> None:
        recuperados = await self.store.recuperar_huerfanos(self.nombre)
        if recuperados:
            logger.warning("Se re-encolaron %d mensajes que quedaron sin terminar en una corrida anterior.", recuperados)
        logger.info("Worker '%s' escuchando '%s' con %d consumidores.", self.nombre, self.store.cola, self.concurrencia)
        await asyncio.gather(*(self._consumidor(i) for i in range(self.concurrencia)))

    async def _consumidor(self, indice: int) -> None:
        while not self._parar.is_set():
            try:
                mensaje = await self.store.tomar(self.nombre, timeout=1.0)
            except Exception:  # Redis caído momentáneamente: esperar y seguir
                logger.exception("[c%d] error leyendo la cola", indice)
                await asyncio.sleep(2)
                continue
            if mensaje is None:
                continue
            try:
                await self.procesar(json.loads(mensaje))
            finally:
                await self.store.confirmar(self.nombre, mensaje)

    async def procesar(self, mensaje: dict[str, Any]) -> None:
        """Ejecuta un mensaje de la cola. NUNCA propaga excepciones: las convierte en FAILED."""
        job_id = mensaje["job_id"]
        inicio = time.perf_counter()
        try:
            await asyncio.wait_for(self._ejecutar(mensaje), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.error("Job %s: timeout de %.0f s", job_id, self.timeout)
            await self._marcar_fallido(job_id, f"Timeout: el grafo superó {self.timeout:.0f} s.", inicio)
        except Exception as exc:
            logger.error("Job %s falló: %s: %s", job_id, type(exc).__name__, exc)
            detalle = "".join(traceback.format_exception_only(exc)).strip()
            await self._marcar_fallido(job_id, detalle, inicio)

    async def _marcar_fallido(self, job_id: str, error: str, inicio: float) -> None:
        try:
            job = await self.store.obtener(job_id)
            previa = job.duracion_ejecucion_segundos if job else 0.0
            await self.store.actualizar(
                job_id,
                status=EstadoJob.FAILED,
                error=error,
                finished_at=ahora(),
                duracion_ejecucion_segundos=round(previa + time.perf_counter() - inicio, 3),
            )
        except Exception:  # si ni siquiera se puede escribir en Redis, al menos queda en el log
            logger.exception("No se pudo marcar FAILED el job %s", job_id)

    async def _ejecutar(self, mensaje: dict[str, Any]) -> None:
        job_id, tipo = mensaje["job_id"], mensaje["tipo"]
        job = await self.store.obtener(job_id)
        if job is None:
            logger.warning("Job %s ya no existe (expiró): se descarta el mensaje.", job_id)
            return
        if job.status in (EstadoJob.DONE, EstadoJob.REJECTED, EstadoJob.FAILED):
            logger.warning("Job %s ya está en %s: se ignora el mensaje duplicado.", job_id, job.status)
            return

        inicio = time.perf_counter()
        await self.store.actualizar(
            job_id,
            status=EstadoJob.RUNNING,
            worker=self.nombre,
            started_at=job.started_at or ahora(),
            ejecuciones=job.ejecuciones + 1,
            error=None,
        )
        uso = UsoTokens()
        token_uso = USO_ACTUAL.set(uso)
        try:
            estado = await self._correr_grafo(job=job, tipo=tipo, decision=mensaje.get("decision"), uso=uso)
        finally:
            USO_ACTUAL.reset(token_uso)
            await self._sumar_uso(job, uso)

        duracion = round(job.duracion_ejecucion_segundos + time.perf_counter() - inicio, 3)
        snapshot = estado["snapshot"]
        if snapshot.interrupts:
            pendiente = snapshot.interrupts[0].value
            await self.store.actualizar(
                job_id, status=EstadoJob.AWAITING_APPROVAL, aprobacion_pendiente=pendiente,
                duracion_ejecucion_segundos=duracion,
            )
            logger.info("Job %s PAUSADO para aprobación humana: %s", job_id, "; ".join(pendiente.get("motivos", [])))
            return

        valores = snapshot.values
        aprobacion = como(DecisionHumana, valores.get("aprobacion"))
        riesgo = como(EvaluacionRiesgo, valores.get("riesgo"))
        final = EstadoJob.REJECTED if riesgo and riesgo.critica and aprobacion and not aprobacion.aprobado else EstadoJob.DONE
        await self.store.actualizar(
            job_id,
            status=final,
            resultado=resumen_resultado(valores),
            aprobacion_pendiente=None,
            aprobacion=_serializar(aprobacion),
            finished_at=ahora(),
            duracion_ejecucion_segundos=duracion,
        )
        logger.info("Job %s %s en %.1f s (%d tokens)", job_id, final, duracion, uso.tokens_totales)

    @trazado(
        "job",
        tipo="CHAIN",
        entrada=lambda self, *, job, tipo, decision, uso: {
            "job_id": job.job_id, "segmento": tipo, "tarea": job.tarea, "publicar": job.publicar, "decision": decision,
        },
        salida=lambda r: {
            "pausado_para_aprobacion": bool(r["snapshot"].interrupts),
            "respuesta_final": r["snapshot"].values.get("respuesta_final", ""),
            "tokens": r["tokens"],
        },
    )
    async def _correr_grafo(self, *, job: Job, tipo: str, decision: dict[str, Any] | None, uso: UsoTokens) -> dict[str, Any]:
        """Un "segmento" de ejecución: la corrida inicial o la reanudación tras el HITL.

        Es el span raíz de la traza en Phoenix; con `session.id = job_id` los
        dos segmentos de un mismo job quedan agrupados en la misma sesión.
        """
        metadata = {"job_id": job.job_id, "segmento": tipo, "prompt_supervisor": PROMPT_VERSION,
                    "modelos": config.MODELOS_POR_ROL, "publicar": job.publicar}
        tags = ["api", f"segmento:{tipo}", PROMPT_VERSION]
        cfg = config_job(job.job_id, callbacks=[uso.handler], run_name=f"orquestador:{tipo}", tags=tags, metadata=metadata)

        with contexto_job(job.job_id, metadata=metadata, tags=tags):
            if job.simular_falla == "excepcion" and config.PERMITIR_FALLAS_SIMULADAS:
                raise FallaSimulada("Falla simulada: el worker lanzó una excepción no controlada.")
            if tipo == "run":
                entrada: Any = estado_inicial(job.tarea, publicar=job.publicar)
            elif tipo == "resume":
                entrada = Command(resume=decision)
            else:
                raise ValueError(f"Tipo de mensaje desconocido: {tipo!r}")
            await self.grafo.ainvoke(entrada, cfg)
            snapshot = await self.grafo.aget_state(cfg)
        return {"snapshot": snapshot, "tokens": uso.tokens_totales, "costo_estimado_usd": uso.costo_estimado_usd}

    async def _sumar_uso(self, job: Job, uso: UsoTokens) -> None:
        """Acumula el uso de este segmento con el de los anteriores (corrida + reanudación)."""
        previo = job.uso
        entrada = previo.tokens_entrada + uso.tokens_entrada
        salida = previo.tokens_salida + uso.tokens_salida
        total = UsoJob(
            tokens_entrada=entrada,
            tokens_salida=salida,
            tokens_totales=entrada + salida,
            costo_estimado_usd=round(previo.costo_estimado_usd + uso.costo_estimado_usd, 6),
            modelos=sorted(set(previo.modelos) | set(uso.modelos)),
        )
        await self.store.actualizar(job.job_id, uso=total)


# --- Proceso independiente: python -m app.worker ------------------------------

async def _main() -> None:
    from redis.asyncio import Redis

    from app.graph import checkpointer_redis, compilar
    from app.hitl import PublicadorRedis
    from app.observability import apagar_observabilidad, configurar_observabilidad
    from llm import crear_llm, crear_llms_por_rol
    from rag.backend import obtener_buscador

    configurar_observabilidad("worker")
    redis = Redis.from_url(config.REDIS_URL, decode_responses=True)
    await redis.ping()
    # Cargar el RAG (descarga el corpus de Pinecone para BM25) fuera del event loop.
    buscador = await asyncio.to_thread(obtener_buscador)
    logger.info("LLM=%s %s | RAG=%s | Redis=%s", config.LLM_PROVIDER, config.MODELOS_POR_ROL, buscador.backend, config.REDIS_URL)

    async with checkpointer_redis() as checkpointer:
        grafo = compilar(crear_llm(), checkpointer=checkpointer, publicador=PublicadorRedis(redis), llms_por_rol=crear_llms_por_rol())
        worker = Worker(grafo, JobStore(redis))
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, worker.parar)
            except NotImplementedError:  # Windows: Ctrl+C llega como KeyboardInterrupt
                pass
        try:
            await worker.correr()
        finally:
            await redis.aclose()
            apagar_observabilidad()


def configurar_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for ruidoso in ("httpx", "httpcore", "urllib3", "google_genai", "pinecone"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    configurar_logging()
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("Worker detenido.")
