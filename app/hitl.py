"""
Human-in-the-loop: pausa obligatoria antes de las acciones CRÍTICAS.

Flujo dentro del grafo (después de la síntesis del M6):

    sintesis ─► evaluar_riesgo ─┬─(crítica)──► aprobacion_humana ─┬─(aprobada + publicar)─► publicar ─► END
                                │               ⏸ interrupt()     └─(rechazada / solo entrega)──────► END
                                └─(no crítica)───────────────────────────────────────────────────────► END

Una tarea es CRÍTICA si se cumple al menos una de estas condiciones:

1. EFECTO SECUNDARIO: el usuario pidió `publicar=true`. El nodo `publicar`
   escribe el informe en un canal compartido (un Stream de Redis que leen
   otros sistemas). Es irreversible, así que no se ejecuta sin un humano.
2. ESCALAMIENTO: el supervisor del M6 marcó `requiere_revision_humana`
   (un especialista agotó sus intentos con output inválido).
3. COSTO ALTO: la corrida superó HITL_MAX_TOKENS o HITL_MAX_COSTO_USD.

Por qué la evaluación está en un nodo separado de la pausa: cuando un grafo
se reanuda con `Command(resume=...)`, LangGraph vuelve a ejecutar DESDE EL
PRINCIPIO el nodo que llamó a `interrupt()`. Si ese nodo recalculara el
riesgo (por ejemplo, con el contador de tokens del nuevo proceso, que arranca
en cero), la decisión podría cambiar entre la pausa y la reanudación. Por eso
`evaluar_riesgo` escribe su veredicto en el estado (queda en el checkpoint de
Redis) y `aprobacion_humana` solo hace `interrupt()` y lee la respuesta.

Este módulo también expone el endpoint `POST /tasks/{id}/approve`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol

from fastapi import APIRouter, HTTPException, Request, status
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt

import config
from app.observability import USO_ACTUAL, trazado
from app.schemas import AprobacionRequest, EstadoJob, TareaEncolada
from state import AgentState, DecisionHumana, EvaluacionRiesgo, Publicacion, como

logger = logging.getLogger("orquestador.hitl")


# --- Nodo 1: evaluar si la tarea es crítica --------------------------------

def crear_nodo_evaluar_riesgo(
    max_tokens: int = config.HITL_MAX_TOKENS,
    max_costo_usd: float = config.HITL_MAX_COSTO_USD,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def nodo_evaluar_riesgo(state: AgentState) -> dict[str, Any]:
        uso = USO_ACTUAL.get()
        tokens = uso.tokens_totales if uso else 0
        costo = uso.costo_estimado_usd if uso else 0.0

        motivos: list[str] = []
        if state.get("publicar"):
            motivos.append(
                f"EFECTO SECUNDARIO: la tarea pide publicar el informe en '{config.STREAM_PUBLICACIONES}' "
                "(canal compartido, no se puede deshacer)."
            )
        if state.get("requiere_revision_humana"):
            motivos.append(f"ESCALAMIENTO: el supervisor pidió revisión humana ({state.get('motivo_escalamiento')}).")
        if tokens > max_tokens:
            motivos.append(f"COSTO ALTO: la corrida usó {tokens} tokens (umbral {max_tokens}).")
        if costo > max_costo_usd:
            motivos.append(f"COSTO ALTO: costo estimado USD {costo:.4f} (umbral USD {max_costo_usd:.4f}).")

        riesgo = EvaluacionRiesgo(
            critica=bool(motivos),
            motivos=motivos,
            accion="publicar_informe" if state.get("publicar") else "entregar_respuesta",
            tokens_totales=tokens,
            costo_estimado_usd=costo,
        )
        return {"riesgo": riesgo}

    return nodo_evaluar_riesgo


def ruta_riesgo(state: AgentState) -> Literal["aprobacion_humana", "__end__"]:
    riesgo = como(EvaluacionRiesgo, state.get("riesgo"))
    return "aprobacion_humana" if riesgo and riesgo.critica else END


# --- Nodo 2: la pausa ---------------------------------------------------

def nodo_aprobacion_humana(state: AgentState) -> dict[str, Any]:
    """Detiene el grafo hasta que llegue una decisión externa.

    La primera vez, `interrupt()` corta la ejecución: el checkpoint queda en
    Redis y el worker se libera (no hay ningún hilo esperando). Cuando el
    revisor llama a `/approve`, el worker reanuda con `Command(resume=...)` y
    `interrupt()` devuelve esa decisión.
    """
    riesgo = como(EvaluacionRiesgo, state["riesgo"])
    assert riesgo is not None  # solo se llega acá si evaluar_riesgo la marcó crítica
    valor = interrupt({
        "tipo": "aprobacion_requerida",
        "accion": riesgo.accion,
        "motivos": riesgo.motivos,
        "borrador": state.get("respuesta_final", ""),
        "tokens_totales": riesgo.tokens_totales,
        "costo_estimado_usd": riesgo.costo_estimado_usd,
        "pregunta": "¿Aprobás esta acción? Respondé con POST /tasks/{id}/approve {\"aprobado\": true|false}.",
    })
    return {"aprobacion": DecisionHumana.model_validate(valor)}


def ruta_aprobacion(state: AgentState) -> Literal["publicar", "__end__"]:
    decision = como(DecisionHumana, state.get("aprobacion"))
    if decision and decision.aprobado and state.get("publicar"):
        return "publicar"
    return END


# --- Nodo 3: la acción con efectos secundarios ----------------------------

class Publicador(Protocol):
    async def __call__(self, job_id: str, informe: dict[str, Any]) -> Publicacion: ...


class PublicadorEnMemoria:
    """Para tests y para dibujar el grafo sin Redis."""

    def __init__(self) -> None:
        self.publicados: dict[str, dict[str, Any]] = {}

    async def __call__(self, job_id: str, informe: dict[str, Any]) -> Publicacion:
        duplicada = job_id in self.publicados
        self.publicados.setdefault(job_id, informe)
        return Publicacion(destino="memoria", id_publicacion=job_id, duplicada=duplicada)


class PublicadorRedis:
    """Publica el informe aprobado en un Stream de Redis.

    Idempotente: `SET publicacion:{job_id} NX` antes del XADD. Si el nodo se
    reintenta (o el worker se cae justo después de publicar y el job se
    reanuda), el informe no se publica dos veces.
    """

    def __init__(self, redis: Any, stream: str = config.STREAM_PUBLICACIONES) -> None:
        self.redis = redis
        self.stream = stream

    @trazado("publicar_informe", tipo="TOOL")
    async def __call__(self, job_id: str, informe: dict[str, Any]) -> Publicacion:
        clave = f"publicacion:{job_id}"
        if not await self.redis.set(clave, "en-curso", nx=True, ex=config.JOB_TTL_SEGUNDOS):
            previo = await self.redis.get(clave)
            return Publicacion(destino=f"redis-stream:{self.stream}", id_publicacion=str(previo), duplicada=True)
        campos = {"job_id": job_id, **{k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in informe.items()}}
        id_mensaje = await self.redis.xadd(self.stream, campos, maxlen=10_000, approximate=True)
        await self.redis.set(clave, id_mensaje, ex=config.JOB_TTL_SEGUNDOS)
        logger.info("Informe del job %s publicado en %s (%s)", job_id, self.stream, id_mensaje)
        return Publicacion(destino=f"redis-stream:{self.stream}", id_publicacion=str(id_mensaje))


def crear_nodo_publicar(publicador: Publicador) -> Callable[..., Awaitable[dict[str, Any]]]:
    async def nodo_publicar(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        job_id = config["configurable"]["thread_id"]
        decision = como(DecisionHumana, state["aprobacion"])
        assert decision is not None and decision.aprobado  # garantizado por ruta_aprobacion
        publicacion = await publicador(
            job_id,
            {
                "tarea": state["tarea"],
                "informe": state["respuesta_final"],
                "aprobado_por": decision.revisor,
                "comentario": decision.comentario,
                "aprobado_en": decision.momento,
            },
        )
        return {"publicacion": publicacion}

    return nodo_publicar


# --- Endpoint de aprobación --------------------------------------------------

router = APIRouter(tags=["human-in-the-loop"])


@router.post(
    "/tasks/{job_id}/approve",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TareaEncolada,
    summary="Aprobar o rechazar una tarea pausada (reanuda el grafo)",
)
async def aprobar_tarea(job_id: str, cuerpo: AprobacionRequest, request: Request) -> TareaEncolada:
    """Solo acepta jobs en `AWAITING_APPROVAL`. Cambia el estado a PENDING y
    encola la reanudación de forma atómica: dos aprobaciones simultáneas no
    pueden reanudar el mismo grafo dos veces (la segunda recibe 409)."""
    store = request.app.state.store
    decision = DecisionHumana(aprobado=cuerpo.aprobado, revisor=cuerpo.revisor, comentario=cuerpo.comentario)
    resultado = await store.encolar_reanudacion(job_id, decision.model_dump(mode="json"))
    if resultado is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}.")
    if resultado is not True:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"El job está en estado {resultado}: solo se puede aprobar en {EstadoJob.AWAITING_APPROVAL.value}.",
        )
    return TareaEncolada(job_id=job_id, status=EstadoJob.PENDING, links={"estado": f"/tasks/{job_id}"})
