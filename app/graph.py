"""
Grafo del orquestador multi-agente del Módulo 6 + Human-in-the-loop + checkpointer de Redis.

    START ─► supervisor ─┬─► investigador ─┐
               ▲         ├─► analista ─────┴─► validador ─┐
               │         └─► sintesis                     │
               └──────────────────│───────────────────────┘
                                  ▼
                          evaluar_riesgo ─┬─(crítica)─► aprobacion_humana ⏸ ─┬─► publicar ─► END
                                          └─────────────► END                └─► END

La parte de arriba es el orquestador del M6 sin cambios (supervisor
jerárquico, especialistas con RAG y calculadora, validador determinístico).
Lo nuevo es la cola del grafo (ver `app/hitl.py`) y la persistencia:

- `AsyncRedisSaver` guarda un checkpoint después de cada super-step, con
  `thread_id = job_id`. Eso permite (1) pausar en `interrupt()` y reanudar
  horas después, incluso en OTRO proceso worker, y (2) inspeccionar el
  historial de cualquier job (`GET /tasks/{id}/checkpoints`).
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

import config
from agents.analyst_agent import crear_nodo_analista
from agents.common import POLITICA_REINTENTO_LLM
from agents.research_agent import crear_nodo_investigador
from agents.supervisor import crear_nodo_supervisor
from agents.synthesis import crear_nodo_sintesis
from agents.validator import nodo_validador
from app.hitl import (
    Publicador,
    PublicadorEnMemoria,
    crear_nodo_evaluar_riesgo,
    crear_nodo_publicar,
    nodo_aprobacion_humana,
    ruta_aprobacion,
    ruta_riesgo,
)
from state import PARCELAS, AgentState


class ViolacionDeParcela(RuntimeError):
    """Un nodo intentó escribir una clave del estado que no le pertenece."""


def con_parcela(nombre: str, nodo: Callable[..., Any]) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Envuelve un nodo y hace cumplir el aislamiento de escritura del M6: si
    devuelve claves fuera de `PARCELAS[nombre]`, falla en vez de pisar el estado ajeno.

    Si el nodo original recibe `config` (p. ej. `publicar`, que necesita el
    thread_id), se lo pasa.
    """
    permitidas = PARCELAS[nombre]
    quiere_config = len(inspect.signature(nodo).parameters) >= 2

    async def envuelto(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        resultado = nodo(state, config) if quiere_config else nodo(state)
        if inspect.isawaitable(resultado):
            resultado = await resultado
        ajenas = set(resultado) - permitidas
        if ajenas:
            raise ViolacionDeParcela(f"El nodo '{nombre}' intentó escribir {sorted(ajenas)}, fuera de su parcela.")
        return resultado

    # Con `from __future__ import annotations` la anotación sería el string
    # "RunnableConfig | None" y LangGraph no inyectaría el config: se pone el tipo real.
    envuelto.__annotations__ = {"state": AgentState, "config": RunnableConfig, "return": dict[str, Any]}
    envuelto.__name__ = getattr(nodo, "__name__", nombre)
    envuelto.__doc__ = nodo.__doc__
    return envuelto


def enrutar(state: AgentState) -> Literal["investigador", "analista", "sintesis"]:
    """Arista condicional del supervisor: mapea su decisión al nombre del nodo."""
    destino = state.get("next_agent")
    if destino in ("investigador", "analista", "sintesis"):
        return destino
    return "sintesis"  # nunca debería pasar; ante la duda, cerrar


def construir_grafo(
    llm: BaseChatModel,
    publicador: Publicador | None = None,
    llms_por_rol: dict[str, BaseChatModel] | None = None,
) -> StateGraph[AgentState]:
    """Arma el StateGraph (sin compilar). El LLM y el publicador se inyectan
    para poder testear con un modelo guionado y sin Redis.

    `llms_por_rol` permite un modelo distinto por rol (supervisor, investigador,
    analista, sintesis); los roles que falten usan `llm`.
    """
    roles = llms_por_rol or {}

    def modelo(rol: str) -> BaseChatModel:
        return roles.get(rol, llm)

    grafo = StateGraph(AgentState)

    # --- Orquestador del Módulo 6 ---
    grafo.add_node("supervisor", con_parcela("supervisor", crear_nodo_supervisor(modelo("supervisor"))), retry_policy=POLITICA_REINTENTO_LLM)
    grafo.add_node("investigador", con_parcela("investigador", crear_nodo_investigador(modelo("investigador"))), retry_policy=POLITICA_REINTENTO_LLM)
    grafo.add_node("analista", con_parcela("analista", crear_nodo_analista(modelo("analista"))), retry_policy=POLITICA_REINTENTO_LLM)
    grafo.add_node("validador", con_parcela("validador", nodo_validador))
    grafo.add_node("sintesis", con_parcela("sintesis", crear_nodo_sintesis(modelo("sintesis"))), retry_policy=POLITICA_REINTENTO_LLM)

    # --- Human-in-the-loop (Módulo 7) ---
    grafo.add_node("evaluar_riesgo", con_parcela("evaluar_riesgo", crear_nodo_evaluar_riesgo()))
    grafo.add_node("aprobacion_humana", con_parcela("aprobacion_humana", nodo_aprobacion_humana))
    grafo.add_node("publicar", con_parcela("publicar", crear_nodo_publicar(publicador or PublicadorEnMemoria())))

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges(
        "supervisor",
        enrutar,
        {"investigador": "investigador", "analista": "analista", "sintesis": "sintesis"},
    )
    grafo.add_edge("investigador", "validador")
    grafo.add_edge("analista", "validador")
    grafo.add_edge("validador", "supervisor")
    grafo.add_edge("sintesis", "evaluar_riesgo")
    grafo.add_conditional_edges("evaluar_riesgo", ruta_riesgo, {"aprobacion_humana": "aprobacion_humana", END: END})
    grafo.add_conditional_edges("aprobacion_humana", ruta_aprobacion, {"publicar": "publicar", END: END})
    grafo.add_edge("publicar", END)
    return grafo


def compilar(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    publicador: Publicador | None = None,
    llms_por_rol: dict[str, BaseChatModel] | None = None,
) -> CompiledStateGraph[AgentState]:
    """Sin checkpointer el grafo corre igual que en el M6 (pero no puede pausarse)."""
    return construir_grafo(llm, publicador, llms_por_rol).compile(checkpointer=checkpointer)


@contextlib.asynccontextmanager
async def checkpointer_redis(redis_url: str = config.REDIS_URL) -> AsyncIterator[BaseCheckpointSaver]:
    """`AsyncRedisSaver` de `langgraph-checkpoint-redis` (requiere Redis 8 o Redis Stack:
    usa los módulos RedisJSON y RediSearch). Crea los índices la primera vez."""
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    ttl = {"default_ttl": config.CHECKPOINT_TTL_MINUTOS, "refresh_on_read": True}
    async with AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl) as saver:
        # Por seguridad, LangGraph solo reconstruye desde JSON las clases que se
        # permiten EXPLÍCITAMENTE (si no, los modelos del estado vuelven como un
        # dict "lc constructor" al reanudar). Se habilitan solo los de state.py.
        saver.serde = JsonPlusRedisSerializer(allowed_json_modules=modelos_del_estado())
        await saver.asetup()
        yield saver


def modelos_del_estado() -> list[tuple[str, str]]:
    """Símbolos exactos (módulo, clase) de los modelos Pydantic que viajan en el estado."""
    import state
    from pydantic import BaseModel

    return [
        ("state", nombre)
        for nombre, obj in vars(state).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == "state"
    ]


def config_job(job_id: str, recursion_limit: int = config.RECURSION_LIMIT, **extra: Any) -> RunnableConfig:
    """`thread_id = job_id`: todos los checkpoints del job quedan bajo ese hilo."""
    return {"configurable": {"thread_id": job_id}, "recursion_limit": recursion_limit, **extra}


@functools.cache
def _llm_sin_red() -> BaseChatModel:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    class _SinLLM(GenericFakeChatModel):
        def bind_tools(self, *args: Any, **kwargs: Any) -> "_SinLLM":
            return self

    return _SinLLM(messages=iter([]))


def dibujar_grafo(destino_mmd: str, destino_png: str | None = None) -> None:
    """Exporta el diagrama (para dibujar no hace falta un LLM real)."""
    from pathlib import Path

    grafo = compilar(_llm_sin_red()).get_graph()
    Path(destino_mmd).write_text(grafo.draw_mermaid(), encoding="utf-8")
    if destino_png:
        Path(destino_png).write_bytes(grafo.draw_mermaid_png())
