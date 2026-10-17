"""
Ejecución de una tarea con log del flujo de delegación.

Usa `astream(stream_mode="updates")` para ver qué devolvió cada nodo a medida
que corre, y loguea exactamente quién decidió qué y por qué:

    [supervisor]   paso 1 -> investigador | razón | instrucción
    [investigador] intento 1: 3 hallazgos de [...]
    [validador]    investigador (intento 1): VÁLIDO
    ...
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

import config
from agents.supervisor import PROMPT_VERSION
from app.graph import config_job
from state import AgentState, estado_inicial

logger = logging.getLogger("orquestador")


def _loguear(nodo: str, update: dict[str, Any]) -> None:
    for d in update.get("decisiones", []):
        regla = f" [guarda: {d.regla_aplicada}]" if d.regla_aplicada else ""
        logger.info("  [supervisor]   paso %d: LLM propuso %s -> ruteo a %s%s", d.paso, d.propuesta_llm, d.destino.upper(), regla)
        logger.info("                 razón: %s", d.razon)
        if d.instruccion:
            logger.info("                 instrucción: %s", d.instruccion)
    if update.get("requiere_revision_humana"):
        logger.warning("  [supervisor]   ESCALA A REVISIÓN HUMANA: %s", update.get("motivo_escalamiento"))
    if nodo == "sintesis":
        logger.info("  [sintesis]     respuesta final generada (%d caracteres)", len(update.get("respuesta_final", "")))
        return
    for c in update.get("contribuciones", []):
        logger.info("  [%-12s] intento %d: %s", c.agente, c.intento, c.detalle)


def _serializar(valor: Any) -> Any:
    if hasattr(valor, "model_dump"):
        return valor.model_dump(mode="json")
    if isinstance(valor, dict):
        return {k: _serializar(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [_serializar(v) for v in valor]
    return valor


async def ejecutar_tarea(
    app: CompiledStateGraph[AgentState],
    tarea: str,
    recursion_limit: int = config.RECURSION_LIMIT,
) -> dict[str, Any]:
    """Corre el grafo sobre una tarea, logueando cada nodo en vivo, y devuelve
    la traza completa (JSON-serializable)."""
    logger.info("USUARIO: %s", tarea)
    estado: dict[str, Any] = dict(estado_inicial(tarea))
    recorrido: list[str] = []
    cortado = False
    inicio = time.perf_counter()
    try:
        async for modo, chunk in app.astream(
            estado_inicial(tarea), config=config_job("cli", recursion_limit), stream_mode=["updates", "values"]
        ):
            if modo == "values":
                estado = chunk
                continue
            for nodo, update in chunk.items():
                recorrido.append(nodo)
                if isinstance(update, dict):
                    _loguear(nodo, update)
    except GraphRecursionError:
        cortado = True
        logger.error("Se alcanzó recursion_limit=%d: el grafo se cortó.", recursion_limit)

    return {
        "tarea": tarea,
        "prompt_supervisor": PROMPT_VERSION,
        "recorrido": recorrido,
        "decisiones": _serializar(estado.get("decisiones", [])),
        "contribuciones": _serializar(estado.get("contribuciones", [])),
        "validaciones": _serializar(estado.get("validaciones", {})),
        "intentos": estado.get("intentos", {}),
        "delegaciones": estado.get("pasos", 0),
        "task_completed": estado.get("task_completed", False),
        "requiere_revision_humana": estado.get("requiere_revision_humana", False),
        "motivo_escalamiento": estado.get("motivo_escalamiento"),
        "research_output": _serializar(estado.get("research_output")),
        "analysis_output": _serializar(estado.get("analysis_output")),
        "respuesta_final": estado.get("respuesta_final", ""),
        "cortado_por_recursion_limit": cortado,
        "duracion_segundos": round(time.perf_counter() - inicio, 2),
    }
