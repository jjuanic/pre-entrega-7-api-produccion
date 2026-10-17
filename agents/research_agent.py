"""
Agente Investigador: busca datos concretos en la documentación técnica.

- Herramientas ACOTADAS: solo puede buscar y listar documentos. No calcula.
- Fuente externa: el RAG híbrido de la Pre-entrega 4 (BM25 + Pinecone), con
  fallback a BM25 local (ver `rag/backend.py`).
- Salida estructurada: `ResearchArtifact` (Pydantic) vía `ToolStrategy`. Si el
  LLM devuelve algo que no valida, `create_agent` le reenvía el error de
  validación para que lo corrija (primer nivel del bucle de reparación).
- Provenance: la herramienta devuelve los fragmentos recuperados como
  `artifact` del ToolMessage. El nodo los guarda en `evidencia`, y el
  validador comprueba que toda fuente citada haya sido recuperada de verdad.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents.common import FALLAS, es_error_transitorio
from rag.backend import obtener_buscador
from state import AgentState, Contribucion, Evidencia, ResearchArtifact, ResearchOutput

NOMBRE = "investigador"
MAX_CARACTERES_FRAGMENTO = 1500
LIMITE_PASOS_INTERNOS = 12  # recursion_limit del sub-agente ReAct

SYSTEM_PROMPT = """Sos el Agente Investigador de un equipo de análisis técnico sobre Python.
Tu ÚNICA función es encontrar datos concretos en la documentación usando tus herramientas.

Reglas:
- Usá `buscar_documentacion` (podés llamarla varias veces con consultas distintas).
  Si no sabés qué documentos hay, usá `listar_documentos`.
- NO hagas cálculos ni estimaciones: eso lo hace otro agente. Solo reportá datos.
- NUNCA inventes datos ni fuentes. Cada hallazgo debe tener:
  * `fuente`: el nombre de archivo EXACTO que devolvió la herramienta (ej. "asyncio.md").
  * `cita`: un fragmento copiado textualmente del resultado de la búsqueda.
- Si la documentación no tiene el dato, decilo en el resumen y bajá el confidence_score.
- Respondé en español."""


class BuscarInput(BaseModel):
    consulta: str = Field(min_length=3, description="Qué buscar, en lenguaje natural o con términos exactos (ej. 'max_workers por defecto ThreadPoolExecutor').")
    categoria: str | None = Field(default=None, description="Filtro opcional por categoría (ver listar_documentos). Dejalo vacío si no estás seguro.")


@tool(args_schema=BuscarInput, response_format="content_and_artifact")
def buscar_documentacion(consulta: str, categoria: str | None = None) -> tuple[str, list[dict[str, str]]]:
    """Busca en la documentación técnica de la librería estándar de Python (asyncio,
    concurrent.futures, functools, dataclasses, pathlib, logging, json, typing y un
    FAQ de errores comunes). Devuelve los fragmentos más relevantes con su fuente
    (nombre de archivo) y sección. Usala para cualquier dato concreto: valores por
    defecto, comportamientos, recomendaciones."""
    resultados = obtener_buscador().buscar(consulta, categoria=categoria)
    if not resultados:
        return "No se encontraron fragmentos relevantes. Probá otra consulta o sin categoría.", []
    evidencia = [
        {"fuente": r["fuente"], "seccion": r.get("seccion") or "", "texto": r["contenido"][:MAX_CARACTERES_FRAGMENTO]}
        for r in resultados
    ]
    texto = "\n\n".join(
        f"[{i}] fuente: {e['fuente']} | sección: {e['seccion']}\n{e['texto']}"
        for i, e in enumerate(evidencia, start=1)
    )
    return texto, evidencia


@tool
def listar_documentos() -> str:
    """Lista los documentos disponibles (archivo, título y categoría). Útil para
    saber qué se puede buscar y qué categorías existen para filtrar."""
    return json.dumps(obtener_buscador().documentos(), ensure_ascii=False)


HERRAMIENTAS = [buscar_documentacion, listar_documentos]


def crear_agente_investigador(llm: BaseChatModel) -> Any:
    return create_agent(
        model=llm,
        tools=HERRAMIENTAS,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(ResearchArtifact),
        name=NOMBRE,
    )


def _evidencia_de(mensajes: list[BaseMessage]) -> list[Evidencia]:
    """Fragmentos realmente recuperados, tomados del `artifact` de cada ToolMessage."""
    evidencia: list[Evidencia] = []
    for m in mensajes:
        if isinstance(m, ToolMessage) and m.name == buscar_documentacion.name and isinstance(m.artifact, list):
            evidencia.extend(Evidencia(**e) for e in m.artifact)
    return evidencia


def construir_mensaje(state: AgentState) -> str:
    """Contexto ACOTADO para el especialista: la instrucción del supervisor y,
    si es un refinamiento, su output anterior + los errores ESPECÍFICOS del
    validador. No recibe el historial completo ni la metadata del sistema."""
    partes = [f"Instrucción del supervisor:\n{state['instruccion'] or state['tarea']}"]
    previo = state.get("research_output")
    validacion = state.get("validaciones", {}).get(NOMBRE)
    if previo is not None and validacion is not None and not validacion.valido:
        anterior = previo.artefacto.model_dump_json(indent=1) if previo.artefacto else f"(sin artefacto: {previo.error})"
        partes.append(
            "Tu intento anterior NO pasó la validación. Output anterior:\n"
            f"{anterior}\n\nErrores a corregir (corregí SOLO esto):\n- " + "\n- ".join(validacion.errores)
        )
    return "\n\n".join(partes)


def crear_nodo_investigador(llm: BaseChatModel) -> Any:
    agente = crear_agente_investigador(llm)

    async def nodo_investigador(state: AgentState) -> dict[str, Any]:
        intento = state.get("intentos", {}).get(NOMBRE, 0) + 1
        instruccion = state["instruccion"] or state["tarea"]
        artefacto: ResearchArtifact | None = None
        evidencia: list[Evidencia] = []
        error: str | None = None
        try:
            if FALLAS.consumir(NOMBRE, "excepcion"):
                raise ConnectionError("Falla simulada: el servicio de búsqueda no respondió.")
            resultado = await agente.ainvoke(
                {"messages": [HumanMessage(content=construir_mensaje(state))]},
                config={"recursion_limit": LIMITE_PASOS_INTERNOS},
            )
            evidencia = _evidencia_de(resultado["messages"])
            artefacto = resultado.get("structured_response")
            if artefacto is None:
                error = "El agente terminó sin devolver un ResearchArtifact."
        except Exception as exc:
            if es_error_transitorio(exc):
                raise  # lo reintenta la RetryPolicy del nodo
            error = f"{type(exc).__name__}: {exc}"

        salida = ResearchOutput(intento=intento, instruccion=instruccion, artefacto=artefacto, evidencia=evidencia, error=error)
        if artefacto:
            detalle = f"{len(artefacto.hallazgos)} hallazgo(s) de {sorted(salida.fuentes_consultadas)}: {artefacto.resumen}"
        else:
            detalle = f"ERROR: {error}"
        return {
            "research_output": salida,
            "intentos": {NOMBRE: 1},
            "contribuciones": [Contribucion(agente=NOMBRE, tipo="investigacion", intento=intento, detalle=detalle)],
            "messages": [AIMessage(content=detalle, name=NOMBRE)],
        }

    return nodo_investigador
