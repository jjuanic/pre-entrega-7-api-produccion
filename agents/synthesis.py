"""
Nodo de Síntesis: redacta la respuesta final a partir de las parcelas validadas.

Solo usa outputs que pasaron el validador (un output inválido no llega al
usuario como si fuera cierto). Si el supervisor escaló a revisión humana, la
respuesta lo dice explícitamente. Las fuentes se agregan de forma
determinística desde los hallazgos, no las redacta el LLM.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.common import texto_de
from state import AgentState, Contribucion

NOMBRE = "sintesis"

SYSTEM_PROMPT = """Sos el redactor final de un equipo de análisis técnico sobre Python.
Combiná los aportes VALIDADOS de los especialistas en una respuesta clara para el usuario,
en español rioplatense y en formato Markdown breve (un párrafo de respuesta directa y
después el detalle con viñetas).

Reglas:
- Usá SOLO los datos y números que te pasan. No agregues datos nuevos ni recalcules.
- Mencioná de qué documento sale cada dato entre paréntesis, p. ej. (concurrent_futures.md).
- Si se indica que hay partes NO validadas, decilo explícitamente y no las presentes como ciertas.
- No incluyas una sección de fuentes al final (se agrega automáticamente)."""


def _contexto(state: AgentState) -> str:
    partes = [f"Pregunta del usuario:\n{state['tarea']}"]
    validaciones = state.get("validaciones", {})

    inv = state.get("research_output")
    if inv and inv.artefacto and validaciones.get("investigador") and validaciones["investigador"].valido:
        hallazgos = "\n".join(f"- {h.dato} ({h.fuente})" for h in inv.artefacto.hallazgos)
        partes.append(f"Investigación (validada):\n{inv.artefacto.resumen}\n{hallazgos}")
    else:
        partes.append("Investigación: NO hay datos validados.")

    ana = state.get("analysis_output")
    if ana and ana.artefacto and validaciones.get("analista") and validaciones["analista"].valido:
        calculos = "\n".join(f"- {c.descripcion}: {c.expresion} = {c.resultado:g}" for c in ana.artefacto.calculos)
        supuestos = "\n".join(f"- {s}" for s in ana.artefacto.supuestos) or "- (ninguno)"
        partes.append(f"Análisis (validado):\n{calculos}\nConclusión: {ana.artefacto.conclusion}\nSupuestos:\n{supuestos}")
    elif ana is not None:
        partes.append("Análisis: el análisis NO pasó la validación; no uses sus números.")

    if state.get("requiere_revision_humana"):
        partes.append(f"ATENCIÓN: el supervisor escaló a revisión humana. Motivo: {state.get('motivo_escalamiento')}")
    return "\n\n".join(partes)


def _fuentes(state: AgentState) -> list[str]:
    inv = state.get("research_output")
    if not inv or not inv.artefacto:
        return []
    return sorted({h.fuente for h in inv.artefacto.hallazgos})


def crear_nodo_sintesis(llm: BaseChatModel) -> Any:
    async def nodo_sintesis(state: AgentState) -> dict[str, Any]:
        respuesta = await llm.ainvoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(_contexto(state))])
        texto = texto_de(respuesta).strip()
        if state.get("requiere_revision_humana"):
            texto = f"> ⚠️ **Requiere revisión humana**: {state.get('motivo_escalamiento')}\n\n{texto}"
        if fuentes := _fuentes(state):
            texto += "\n\n**Fuentes:** " + ", ".join(f"`{f}`" for f in fuentes)
        return {
            "respuesta_final": texto,
            "messages": [AIMessage(content=texto, name=NOMBRE)],
            "contribuciones": [Contribucion(agente=NOMBRE, tipo="sintesis", detalle=texto)],
        }

    return nodo_sintesis
