"""
Nodo Supervisor: el router inteligente y controlador de flujo.

Responde una sola pregunta: "dado el estado actual, ¿quién interviene ahora o
es momento de finalizar?". Coordina, NO ejecuta: no busca ni calcula (evita el
"Supervisor Todólogo").

La decisión tiene dos capas:

1. LLM con `with_structured_output(DecisionSupervisor)`: elige el próximo
   agente siguiendo una rúbrica explícita y redacta una instrucción puntual.
   No ve el historial completo: recibe un RESUMEN del estado (qué aportó cada
   especialista y qué dijo el validador).

2. Guardas determinísticas (código) que el LLM no puede saltear:
   - MAX_PASOS: tope de delegaciones -> cierra (anti "Supervisor Infinito").
   - MAX_INTENTOS_POR_AGENTE: un especialista que agotó sus intentos no se
     vuelve a invocar; si su output sigue inválido se escala a revisión humana.
   - SUFICIENCIA: no se puede cerrar con un output INVÁLIDO si al especialista
     le quedan intentos, ni sin al menos una investigación válida.
   - FALLBACK: si el LLM del supervisor falla, se rutea con reglas fijas en
     vez de romper el grafo.

Cada decisión queda en `decisiones` (quién decidió qué, por qué y si alguna
guarda corrigió al LLM).
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

import config
from agents.common import es_error_transitorio
from state import ESPECIALISTAS, AgentState, DecisionSupervisor, Destino, Especialista, RegistroDecision

NOMBRE = "supervisor"

# Versión del prompt: un cambio chico en el prompt del supervisor puede
# romper todo el ruteo, así que se versiona y queda registrada en la traza.
PROMPT_VERSION = "supervisor-v3"

SYSTEM_PROMPT = """Sos el Supervisor de un equipo de análisis técnico sobre Python. Coordinás, NO resolvés:
nunca hagas búsquedas ni cálculos vos mismo.

Especialistas disponibles:
- investigador: busca datos concretos en la documentación técnica de Python (RAG sobre
  asyncio, concurrent.futures, functools, dataclasses, pathlib, logging, json, typing y un FAQ).
  No calcula.
- analista: hace cálculos, estimaciones y comparaciones numéricas con una calculadora,
  a partir de datos ya obtenidos. No busca información.

Después de cada especialista, un VALIDADOR determinístico revisa su output (fuentes reales,
cálculos reproducibles, confianza mínima). Vas a ver su veredicto en el estado.

Rúbrica (aplicala en orden):
1. Si un output está INVÁLIDO, mandá a ese MISMO especialista a refinarlo (el sistema le
   adjunta su output anterior y los errores; vos dale una instrucción breve).
2. Si a la tarea le falta un dato de la documentación -> investigador.
3. Si la tarea pide cálculos, estimaciones o comparaciones y no hay un análisis VÁLIDO -> analista.
4. FINISH solo si: (a) todas las partes de la tarea están cubiertas, (b) los outputs están
   VÁLIDOS y (c) los números del análisis usan los datos de la investigación.

La `instruccion` tiene que ser concreta y autocontenida (el especialista NO ve la conversación).
Para el analista, enumerá los cálculos a hacer con los números concretos."""


def estado_de(state: AgentState, agente: Especialista) -> Literal["pendiente", "valido", "invalido"]:
    reporte = state.get("validaciones", {}).get(agente)
    if reporte is None:
        return "pendiente"
    return "valido" if reporte.valido else "invalido"


def _intentos(state: AgentState, agente: Especialista) -> int:
    return state.get("intentos", {}).get(agente, 0)


def resumen_estado(state: AgentState) -> str:
    """Vista compacta del estado para el LLM del supervisor (estado mínimo)."""
    lineas = [
        f"TAREA DEL USUARIO:\n{state['tarea']}",
        f"\nDelegaciones usadas: {state['pasos']}/{config.MAX_PASOS}",
    ]
    inv, ana = state.get("research_output"), state.get("analysis_output")
    for agente, salida in (("investigador", inv), ("analista", ana)):
        estado = estado_de(state, agente)
        lineas.append(
            f"\n[{agente.upper()}] estado={estado.upper()} | intentos {_intentos(state, agente)}/{config.MAX_INTENTOS_POR_AGENTE}"
        )
        if salida is None:
            continue
        lineas.append(f"  Última instrucción: {salida.instruccion}")
        if agente == "investigador" and inv and inv.artefacto:
            lineas.append(f"  Resumen: {inv.artefacto.resumen}")
            lineas.extend(f"  - {h.dato} (fuente: {h.fuente})" for h in inv.artefacto.hallazgos)
        elif agente == "analista" and ana and ana.artefacto:
            lineas.extend(f"  - {c.descripcion}: {c.expresion} = {c.resultado:g}" for c in ana.artefacto.calculos)
            lineas.append(f"  Conclusión: {ana.artefacto.conclusion}")
        elif salida.error:
            lineas.append(f"  Error: {salida.error}")
        reporte = state.get("validaciones", {}).get(agente)
        if reporte and reporte.errores:
            lineas.extend(f"  ! Error de validación: {e}" for e in reporte.errores)
    lineas.append("\n¿Quién debe intervenir ahora, o es momento de FINISH?")
    return "\n".join(lineas)


def decision_por_defecto(state: AgentState) -> DecisionSupervisor:
    """Ruteo determinístico de respaldo si el LLM del supervisor falla."""
    for agente in ESPECIALISTAS:
        if estado_de(state, agente) != "valido":
            return DecisionSupervisor(
                next=agente,
                razon=f"Fallback: el output de {agente} todavía no es válido.",
                instruccion=state["tarea"],
            )
    return DecisionSupervisor(next="FINISH", razon="Fallback: ambos outputs son válidos.")


def aplicar_guardas(
    decision: DecisionSupervisor, state: AgentState
) -> tuple[Destino, str, str, str | None, str | None]:
    """Devuelve (destino, razon, instruccion, regla_aplicada, motivo_escalamiento)."""
    invalidos = [a for a in ESPECIALISTAS if estado_de(state, a) == "invalido"]
    con_presupuesto = [a for a in invalidos if _intentos(state, a) < config.MAX_INTENTOS_POR_AGENTE]
    agotados = [a for a in invalidos if a not in con_presupuesto]

    def cerrar(razon: str, regla: str | None) -> tuple[Destino, str, str, str | None, str | None]:
        motivo = None
        if agotados:
            motivo = f"{', '.join(agotados)} agotó sus {config.MAX_INTENTOS_POR_AGENTE} intentos con output inválido."
        return "sintesis", razon, "", regla, motivo

    if decision.next != "FINISH":
        agente: Especialista = decision.next
        if _intentos(state, agente) >= config.MAX_INTENTOS_POR_AGENTE:
            return cerrar(
                f"El LLM pidió a {agente}, pero ya usó {config.MAX_INTENTOS_POR_AGENTE} intentos. Se cierra con lo disponible.",
                "MAX_INTENTOS_POR_AGENTE",
            )
        return agente, decision.razon, decision.instruccion or state["tarea"], None, None

    # El LLM quiere terminar: criterio de suficiencia estricto.
    if con_presupuesto:
        agente = con_presupuesto[0]
        previa = state["research_output"] if agente == "investigador" else state["analysis_output"]
        instruccion = previa.instruccion if previa else state["tarea"]
        return agente, f"No se puede cerrar: el output de {agente} es INVÁLIDO y le quedan intentos.", instruccion, "SUFICIENCIA", None
    if estado_de(state, "investigador") == "pendiente":
        return "investigador", "No se puede cerrar sin ninguna investigación que respalde la respuesta.", state["tarea"], "SUFICIENCIA", None
    return cerrar(decision.razon, None)


def crear_nodo_supervisor(llm: BaseChatModel) -> Any:
    llm_router = llm.with_structured_output(DecisionSupervisor)

    async def nodo_supervisor(state: AgentState) -> dict[str, Any]:
        paso = len(state.get("decisiones", [])) + 1

        if state["pasos"] >= config.MAX_PASOS:
            # Guarda dura: ni siquiera se consulta al LLM.
            pendientes = [a for a in ESPECIALISTAS if estado_de(state, a) == "invalido"]
            destino, instruccion, regla, propuesta = "sintesis", "", "MAX_PASOS", "-"
            razon = f"Se alcanzó el límite de {config.MAX_PASOS} delegaciones: se cierra sin consultar al LLM."
            motivo = f"Límite de delegaciones alcanzado con {', '.join(pendientes)} inválido." if pendientes else None
        else:
            try:
                decision = await llm_router.ainvoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(resumen_estado(state))])
                if not isinstance(decision, DecisionSupervisor):
                    decision = DecisionSupervisor.model_validate(decision)
                regla_fallback = None
            except Exception as exc:
                if es_error_transitorio(exc):
                    raise
                decision = decision_por_defecto(state)
                decision.razon += f" (el LLM del supervisor falló: {type(exc).__name__})"
                regla_fallback = "FALLBACK"
            propuesta = decision.next
            destino, razon, instruccion, regla, motivo = aplicar_guardas(decision, state)
            regla = regla or regla_fallback

        registro = RegistroDecision(
            paso=paso, propuesta_llm=propuesta, destino=destino, razon=razon, instruccion=instruccion, regla_aplicada=regla
        )
        update: dict[str, Any] = {"next_agent": destino, "instruccion": instruccion, "decisiones": [registro]}
        if destino == "sintesis":
            update["task_completed"] = motivo is None
            update["requiere_revision_humana"] = motivo is not None
            update["motivo_escalamiento"] = motivo
        else:
            update["pasos"] = state["pasos"] + 1
        return update

    return nodo_supervisor
