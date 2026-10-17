"""
Nodo Validador: puerta de calidad DETERMINÍSTICA (código Python, no un LLM).

Corre después de cada especialista y valida solo la parcela del agente que
acaba de trabajar. "Pydantic, no vibras": que un texto suene convincente no
garantiza que sea correcto, así que acá se chequean reglas verificables.

Investigador:
  - terminó sin error y devolvió un `ResearchArtifact` (ya validado por Pydantic);
  - Provenance: cada `fuente` citada tiene que estar entre los documentos que la
    herramienta de búsqueda devolvió DE VERDAD (detecta fuentes inventadas);
  - la `cita` tiene que poder rastrearse en el texto recuperado de esa fuente
    (si no, es advertencia: puede ser una paráfrasis);
  - `confidence_score` >= CONFIANZA_MINIMA.

Analista:
  - terminó sin error y devolvió un `AnalysisArtifact`;
  - cada `expresion` se vuelve a evaluar con el MISMO evaluador seguro y el
    resultado tiene que coincidir con el declarado (detecta números alucinados);
  - advertencia si un cálculo no pasó por la calculadora;
  - `confidence_score` >= CONFIANZA_MINIMA.

Los errores son ESPECÍFICOS ("el cálculo X declara 25 pero la expresión da 10")
porque se le reenvían al especialista en el bucle de reparación.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import config
from agents.analyst_agent import evaluar_expresion
from state import AgentState, AnalysisOutput, Contribucion, ResearchOutput, ValidationReport

NOMBRE = "validador"
# Se tolera el redondeo que hace el LLM al transcribir (1.6667 por 10/6), pero
# no un número distinto: 0,5 % relativo o 0,01 absoluto.
TOLERANCIA_RELATIVA = 5e-3
TOLERANCIA_ABSOLUTA = 1e-2
SOLAPAMIENTO_MINIMO_CITA = 0.6


def _palabras(texto: str) -> set[str]:
    sin_tildes = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return {p for p in re.findall(r"[a-z0-9_]+", sin_tildes) if len(p) > 2}


def validar_investigacion(salida: ResearchOutput) -> ValidationReport:
    errores: list[str] = []
    advertencias: list[str] = []
    if salida.error:
        errores.append(f"La investigación falló: {salida.error}")
    elif salida.artefacto is None:
        errores.append("No se devolvió ningún ResearchArtifact.")
    else:
        art = salida.artefacto
        consultadas = salida.fuentes_consultadas
        if not salida.evidencia:
            errores.append("No se usó la herramienta buscar_documentacion: los hallazgos no tienen respaldo.")
        for i, h in enumerate(art.hallazgos, start=1):
            if h.fuente not in consultadas:
                errores.append(
                    f"Hallazgo {i} cita la fuente '{h.fuente}', que no aparece en los resultados de búsqueda "
                    f"(fuentes recuperadas: {sorted(consultadas)}). Usá el nombre exacto o eliminá el hallazgo."
                )
                continue
            texto_fuente = " ".join(e.texto for e in salida.evidencia if e.fuente == h.fuente)
            palabras_cita = _palabras(h.cita)
            if palabras_cita:
                solapamiento = len(palabras_cita & _palabras(texto_fuente)) / len(palabras_cita)
                if solapamiento < SOLAPAMIENTO_MINIMO_CITA:
                    advertencias.append(
                        f"La cita del hallazgo {i} no se encuentra textual en '{h.fuente}' "
                        f"(solapamiento {solapamiento:.0%}): posible paráfrasis."
                    )
        if art.confidence_score < config.CONFIANZA_MINIMA:
            errores.append(
                f"confidence_score={art.confidence_score:.2f} es menor al mínimo {config.CONFIANZA_MINIMA}: "
                "la documentación recuperada no alcanza; probá otras consultas."
            )
    return ValidationReport(agente="investigador", intento=salida.intento, valido=not errores, errores=errores, advertencias=advertencias)


def _coinciden(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=TOLERANCIA_RELATIVA, abs_tol=TOLERANCIA_ABSOLUTA)


def validar_analisis(salida: AnalysisOutput) -> ValidationReport:
    errores: list[str] = []
    advertencias: list[str] = []
    if salida.error:
        errores.append(f"El análisis falló: {salida.error}")
    elif salida.artefacto is None:
        errores.append("No se devolvió ningún AnalysisArtifact.")
    else:
        art = salida.artefacto
        ejecutadas = {re.sub(r"\s+", "", c.expresion) for c in salida.calculos_ejecutados if c.error is None}
        for c in art.calculos:
            try:
                esperado = evaluar_expresion(c.expresion)
            except ValueError as exc:
                errores.append(f"El cálculo '{c.descripcion}' tiene una expresión inválida ('{c.expresion}'): {exc}.")
                continue
            if not _coinciden(esperado, c.resultado):
                errores.append(
                    f"El cálculo '{c.descripcion}' declara resultado={c.resultado:.10g}, "
                    f"pero la expresión '{c.expresion}' da {esperado:.10g}."
                )
            if re.sub(r"\s+", "", c.expresion) not in ejecutadas:
                advertencias.append(f"La expresión '{c.expresion}' no se ejecutó con la calculadora.")
        if art.confidence_score < config.CONFIANZA_MINIMA:
            errores.append(f"confidence_score={art.confidence_score:.2f} es menor al mínimo {config.CONFIANZA_MINIMA}.")
    return ValidationReport(agente="analista", intento=salida.intento, valido=not errores, errores=errores, advertencias=advertencias)


def nodo_validador(state: AgentState) -> dict[str, Any]:
    """Valida la parcela del especialista que acaba de correr (`next_agent`)."""
    agente = state["next_agent"]
    if agente == "investigador" and state.get("research_output") is not None:
        reporte = validar_investigacion(state["research_output"])  # type: ignore[arg-type]
    elif agente == "analista" and state.get("analysis_output") is not None:
        reporte = validar_analisis(state["analysis_output"])  # type: ignore[arg-type]
    else:
        return {}

    if reporte.valido:
        detalle = f"{agente} (intento {reporte.intento}): VÁLIDO"
        if reporte.advertencias:
            detalle += f" con {len(reporte.advertencias)} advertencia(s): " + " | ".join(reporte.advertencias)
    else:
        detalle = f"{agente} (intento {reporte.intento}): INVÁLIDO -> " + " | ".join(reporte.errores)
    return {
        "validaciones": {agente: reporte},
        "contribuciones": [Contribucion(agente=NOMBRE, tipo="validacion", intento=reporte.intento, detalle=detalle)],
    }
