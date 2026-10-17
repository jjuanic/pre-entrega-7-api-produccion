"""
Agente Analista: procesa los datos que encontró el Investigador.

- Herramientas ACOTADAS: una calculadora segura y estadísticas descriptivas.
  No puede buscar información nueva.
- Contexto mínimo: recibe la instrucción del supervisor + los hallazgos
  validados del investigador (dato y fuente), no el historial completo.
- Salida estructurada: `AnalysisArtifact`, donde cada cálculo declara la
  expresión exacta que usó. El validador la vuelve a evaluar con el mismo
  evaluador: si el número declarado no coincide, el cálculo se rechaza.

La calculadora NO usa `eval()`: parsea la expresión con `ast` y solo permite
números, operadores aritméticos y una lista blanca de funciones. Evaluar
texto generado por un LLM con `eval()` sería ejecución de código arbitrario.
"""

from __future__ import annotations

import ast
import math
import operator as op
import statistics
from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents.common import FALLAS, es_error_transitorio
from state import AgentState, AnalysisArtifact, AnalysisOutput, CalculoEjecutado, Contribucion

NOMBRE = "analista"
LIMITE_PASOS_INTERNOS = 16

SYSTEM_PROMPT = """Sos el Agente Analista de un equipo de análisis técnico sobre Python.
Tu ÚNICA función es hacer cálculos y comparaciones a partir de los datos que te pasan.

Reglas:
- TODO número que reportes tiene que salir de la herramienta `calculadora` (o
  `estadisticas`). No hagas cuentas de cabeza.
- Pedí todos los cálculos independientes JUNTOS, en una sola ronda (varias llamadas a
  `calculadora` en paralelo), y limitate a los cálculos que la instrucción necesita.
- En cada cálculo del resultado final poné la `expresion` EXACTA que le pasaste a
  la calculadora y el `resultado` que te devolvió.
- NO busques información nueva. Si te falta un dato, explicitalo como supuesto.
- Explicá los supuestos (ej. "latencia constante por llamada").
- Respondé en español."""

# --- Evaluador seguro -----------------------------------------------------

_OPERADORES: dict[type, Callable[..., Any]] = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
    ast.USub: op.neg, ast.UAdd: op.pos,
}
_FUNCIONES: dict[str, Callable[..., Any]] = {
    "ceil": math.ceil, "floor": math.floor, "sqrt": math.sqrt, "log": math.log,
    "log2": math.log2, "min": min, "max": max, "abs": abs, "round": round,
}
_MAX_EXPONENTE = 100


def _evaluar(nodo: ast.AST) -> float:
    match nodo:
        case ast.Constant(value=bool()):
            raise ValueError("No se permiten booleanos")
        case ast.Constant(value=int() | float() as valor):
            return valor
        case ast.BinOp(left=izq, op=operador, right=der) if type(operador) in _OPERADORES:
            a, b = _evaluar(izq), _evaluar(der)
            if isinstance(operador, ast.Pow) and abs(b) > _MAX_EXPONENTE:
                raise ValueError("Exponente demasiado grande")
            return _OPERADORES[type(operador)](a, b)
        case ast.UnaryOp(op=operador, operand=operando) if type(operador) in _OPERADORES:
            return _OPERADORES[type(operador)](_evaluar(operando))
        case ast.Call(func=ast.Name(id=nombre), args=args, keywords=[]) if nombre in _FUNCIONES:
            return _FUNCIONES[nombre](*(_evaluar(a) for a in args))
        case _:
            raise ValueError(f"Elemento no permitido: {ast.dump(nodo)[:60]}")


def evaluar_expresion(expresion: str) -> float:
    """Evalúa una expresión aritmética de forma segura. Lanza ValueError si no es válida."""
    expresion = expresion.replace("^", "**").strip()
    try:
        arbol = ast.parse(expresion, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Sintaxis inválida: {exc.msg}") from exc
    try:
        return float(_evaluar(arbol.body))
    except ZeroDivisionError as exc:
        raise ValueError("División por cero") from exc


# --- Herramientas -----------------------------------------------------------

class CalculadoraInput(BaseModel):
    expresion: str = Field(
        min_length=1,
        description="Expresión aritmética. Operadores: + - * / // % ** y paréntesis. "
        "Funciones: ceil, floor, sqrt, log, log2, min, max, abs, round. Ej: 'ceil(240 / 12) * 0.5'.",
    )


@tool(args_schema=CalculadoraInput, response_format="content_and_artifact")
def calculadora(expresion: str) -> tuple[str, dict[str, Any]]:
    """Evalúa una expresión matemática y devuelve el resultado exacto. Usala para
    TODOS los cálculos (tiempos, cantidades, porcentajes, speedups)."""
    try:
        resultado = evaluar_expresion(expresion)
    except (ValueError, TypeError, OverflowError) as exc:
        return f"Error al calcular '{expresion}': {exc}", {"expresion": expresion, "resultado": None, "error": str(exc)}
    return f"{expresion} = {resultado:g}", {"expresion": expresion, "resultado": resultado}


class EstadisticasInput(BaseModel):
    valores: list[float] = Field(min_length=1, description="Lista de números a resumir.")


@tool(args_schema=EstadisticasInput)
def estadisticas(valores: list[float]) -> str:
    """Estadísticas descriptivas de una lista de números: cantidad, suma, mínimo,
    máximo, media, mediana y desvío estándar."""
    desvio = statistics.stdev(valores) if len(valores) > 1 else 0.0
    return (
        f"n={len(valores)}, suma={sum(valores):g}, min={min(valores):g}, max={max(valores):g}, "
        f"media={statistics.mean(valores):g}, mediana={statistics.median(valores):g}, desvio={desvio:g}"
    )


HERRAMIENTAS = [calculadora, estadisticas]


def crear_agente_analista(llm: BaseChatModel) -> Any:
    return create_agent(
        model=llm,
        tools=HERRAMIENTAS,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(AnalysisArtifact),
        name=NOMBRE,
    )


def _calculos_de(mensajes: list[BaseMessage]) -> list[CalculoEjecutado]:
    return [
        CalculoEjecutado(**m.artifact)
        for m in mensajes
        if isinstance(m, ToolMessage) and m.name == calculadora.name and isinstance(m.artifact, dict)
    ]


def construir_mensaje(state: AgentState) -> str:
    """Contexto ACOTADO: instrucción + datos del investigador (si pasaron la
    validación) + errores del validador si es un refinamiento."""
    partes = [f"Instrucción del supervisor:\n{state['instruccion'] or state['tarea']}"]

    investigacion = state.get("research_output")
    validacion_inv = state.get("validaciones", {}).get("investigador")
    if investigacion and investigacion.artefacto and validacion_inv and validacion_inv.valido:
        datos = "\n".join(f"- {h.dato} (fuente: {h.fuente})" for h in investigacion.artefacto.hallazgos)
        partes.append(f"Datos verificados por el Investigador:\n{datos}")

    previo = state.get("analysis_output")
    validacion = state.get("validaciones", {}).get(NOMBRE)
    if previo is not None and validacion is not None and not validacion.valido:
        anterior = previo.artefacto.model_dump_json(indent=1) if previo.artefacto else f"(sin artefacto: {previo.error})"
        partes.append(
            "Tu intento anterior NO pasó la validación. Output anterior:\n"
            f"{anterior}\n\nErrores a corregir (corregí SOLO esto, recalculando con la calculadora):\n- "
            + "\n- ".join(validacion.errores)
        )
    return "\n\n".join(partes)


def crear_nodo_analista(llm: BaseChatModel) -> Any:
    agente = crear_agente_analista(llm)

    async def nodo_analista(state: AgentState) -> dict[str, Any]:
        intento = state.get("intentos", {}).get(NOMBRE, 0) + 1
        instruccion = state["instruccion"] or state["tarea"]
        artefacto: AnalysisArtifact | None = None
        ejecutados: list[CalculoEjecutado] = []
        error: str | None = None
        try:
            if FALLAS.consumir(NOMBRE, "excepcion"):
                raise RuntimeError("Falla simulada: el analista se cayó a mitad de la tarea.")
            resultado = await agente.ainvoke(
                {"messages": [HumanMessage(content=construir_mensaje(state))]},
                config={"recursion_limit": LIMITE_PASOS_INTERNOS},
            )
            ejecutados = _calculos_de(resultado["messages"])
            artefacto = resultado.get("structured_response")
            if artefacto is None:
                error = "El agente terminó sin devolver un AnalysisArtifact."
            elif FALLAS.consumir(NOMBRE, "resultado_alterado"):
                # Simula una alucinación: el número reportado no es el que dio la calculadora.
                primero = artefacto.calculos[0]
                alterado = primero.model_copy(update={"resultado": primero.resultado * 1.5 + 7})
                artefacto = artefacto.model_copy(update={"calculos": [alterado, *artefacto.calculos[1:]]})
        except Exception as exc:
            if es_error_transitorio(exc):
                raise
            error = f"{type(exc).__name__}: {exc}"

        salida = AnalysisOutput(
            intento=intento, instruccion=instruccion, artefacto=artefacto, calculos_ejecutados=ejecutados, error=error
        )
        if artefacto:
            cuentas = "; ".join(f"{c.descripcion} = {c.resultado:g}" for c in artefacto.calculos)
            detalle = f"{cuentas}. Conclusión: {artefacto.conclusion}"
        else:
            detalle = f"ERROR: {error}"
        return {
            "analysis_output": salida,
            "intentos": {NOMBRE: 1},
            "contribuciones": [Contribucion(agente=NOMBRE, tipo="analisis", intento=intento, detalle=detalle)],
            "messages": [AIMessage(content=detalle, name=NOMBRE)],
        }

    return nodo_analista
