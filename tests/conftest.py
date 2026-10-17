"""Utilidades de test: un LLM "guionado" que no requiere API key.

Todos los nodos (supervisor, especialistas, síntesis) comparten el mismo LLM,
así que el guion es la secuencia EXACTA de respuestas en el orden en que el
grafo las va a pedir.
"""

from __future__ import annotations

import os

# El investigador usa el BM25 local sobre data/docs (sin Pinecone ni red).
os.environ["RESEARCH_BACKEND"] = "local"
os.environ["CONFIANZA_MINIMA"] = "0.5"

from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from agents.common import FALLAS


class LLMGuionado(GenericFakeChatModel):
    """Devuelve respuestas predefinidas (incluidas tool_calls) y registra lo
    que recibió en cada llamada, para verificar el contexto de cada agente."""

    recibidos: list[list[BaseMessage]] = []

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "LLMGuionado":
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.recibidos.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


_contador = iter(range(10_000))


def llamada(nombre: str, **args: Any) -> AIMessage:
    """Un AIMessage con una única tool_call (herramienta o salida estructurada)."""
    return AIMessage(content="", tool_calls=[{"name": nombre, "args": args, "id": f"call_{next(_contador)}"}])


def decidir(next_: str, razon: str = "rubrica", instruccion: str = "") -> AIMessage:
    """Respuesta del supervisor (with_structured_output -> tool_call DecisionSupervisor)."""
    return llamada("DecisionSupervisor", next=next_, razon=razon, instruccion=instruccion)


def guion(*mensajes: AIMessage) -> LLMGuionado:
    iterador: Iterator[AIMessage] = iter(mensajes)
    return LLMGuionado(messages=iterador, recibidos=[])


# --- Bloques de guion reutilizables --------------------------------------

CITA_WORKERS = "Si no se indica `max_workers`, `ThreadPoolExecutor` usa `min(32, os.cpu_count() + 4)` hilos"


def investigacion_ok(fuente: str = "concurrent_futures.md", confianza: float = 0.9) -> list[AIMessage]:
    return [
        llamada("buscar_documentacion", consulta="max_workers por defecto ThreadPoolExecutor"),
        llamada(
            "ResearchArtifact",
            hallazgos=[{"dato": "ThreadPoolExecutor usa min(32, os.cpu_count() + 4) hilos por defecto", "fuente": fuente, "cita": CITA_WORKERS}],
            resumen="El pool de hilos por defecto usa min(32, cpu + 4) workers.",
            confidence_score=confianza,
        ),
    ]


def analisis(resultado_declarado: float = 12.0, expresion: str = "min(32, 8 + 4)") -> list[AIMessage]:
    return [
        llamada("calculadora", expresion=expresion),
        llamada(
            "AnalysisArtifact",
            calculos=[{"descripcion": "Workers por defecto con 8 núcleos", "expresion": expresion, "resultado": resultado_declarado}],
            conclusion="Con 8 núcleos el pool por defecto tiene 12 hilos.",
            supuestos=["La máquina tiene 8 núcleos"],
            confidence_score=0.9,
        ),
    ]


@pytest.fixture(autouse=True)
def sin_fallas() -> Iterator[None]:
    FALLAS.limpiar()
    yield
    FALLAS.limpiar()
