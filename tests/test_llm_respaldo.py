"""Modelo de respaldo: si el principal agota su cuota, el grafo sigue con el siguiente."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.graph import compilar
from conftest import analisis, decidir, guion, investigacion_ok
from llm import con_respaldo
from state import estado_inicial


class SinCuota(GenericFakeChatModel):
    """Simula el 429 de cuota diaria agotada del free tier."""

    llamadas: int = 0

    def bind_tools(self, *args: Any, **kwargs: Any) -> "SinCuota":
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        self.llamadas += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")


def test_grafo_completo_con_el_modelo_de_respaldo() -> None:
    principal = SinCuota(messages=iter([]))
    respaldo = guion(
        decidir("investigador", instruccion="Buscá el max_workers por defecto."),
        *investigacion_ok(),
        decidir("analista", instruccion="Calculá min(32, 8 + 4)."),
        *analisis(),
        decidir("FINISH", razon="Todo válido."),
        AIMessage(content="Usa 12 hilos (concurrent_futures.md)."),
    )
    llm = con_respaldo(principal, [respaldo])
    resultado = asyncio.run(compilar(llm).ainvoke(estado_inicial("¿Cuántos hilos usa ThreadPoolExecutor?")))
    assert resultado["respuesta_final"].startswith("Usa 12 hilos")
    assert resultado["validaciones"]["investigador"].valido and resultado["validaciones"]["analista"].valido
    assert principal.llamadas >= 6  # se intentó primero con el principal en cada llamada


def test_sin_respaldo_devuelve_el_mismo_modelo() -> None:
    modelo = guion()
    assert con_respaldo(modelo, []) is modelo
