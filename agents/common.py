"""Utilidades compartidas por los nodos: normalización de mensajes, errores
transitorios del proveedor e inyección de fallas para probar el camino de error."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from langchain_core.messages import BaseMessage
from langgraph.types import RetryPolicy

import config


def texto_de(mensaje: BaseMessage) -> str:
    """Normaliza `content`: Gemini a veces devuelve una lista de bloques
    ([{'type': 'text', 'text': ..., 'extras': {'signature': ...}}]) en vez de
    un string. Nos quedamos solo con el texto."""
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido
    return "".join(
        bloque.get("text", "") if isinstance(bloque, dict) else str(bloque)
        for bloque in contenido
    )


def es_error_transitorio(error: BaseException) -> bool:
    """Rate limit (429) o caída temporal del proveedor: vale la pena reintentar."""
    texto = f"{type(error).__name__} {error}".lower()
    if "perday" in texto:
        return False  # cuota DIARIA agotada: reintentar en segundos no sirve
    return any(m in texto for m in ("ratelimit", "rate limit", "429", "resource_exhausted", "503", "unavailable"))


# Los nodos que llaman a un LLM re-lanzan los errores transitorios para que
# esta política los reintente con backoff (en vez de tratarlos como un
# output inválido del especialista).
POLITICA_REINTENTO_LLM = RetryPolicy(
    initial_interval=config.LLM_RETRY_INTERVALO_INICIAL,
    backoff_factor=2.0,
    max_interval=90.0,
    max_attempts=config.LLM_RETRY_INTENTOS,
    retry_on=es_error_transitorio,
)


TipoFalla = Literal["excepcion", "resultado_alterado"]


class FallasSimuladas:
    """Inyector de fallas para demostrar que el supervisor no se rompe.

    - `programar("investigador", "excepcion")`: la próxima invocación del
      investigador lanza una excepción (como un servicio externo caído).
    - `programar("analista", "resultado_alterado")`: el próximo artefacto del
      analista sale con un resultado numérico corrupto (como una alucinación);
      el validador determinístico tiene que detectarlo.
    """

    def __init__(self) -> None:
        self._pendientes: Counter[tuple[str, str]] = Counter()

    def programar(self, agente: str, tipo: TipoFalla, veces: int = 1) -> None:
        self._pendientes[(agente, tipo)] += veces

    def consumir(self, agente: str, tipo: TipoFalla) -> bool:
        if self._pendientes[(agente, tipo)] > 0:
            self._pendientes[(agente, tipo)] -= 1
            return True
        return False

    def limpiar(self) -> None:
        self._pendientes.clear()


FALLAS = FallasSimuladas()
