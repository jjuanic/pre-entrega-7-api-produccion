"""Construcción de los modelos de chat según el proveedor configurado en `.env`.

Cada rol del grafo puede usar un modelo distinto (`LLM_MODELO_SUPERVISOR`,
`LLM_MODELO_INVESTIGADOR`, `LLM_MODELO_ANALISTA`, `LLM_MODELO_SINTESIS`); si
no se define, usa `LLM_MODEL`. Sirve para optimizar costo (un modelo "lite"
para la síntesis, uno más capaz para el ruteo) y, con el free tier de Gemini,
para repartir la carga entre cuotas diarias independientes por modelo.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableWithFallbacks

import config


def crear_llm(modelo: str | None = None) -> BaseChatModel:
    """Instancia el chat model del proveedor elegido (import perezoso del SDK)."""
    config.require_api_key()
    modelo = modelo or config.LLM_MODEL

    match config.LLM_PROVIDER:
        case "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model=modelo, temperature=config.LLM_TEMPERATURE)
        case "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=modelo, temperature=config.LLM_TEMPERATURE)
        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model_name=modelo, temperature=config.LLM_TEMPERATURE)  # type: ignore[call-arg]


class ModeloConRespaldo(RunnableWithFallbacks):
    """`RunnableWithFallbacks` para chat models: si una llamada falla (p. ej. 429
    por cuota diaria agotada), se reintenta con el siguiente modelo de la lista.

    `bind_tools` y `with_structured_output` (lo que usan `create_agent` y el
    router del supervisor) se aplican a TODOS los modelos. Se definen acá en vez
    de depender del `__getattr__` genérico de LangChain, que inspecciona type
    hints y falla en Python 3.14 con algunos proveedores.
    """

    def _en_todos(self, metodo: str, *args: Any, **kwargs: Any) -> "ModeloConRespaldo":
        return ModeloConRespaldo(
            runnable=getattr(self.runnable, metodo)(*args, **kwargs),
            fallbacks=[getattr(f, metodo)(*args, **kwargs) for f in self.fallbacks],
            exceptions_to_handle=self.exceptions_to_handle,
        )

    def bind_tools(self, *args: Any, **kwargs: Any) -> "ModeloConRespaldo":
        return self._en_todos("bind_tools", *args, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "ModeloConRespaldo":
        return self._en_todos("with_structured_output", *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Atributos simples (model_name, profile...) se leen del modelo principal.
        return getattr(self.__dict__.get("runnable") or object.__getattribute__(self, "runnable"), name)


def con_respaldo(principal: Any, respaldos: Sequence[Any]) -> Any:
    """El modelo principal con `respaldos` en orden; sin respaldos, el mismo modelo.

    Los errores transitorios por minuto (429/503) los sigue reintentando la
    RetryPolicy del nodo; esto cubre lo que no se arregla esperando.
    """
    return ModeloConRespaldo(runnable=principal, fallbacks=list(respaldos)) if respaldos else principal


def crear_llms_por_rol() -> dict[str, Any]:
    """Un modelo por rol, cada uno con los modelos de LLM_MODELOS_RESPALDO como respaldo."""
    instancias: dict[str, BaseChatModel] = {}

    def instancia(modelo: str) -> BaseChatModel:
        return instancias.setdefault(modelo, crear_llm(modelo))

    return {
        rol: con_respaldo(instancia(modelo), [instancia(m) for m in config.LLM_MODELOS_RESPALDO if m != modelo])
        for rol, modelo in config.MODELOS_POR_ROL.items()
    }
