"""
Capa de observabilidad: Arize Phoenix (OpenInference/OpenTelemetry) o LangSmith.

Phoenix (por defecto, `OBSERVABILIDAD=phoenix`):
    `phoenix.otel.register()` crea el TracerProvider con un exporter OTLP
    hacia el colector (localhost:6006) y `LangChainInstrumentor` envuelve
    todo lo que corre LangChain/LangGraph: cada nodo del grafo, cada
    sub-agente `create_agent`, cada llamada al LLM (con tokens de entrada y
    salida, que Phoenix usa para calcular el costo) y cada herramienta.

LangSmith (`OBSERVABILIDAD=langsmith`):
    alcanza con LANGSMITH_TRACING=true + LANGSMITH_API_KEY: LangChain manda
    los runs solo. Los `tags`/`metadata` que pone el worker en el config
    aparecen en los dos dashboards.

Además de la instrumentación automática hay dos herramientas manuales:

- `@trazado(...)`: decorador para funciones propias (el job del worker, la
  publicación del informe). Abre un span con su input/output y marca el
  error si la función lanza una excepción.
- `contexto_job(...)`: pone `session.id = job_id` en todos los spans del job,
  así la corrida inicial y la reanudación después de la aprobación humana
  quedan agrupadas en la misma sesión de Phoenix.

`UsoTokens` cuenta los tokens de cada job (callback de LangChain) para la
estimación interna de costo que usa el nodo Human-in-the-loop.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar
from typing import Any, ParamSpec, TypeVar

from langchain_core.callbacks import UsageMetadataCallbackHandler
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

import config

logger = logging.getLogger("orquestador.observabilidad")

P = ParamSpec("P")
R = TypeVar("R")

_TRACER_NAME = "orquestador-api"
_proveedor: Any = None  # TracerProvider de Phoenix, para hacer flush al apagar


def configurar_observabilidad(servicio: str) -> str:
    """Inicializa la plataforma elegida. Devuelve el backend activo."""
    global _proveedor
    backend = config.OBSERVABILIDAD
    if backend == "phoenix":
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        endpoint = config.PHOENIX_COLLECTOR_ENDPOINT.rstrip("/") + "/v1/traces"
        # La API manda a un proyecto aparte: FastAPI emite un span por cada request
        # (telemetría nativa) y los cientos de GET de polling ensuciarían la
        # latencia p95 y el conteo de trazas del agente. Así cada proyecto mide una cosa:
        #   <proyecto>      -> una traza por segmento de job (grafo, nodos, LLM, tools)
        #   <proyecto>-api  -> latencia de los endpoints (el POST tarda milisegundos)
        proyecto = config.PHOENIX_PROJECT_NAME if servicio != "api" else f"{config.PHOENIX_PROJECT_NAME}-api"
        _proveedor = register(
            project_name=proyecto,
            endpoint=endpoint,
            batch=True,  # exporta en segundo plano: no agrega latencia a los nodos
            set_global_tracer_provider=True,
            verbose=False,
        )
        instrumentor = LangChainInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument(tracer_provider=_proveedor)
        logger.info("Observabilidad: Phoenix (%s) proyecto=%s servicio=%s", endpoint, proyecto, servicio)
    elif backend == "langsmith":
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        if not os.getenv("LANGSMITH_API_KEY"):
            logger.warning("OBSERVABILIDAD=langsmith pero falta LANGSMITH_API_KEY: no se van a enviar trazas.")
        logger.info("Observabilidad: LangSmith proyecto=%s", os.getenv("LANGSMITH_PROJECT", "default"))
    else:
        backend = "none"
        logger.info("Observabilidad desactivada (OBSERVABILIDAD=none).")
    return backend


def apagar_observabilidad() -> None:
    """Fuerza el envío de los spans pendientes (si no, los últimos se pierden al salir)."""
    if _proveedor is not None:
        with contextlib.suppress(Exception):
            _proveedor.force_flush()
            _proveedor.shutdown()


def _a_texto(valor: Any) -> str:
    if isinstance(valor, str):
        return valor
    try:
        return json.dumps(valor, ensure_ascii=False, default=str)[:8000]
    except Exception:
        return str(valor)[:8000]


def trazado(
    nombre: str,
    tipo: str = "CHAIN",
    entrada: Callable[..., Any] | None = None,
    salida: Callable[[Any], Any] | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorador: envuelve una corrutina en un span OpenInference.

    `tipo` es el `openinference.span.kind` (CHAIN, TOOL, AGENT, LLM...).
    `entrada(*args, **kwargs)` y `salida(resultado)` eligen qué se muestra como
    input/output del span (por defecto, los kwargs y el resultado).
    Si no hay TracerProvider configurado, el tracer de OpenTelemetry es no-op.
    """

    def decorador(funcion: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(funcion)
        async def envuelta(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = trace.get_tracer(_TRACER_NAME)
            with tracer.start_as_current_span(nombre) as span:
                span.set_attribute("openinference.span.kind", tipo)
                valor_entrada = entrada(*args, **kwargs) if entrada else (kwargs or list(args[1:]))
                if valor_entrada:
                    span.set_attribute("input.value", _a_texto(valor_entrada))
                try:
                    resultado = await funcion(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
                    raise
                span.set_attribute("output.value", _a_texto(salida(resultado) if salida else resultado))
                span.set_status(Status(StatusCode.OK))
                return resultado

        return envuelta

    return decorador


@contextlib.contextmanager
def contexto_job(job_id: str, metadata: dict[str, Any], tags: list[str]) -> Iterator[None]:
    """Propaga session.id / metadata / tags de OpenInference a todos los spans hijos."""
    try:
        from openinference.instrumentation import using_attributes
    except ImportError:  # sin OpenInference instalado: no-op
        yield
        return
    with using_attributes(session_id=job_id, metadata=metadata, tags=tags):
        yield


# --- Conteo de tokens y costo estimado -----------------------------------

class UsoTokens:
    """Acumula los tokens de TODAS las llamadas al LLM de un job.

    Es un callback de LangChain: el worker lo pasa en `config["callbacks"]`
    y se propaga a los sub-agentes. También se publica en un ContextVar para
    que el nodo `evaluar_riesgo` lo lea sin cambiar la firma de los nodos.
    """

    def __init__(self) -> None:
        self.handler = UsageMetadataCallbackHandler()

    @property
    def tokens_entrada(self) -> int:
        return sum(u.get("input_tokens", 0) for u in self.handler.usage_metadata.values())

    @property
    def tokens_salida(self) -> int:
        return sum(u.get("output_tokens", 0) for u in self.handler.usage_metadata.values())

    @property
    def tokens_totales(self) -> int:
        return self.tokens_entrada + self.tokens_salida

    @property
    def costo_estimado_usd(self) -> float:
        return round(
            self.tokens_entrada / 1e6 * config.PRECIO_INPUT_USD_POR_MTOK
            + self.tokens_salida / 1e6 * config.PRECIO_OUTPUT_USD_POR_MTOK,
            6,
        )

    @property
    def modelos(self) -> list[str]:
        return sorted(self.handler.usage_metadata)


USO_ACTUAL: ContextVar[UsoTokens | None] = ContextVar("uso_tokens_actual", default=None)
