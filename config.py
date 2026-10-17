"""
Configuración centralizada del orquestador.

Todo lo que depende del entorno (proveedor de LLM, API keys, límites) se lee
de variables de entorno / `.env`, para que ninguna credencial quede escrita
en el código ni se suba al repositorio.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Literal

from dotenv import load_dotenv

load_dotenv()

Proveedor = Literal["google", "openai", "anthropic"]

# --- Rutas ---------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
TRAZAS_DIR: Final[Path] = BASE_DIR / "trazas"

# --- LLM -----------------------------------------------------------------
_PROVEEDORES_VALIDOS: Final[tuple[Proveedor, ...]] = ("google", "openai", "anthropic")

_proveedor_env = os.getenv("LLM_PROVIDER", "google").strip().lower()
if _proveedor_env not in _PROVEEDORES_VALIDOS:
    raise RuntimeError(
        f"LLM_PROVIDER='{_proveedor_env}' no es válido. Opciones: {', '.join(_PROVEEDORES_VALIDOS)}."
    )
LLM_PROVIDER: Final[Proveedor] = _proveedor_env  # type: ignore[assignment]

MODELOS_POR_DEFECTO: Final[dict[Proveedor, str]] = {
    "google": "gemini-flash-latest",
    "openai": "gpt-5-mini",
    "anthropic": "claude-sonnet-5",
}
LLM_MODEL: Final[str] = os.getenv("LLM_MODEL") or MODELOS_POR_DEFECTO[LLM_PROVIDER]
LLM_TEMPERATURE: Final[float] = 0.0

# Modelo por rol del grafo (opcional). Sin definir, todos usan LLM_MODEL.
ROLES_LLM: Final[tuple[str, ...]] = ("supervisor", "investigador", "analista", "sintesis")
MODELOS_POR_ROL: Final[dict[str, str]] = {
    rol: os.getenv(f"LLM_MODELO_{rol.upper()}") or LLM_MODEL for rol in ROLES_LLM
}
# Modelos de respaldo (en orden) si el del rol falla, p. ej. por cuota agotada.
LLM_MODELOS_RESPALDO: Final[tuple[str, ...]] = tuple(
    m.strip() for m in os.getenv("LLM_MODELOS_RESPALDO", "").split(",") if m.strip()
)

API_KEY_POR_PROVEEDOR: Final[dict[Proveedor, str]] = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# --- Condiciones de parada (anti "Supervisor Infinito") -------------------
# Máximo de delegaciones a especialistas en una misma tarea.
MAX_PASOS: Final[int] = int(os.getenv("MAX_PASOS", "6"))
# Máximo de veces que se puede invocar a un mismo especialista (1 intento +
# refinamientos). Si se agota y su output sigue inválido -> revisión humana.
MAX_INTENTOS_POR_AGENTE: Final[int] = int(os.getenv("MAX_INTENTOS_POR_AGENTE", "3"))
# Techo de super-steps de LangGraph: red de seguridad por si fallara la lógica anterior.
RECURSION_LIMIT: Final[int] = int(os.getenv("RECURSION_LIMIT", "40"))

# --- Validación determinística -------------------------------------------
CONFIANZA_MINIMA: Final[float] = float(os.getenv("CONFIANZA_MINIMA", "0.5"))

# Reintentos ante errores transitorios del proveedor (429 / 503), con backoff.
LLM_RETRY_INTENTOS: Final[int] = int(os.getenv("LLM_RETRY_INTENTOS", "4"))
LLM_RETRY_INTERVALO_INICIAL: Final[float] = float(os.getenv("LLM_RETRY_INTERVALO_INICIAL", "20"))


def require_api_key() -> str:
    """Falla rápido y con un mensaje claro si falta la API key del proveedor."""
    variable = API_KEY_POR_PROVEEDOR[LLM_PROVIDER]
    valor = os.getenv(variable)
    if not valor:
        raise RuntimeError(
            f"Falta {variable} (LLM_PROVIDER={LLM_PROVIDER}). Copiá .env.example a .env y completá la clave."
        )
    return valor


# =========================================================================
# Pre-entrega 7: API de producción (Redis, worker, HITL, observabilidad)
# =========================================================================

def _bool(nombre: str, default: bool) -> bool:
    return os.getenv(nombre, str(default)).strip().lower() in ("1", "true", "yes", "si", "sí")


# --- Redis ---------------------------------------------------------------
# Redis 8 (o Redis Stack): el checkpointer de LangGraph usa los módulos JSON y Search.
REDIS_URL: Final[str] = os.getenv("REDIS_URL", "redis://localhost:6379/0")
COLA_TAREAS: Final[str] = os.getenv("COLA_TAREAS", "cola:tareas")
# Cuánto vive el registro de un job en Redis (segundos). 7 días por defecto.
JOB_TTL_SEGUNDOS: Final[int] = int(os.getenv("JOB_TTL_SEGUNDOS", str(7 * 24 * 3600)))
# TTL de los checkpoints de LangGraph (minutos). Tiene que superar el tiempo que
# un job puede quedar esperando aprobación humana.
CHECKPOINT_TTL_MINUTOS: Final[int] = int(os.getenv("CHECKPOINT_TTL_MINUTOS", str(7 * 24 * 60)))

# --- Worker --------------------------------------------------------------
# Tareas que un proceso worker ejecuta a la vez (consumidores asyncio de la cola).
WORKER_CONCURRENCIA: Final[int] = int(os.getenv("WORKER_CONCURRENCIA", "5"))
# Si un job tarda más que esto, se cancela y queda FAILED (evita jobs colgados).
JOB_TIMEOUT_SEGUNDOS: Final[float] = float(os.getenv("JOB_TIMEOUT_SEGUNDOS", "900"))
# true = la API levanta los consumidores dentro de su propio proceso (modo simple).
# false = se corre `python -m app.worker` aparte (recomendado; así lo hace docker-compose).
WORKER_EMBEBIDO: Final[bool] = _bool("WORKER_EMBEBIDO", False)
# Permite el campo `simular_falla` del request (para demostrar el paso a FAILED).
PERMITIR_FALLAS_SIMULADAS: Final[bool] = _bool("PERMITIR_FALLAS_SIMULADAS", True)

# --- Human-in-the-loop ---------------------------------------------------
# Una tarea es CRÍTICA (pausa obligatoria) si pide publicar el informe, si el
# supervisor la escaló a revisión humana, o si su costo supera estos umbrales.
HITL_MAX_TOKENS: Final[int] = int(os.getenv("HITL_MAX_TOKENS", "60000"))
HITL_MAX_COSTO_USD: Final[float] = float(os.getenv("HITL_MAX_COSTO_USD", "0.05"))
# Precio por millón de tokens, solo para la estimación interna del umbral de costo
# (el costo "oficial" lo calcula la plataforma de observabilidad).
PRECIO_INPUT_USD_POR_MTOK: Final[float] = float(os.getenv("PRECIO_INPUT_USD_POR_MTOK", "0.30"))
PRECIO_OUTPUT_USD_POR_MTOK: Final[float] = float(os.getenv("PRECIO_OUTPUT_USD_POR_MTOK", "2.50"))
STREAM_PUBLICACIONES: Final[str] = os.getenv("STREAM_PUBLICACIONES", "informes:publicados")

# --- Observabilidad ------------------------------------------------------
# phoenix | langsmith | none
OBSERVABILIDAD: Final[str] = os.getenv("OBSERVABILIDAD", "phoenix").strip().lower()
PHOENIX_COLLECTOR_ENDPOINT: Final[str] = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
PHOENIX_PROJECT_NAME: Final[str] = os.getenv("PHOENIX_PROJECT_NAME", "orquestador-agentes")
