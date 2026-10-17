"""
Estado compartido del orquestador: el CONTRATO entre agentes.

Dos capas:

1. Artefactos Pydantic (lo que produce cada especialista). Se validan ANTES
   de entrar al estado: si el LLM devuelve un `confidence_score` de 1.7 o una
   lista de hallazgos vacía, el error se captura en el borde del especialista
   y no contamina al resto del sistema.

2. `AgentState(MessagesState)`: el TypedDict que fluye por el grafo.
   - Aislamiento de escritura: cada nodo es dueño de una "parcela" del estado
     (`PARCELAS`). El investigador solo escribe `research_output`, el analista
     solo `analysis_output`, el validador solo `validaciones`, etc. `graph.py`
     lo hace cumplir en tiempo de ejecución, así ningún agente pisa lo que
     calculó otro.
   - Trazabilidad: `contribuciones` y `decisiones` usan `operator.add` como
     reducer, entonces cada nodo AGREGA su registro en vez de reemplazar la
     lista. Al final queda el log completo de quién aportó qué y quién decidió
     qué y por qué.
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Any, Literal, TypeVar

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, field_validator

M = TypeVar("M", bound=BaseModel)

# --- Nombres de nodos ------------------------------------------------------

Especialista = Literal["investigador", "analista"]
Destino = Literal["investigador", "analista", "sintesis"]
ESPECIALISTAS: tuple[Especialista, ...] = ("investigador", "analista")


# --- Artefactos del Investigador ---------------------------------------------

class Hallazgo(BaseModel):
    """Un dato concreto encontrado en la documentación, con su procedencia."""

    dato: str = Field(min_length=3, description="El dato concreto, en una oración. Ej: 'ThreadPoolExecutor usa min(32, os.cpu_count() + 4) hilos por defecto'.")
    fuente: str = Field(min_length=3, description="Archivo de donde sale el dato, EXACTAMENTE como lo devolvió la herramienta. Ej: 'concurrent_futures.md'.")
    cita: str = Field(min_length=3, description="Fragmento textual (copiado, no parafraseado) del documento que respalda el dato.")


class ResearchArtifact(BaseModel):
    """Salida estructurada del Agente Investigador."""

    hallazgos: list[Hallazgo] = Field(min_length=1, description="Datos encontrados, cada uno con fuente y cita.")
    resumen: str = Field(min_length=10, description="Resumen breve de lo encontrado, sin cálculos.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Qué tan seguro estás de que los hallazgos responden la instrucción (0 a 1).")


# --- Artefactos del Analista -----------------------------------------------

class Calculo(BaseModel):
    """Un cálculo reproducible: el validador vuelve a evaluar la expresión."""

    descripcion: str = Field(min_length=3, description="Qué se calcula. Ej: 'Tiempo secuencial de 240 llamadas de 0.5 s'.")
    expresion: str = Field(min_length=1, description="La expresión EXACTA que pasaste a la calculadora. Ej: '240 * 0.5'.")
    resultado: float = Field(description="El resultado que devolvió la calculadora.")


class AnalysisArtifact(BaseModel):
    """Salida estructurada del Agente Analista."""

    calculos: list[Calculo] = Field(min_length=1, description="Cálculos realizados con la calculadora.")
    conclusion: str = Field(min_length=10, description="Interpretación de los resultados, comparando escenarios si corresponde.")
    supuestos: list[str] = Field(default_factory=list, description="Supuestos que hiciste (ej: 'latencia constante').")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Qué tan seguro estás del análisis (0 a 1).")


# --- Parcelas del estado (lo que escribe cada nodo) ---------------------------

class Evidencia(BaseModel):
    """Fragmento que la herramienta de búsqueda devolvió de verdad (provenance).

    Lo captura el nodo a partir del `artifact` de los ToolMessage, NO lo
    declara el LLM: por eso sirve para detectar fuentes inventadas.
    """

    fuente: str
    seccion: str = ""
    texto: str


class ResearchOutput(BaseModel):
    """Parcela del investigador: `state["research_output"]`."""

    intento: int = Field(ge=1)
    instruccion: str
    artefacto: ResearchArtifact | None = None
    evidencia: list[Evidencia] = Field(default_factory=list)
    error: str | None = None

    @property
    def fuentes_consultadas(self) -> set[str]:
        return {e.fuente for e in self.evidencia}


class CalculoEjecutado(BaseModel):
    """Llamada real a la calculadora (capturada del ToolMessage)."""

    expresion: str
    resultado: float | None
    error: str | None = None


class AnalysisOutput(BaseModel):
    """Parcela del analista: `state["analysis_output"]`."""

    intento: int = Field(ge=1)
    instruccion: str
    artefacto: AnalysisArtifact | None = None
    calculos_ejecutados: list[CalculoEjecutado] = Field(default_factory=list)
    error: str | None = None


class ValidationReport(BaseModel):
    """Resultado de la validación DETERMINÍSTICA (código, no LLM) de una parcela."""

    agente: Especialista
    intento: int
    valido: bool
    errores: list[str] = Field(default_factory=list)
    advertencias: list[str] = Field(default_factory=list)


class DecisionSupervisor(BaseModel):
    """Salida estructurada del LLM del Supervisor (with_structured_output)."""

    next: Literal["investigador", "analista", "FINISH"] = Field(
        description="Próximo especialista a invocar, o FINISH si la tarea ya está completa y validada."
    )
    razon: str = Field(description="Justificación breve de la decisión, citando el estado de las validaciones.")
    instruccion: str = Field(
        default="",
        description="Instrucción específica y autocontenida para el especialista elegido (vacía si es FINISH).",
    )


class RegistroDecision(BaseModel):
    """Log de cada decisión del supervisor: quién decidió qué y por qué."""

    paso: int
    propuesta_llm: str  # lo que propuso el LLM ("-" si no se lo consultó)
    destino: Destino  # adónde se ruteó finalmente
    razon: str
    instruccion: str = ""
    regla_aplicada: str | None = None  # guarda determinística que corrigió al LLM, si hubo


class Contribucion(BaseModel):
    """Registro de quién aportó qué (append-only)."""

    agente: str
    tipo: Literal["investigacion", "analisis", "validacion", "sintesis"]
    intento: int = 1
    detalle: str
    momento: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @field_validator("detalle")
    @classmethod
    def _recortar(cls, v: str) -> str:
        return v if len(v) <= 600 else v[:597] + "..."


# --- Human-in-the-loop (Pre-entrega 7) -----------------------------------

class EvaluacionRiesgo(BaseModel):
    """Resultado del nodo `evaluar_riesgo`: si la tarea es CRÍTICA y por qué.

    Se calcula en un nodo propio (y queda en el checkpoint) para que la
    decisión sea estable: cuando se reanuda el grafo después de la pausa, el
    nodo de aprobación no vuelve a evaluar nada, solo lee esto.
    """

    critica: bool
    motivos: list[str] = Field(default_factory=list)
    accion: Literal["publicar_informe", "entregar_respuesta"]
    tokens_totales: int = 0
    costo_estimado_usd: float = 0.0


class DecisionHumana(BaseModel):
    """Lo que manda el revisor por `POST /tasks/{id}/approve` (valor del `Command(resume=...)`)."""

    aprobado: bool
    revisor: str = Field(default="anonimo", min_length=1, max_length=80)
    comentario: str = Field(default="", max_length=2000)
    momento: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class Publicacion(BaseModel):
    """Efecto secundario ejecutado por el nodo `publicar` (solo con aprobación)."""

    destino: str
    id_publicacion: str
    duplicada: bool = False  # True si ya se había publicado (idempotencia ante reintentos)


def como(modelo: type[M], valor: Any) -> M | None:
    """Devuelve `valor` como instancia de `modelo` (acepta el objeto o un dict).

    El checkpointer de Redis reconstruye los modelos de `state.py` porque están
    en su allowlist (ver `app.graph.checkpointer_redis`). Esto es una red de
    seguridad para lecturas que llegan como dict plano (p. ej. un checkpoint
    viejo o un serializer sin allowlist).
    """
    if valor is None or isinstance(valor, modelo):
        return valor
    return modelo.model_validate(valor)


# --- Reducers -----------------------------------------------------------

def sumar_conteos(actual: dict[str, int] | None, nuevo: dict[str, int] | None) -> dict[str, int]:
    """Suma contadores por clave: {"investigador": 1} + {"investigador": 1} = 2."""
    resultado = dict(actual or {})
    for clave, valor in (nuevo or {}).items():
        resultado[clave] = resultado.get(clave, 0) + valor
    return resultado


def fusionar(actual: dict[str, ValidationReport] | None, nuevo: dict[str, ValidationReport] | None) -> dict[str, ValidationReport]:
    """Fusiona por clave: la validación nueva de un agente reemplaza solo la suya."""
    return {**(actual or {}), **(nuevo or {})}


# --- Estado del grafo ---------------------------------------------------

class AgentState(MessagesState):
    """Estado compartido. Hereda `messages` (reducer `add_messages`)."""

    tarea: str  # pedido original del usuario (inmutable)

    # Control de flujo (parcela del supervisor)
    next_agent: Destino | None
    instruccion: str  # instrucción puntual para el próximo especialista
    pasos: int  # delegaciones realizadas (condición de parada)
    task_completed: bool
    requiere_revision_humana: bool
    motivo_escalamiento: str | None
    decisiones: Annotated[list[RegistroDecision], operator.add]

    # Parcelas de los especialistas (aislamiento de escritura)
    research_output: ResearchOutput | None
    analysis_output: AnalysisOutput | None
    intentos: Annotated[dict[str, int], sumar_conteos]

    # Parcela del validador
    validaciones: Annotated[dict[str, ValidationReport], fusionar]

    # Trazabilidad compartida (append-only) y salida
    contribuciones: Annotated[list[Contribucion], operator.add]
    respuesta_final: str

    # Human-in-the-loop y efectos secundarios (Pre-entrega 7)
    publicar: bool  # el usuario pidió publicar el informe (acción con efectos secundarios)
    riesgo: EvaluacionRiesgo | None
    aprobacion: DecisionHumana | None
    publicacion: Publicacion | None


# Qué claves puede escribir cada nodo. `graph.py` rechaza cualquier escritura
# fuera de la parcela propia (evita condiciones de carrera entre agentes).
PARCELAS: dict[str, frozenset[str]] = {
    "supervisor": frozenset({
        "next_agent", "instruccion", "pasos", "task_completed",
        "requiere_revision_humana", "motivo_escalamiento", "decisiones",
    }),
    "investigador": frozenset({"research_output", "intentos", "contribuciones", "messages"}),
    "analista": frozenset({"analysis_output", "intentos", "contribuciones", "messages"}),
    "validador": frozenset({"validaciones", "contribuciones"}),
    "sintesis": frozenset({"respuesta_final", "contribuciones", "messages", "task_completed"}),
    "evaluar_riesgo": frozenset({"riesgo"}),
    "aprobacion_humana": frozenset({"aprobacion"}),
    "publicar": frozenset({"publicacion"}),
}


def estado_inicial(tarea: str, publicar: bool = False) -> AgentState:
    return {
        "messages": [HumanMessage(content=tarea)],
        "tarea": tarea,
        "next_agent": None,
        "instruccion": "",
        "pasos": 0,
        "task_completed": False,
        "requiere_revision_humana": False,
        "motivo_escalamiento": None,
        "decisiones": [],
        "research_output": None,
        "analysis_output": None,
        "intentos": {},
        "validaciones": {},
        "contribuciones": [],
        "respuesta_final": "",
        "publicar": publicar,
        "riesgo": None,
        "aprobacion": None,
        "publicacion": None,
    }
