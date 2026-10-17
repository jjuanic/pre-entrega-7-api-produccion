"""Contratos de la API (requests, responses y estados del job)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EstadoJob(StrEnum):
    PENDING = "PENDING"                      # registrado y en la cola (o esperando reanudarse)
    RUNNING = "RUNNING"                      # un worker está ejecutando el grafo
    AWAITING_APPROVAL = "AWAITING_APPROVAL"  # pausado en el nodo Human-in-the-loop
    DONE = "DONE"                            # terminó (y si era crítica, fue aprobada)
    REJECTED = "REJECTED"                    # el revisor humano rechazó la acción crítica
    FAILED = "FAILED"                        # excepción o timeout: el error queda en `error`


TERMINALES: frozenset[EstadoJob] = frozenset({EstadoJob.DONE, EstadoJob.REJECTED, EstadoJob.FAILED})


class TareaRequest(BaseModel):
    tarea: str = Field(
        min_length=10,
        max_length=4000,
        description="Consulta técnica sobre Python que necesita investigación en la documentación y cálculos.",
        examples=["¿Cuántos hilos usa ThreadPoolExecutor por defecto? Estimá cuánto tardan 240 llamadas de 0,5 s con ese pool."],
    )
    publicar: bool = Field(
        default=False,
        description="Publicar el informe final en el canal compartido. Es una acción con efectos secundarios: "
        "la tarea se vuelve CRÍTICA y se pausa para aprobación humana.",
    )
    simular_falla: Literal["excepcion"] | None = Field(
        default=None,
        description="Solo para demo (PERMITIR_FALLAS_SIMULADAS=true): el worker lanza una excepción no "
        "controlada para mostrar el paso a FAILED.",
    )


class TareaEncolada(BaseModel):
    job_id: str
    status: EstadoJob
    links: dict[str, str] = Field(default_factory=dict)


class AprobacionRequest(BaseModel):
    aprobado: bool
    revisor: str = Field(default="anonimo", min_length=1, max_length=80)
    comentario: str = Field(default="", max_length=2000)


class UsoJob(BaseModel):
    tokens_entrada: int = 0
    tokens_salida: int = 0
    tokens_totales: int = 0
    costo_estimado_usd: float = 0.0
    modelos: list[str] = Field(default_factory=list)


class Job(BaseModel):
    """Registro completo de un job tal como queda en Redis (`job:{id}`)."""

    job_id: str
    status: EstadoJob
    tarea: str
    publicar: bool = False
    simular_falla: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duracion_ejecucion_segundos: float = 0.0  # tiempo de worker (sin contar la espera humana)
    ejecuciones: int = 0  # segmentos ejecutados: 1 = corrida inicial, 2 = + reanudación
    worker: str | None = None
    aprobacion_pendiente: dict[str, Any] | None = None
    aprobacion: dict[str, Any] | None = None
    resultado: dict[str, Any] | None = None
    error: str | None = None
    uso: UsoJob = Field(default_factory=UsoJob)
