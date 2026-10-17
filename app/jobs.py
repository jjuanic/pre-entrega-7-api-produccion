"""
Estado de los jobs y cola de trabajo en Redis (todo con `redis.asyncio`:
ninguna operación bloquea el event loop de FastAPI).

Claves:
    job:{id}               HASH   registro del job (cada campo en JSON)
    jobs:index             ZSET   job_ids por fecha de creación (para listar)
    cola:tareas            LIST   mensajes {"tipo": "run"|"resume", "job_id": ...}
    cola:tareas:procesando:{worker}  LIST  mensajes que tomó ESE worker y todavía no terminó

Reglas que evitan los errores típicos del patrón:
- El estado PENDING se escribe ANTES de encolar, en la misma transacción
  MULTI/EXEC: un worker rápido nunca encuentra un job que "no existe" y el
  cliente nunca recibe un 404 por una carrera.
- La cola es "confiable": el worker toma el mensaje con BLMOVE (lo pasa a
  `procesando`) y lo borra recién al terminar. Si el proceso muere a la
  mitad, el mensaje sigue en `procesando` y se puede recuperar al reiniciar.
- La reanudación (aprobación humana) usa WATCH: solo un request puede pasar
  el job de AWAITING_APPROVAL a PENDING.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import WatchError

import config
from app.schemas import EstadoJob, Job

INDICE = "jobs:index"


def ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clave(job_id: str) -> str:
    return f"job:{job_id}"


def _codificar(campos: dict[str, Any]) -> dict[str, str]:
    salida = {}
    for k, v in campos.items():
        if hasattr(v, "model_dump"):
            v = v.model_dump(mode="json")
        salida[k] = json.dumps(v, ensure_ascii=False, default=str)
    return salida


def _decodificar(crudo: dict[str, str]) -> dict[str, Any]:
    return {k: json.loads(v) for k, v in crudo.items()}


class JobStore:
    def __init__(self, redis: Redis, cola: str = config.COLA_TAREAS, ttl: int = config.JOB_TTL_SEGUNDOS) -> None:
        self.redis = redis
        self.cola = cola
        self.ttl = ttl

    # --- escritura ---------------------------------------------------------

    async def crear_y_encolar(self, job: Job) -> None:
        """Registra el job como PENDING y lo encola, atómicamente y en ese orden."""
        mensaje = json.dumps({"tipo": "run", "job_id": job.job_id})
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(_clave(job.job_id), mapping=_codificar(job.model_dump(mode="json")))
            pipe.expire(_clave(job.job_id), self.ttl)
            pipe.zadd(INDICE, {job.job_id: datetime.now(timezone.utc).timestamp()})
            pipe.rpush(self.cola, mensaje)
            await pipe.execute()

    async def actualizar(self, job_id: str, **campos: Any) -> None:
        campos["updated_at"] = ahora()
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(_clave(job_id), mapping=_codificar(campos))
            pipe.expire(_clave(job_id), self.ttl)
            await pipe.execute()

    async def encolar_reanudacion(self, job_id: str, decision: dict[str, Any]) -> bool | str | None:
        """AWAITING_APPROVAL -> PENDING + mensaje "resume" en la cola.

        Devuelve True si se encoló, None si el job no existe, o el estado
        actual (str) si no estaba esperando aprobación.
        """
        clave = _clave(job_id)
        mensaje = json.dumps({"tipo": "resume", "job_id": job_id, "decision": decision})
        async with self.redis.pipeline(transaction=True) as pipe:
            for _ in range(3):
                try:
                    await pipe.watch(clave)
                    crudo = await pipe.hget(clave, "status")
                    if crudo is None:
                        return None
                    estado = json.loads(crudo)
                    if estado != EstadoJob.AWAITING_APPROVAL:
                        return estado
                    pipe.multi()
                    pipe.hset(clave, mapping=_codificar({
                        "status": EstadoJob.PENDING, "aprobacion": decision, "updated_at": ahora(),
                    }))
                    pipe.rpush(self.cola, mensaje)
                    await pipe.execute()
                    return True
                except WatchError:  # otro request cambió el job en el medio: releer
                    continue
        return EstadoJob.PENDING

    # --- lectura -----------------------------------------------------------

    async def obtener(self, job_id: str) -> Job | None:
        crudo = await self.redis.hgetall(_clave(job_id))
        return Job.model_validate(_decodificar(crudo)) if crudo else None

    async def listar(self, limite: int = 20, estado: EstadoJob | None = None) -> list[Job]:
        ids = await self.redis.zrevrange(INDICE, 0, max(limite * 5, limite) - 1)
        jobs: list[Job] = []
        for job_id in ids:
            job = await self.obtener(job_id)
            if job is None:  # expiró el TTL: limpiar el índice
                await self.redis.zrem(INDICE, job_id)
                continue
            if estado is None or job.status == estado:
                jobs.append(job)
            if len(jobs) >= limite:
                break
        return jobs

    async def largo_cola(self) -> dict[str, int]:
        procesando = 0
        async for clave in self.redis.scan_iter(match=f"{self.cola}:procesando:*"):
            procesando += await self.redis.llen(clave)
        return {"en_cola": await self.redis.llen(self.cola), "procesando": procesando}

    # --- cola (lado worker) -------------------------------------------------

    def lista_procesando(self, worker: str) -> str:
        return f"{self.cola}:procesando:{worker}"

    async def tomar(self, worker: str, timeout: float = 1.0) -> str | None:
        """Bloquea hasta `timeout` s esperando un mensaje (BLMOVE es async: no frena el loop)."""
        return await self.redis.blmove(self.cola, self.lista_procesando(worker), timeout, "LEFT", "RIGHT")

    async def confirmar(self, worker: str, mensaje: str) -> None:
        await self.redis.lrem(self.lista_procesando(worker), 1, mensaje)

    async def recuperar_huerfanos(self, worker: str) -> int:
        """Devuelve a la cola los mensajes que ESTE worker no llegó a terminar (se cayó)."""
        movidos = 0
        while await self.redis.lmove(self.lista_procesando(worker), self.cola, "RIGHT", "LEFT"):
            movidos += 1
        return movidos
