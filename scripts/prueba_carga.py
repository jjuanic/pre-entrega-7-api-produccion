"""
Prueba de carga: N peticiones CONCURRENTES contra la API y polling hasta que terminen.

    python scripts/prueba_carga.py                 # 5 tareas concurrentes
    python scripts/prueba_carga.py --con-hitl      # la 1ra pide publicar: se pausa y el script la aprueba
    python scripts/prueba_carga.py --n 10 --url http://localhost:8000

Mide desde el lado del cliente:
- latencia del POST (tiene que ser de milisegundos: la API solo encola);
- latencia de punta a punta de cada job y su tiempo de ejecución en el worker;
- tokens y costo estimado que registró el worker;
- p50 / p95 de esas latencias.

El costo "oficial" y la latencia p95 por traza se leen en el dashboard de
Phoenix (o con `python scripts/metricas_phoenix.py`). El resultado queda en
`resultados/carga_<fecha>.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

TERMINALES = {"DONE", "REJECTED", "FAILED"}

TAREAS = [
    "Tengo que hacer 240 llamadas HTTP de 0,5 segundos cada una en una máquina de 8 núcleos. Según la "
    "documentación, ¿cuántos hilos usa ThreadPoolExecutor si no le paso max_workers? Estimá cuánto tardaría "
    "secuencialmente y con ese pool por defecto, y cuánto más rápido es.",
    "Según la documentación, ¿cuál es el maxsize por defecto de functools.lru_cache? Si una función cacheada "
    "recibe 1000 argumentos distintos, ¿qué porcentaje de esos resultados puede quedar guardado como máximo "
    "con ese valor por defecto, y cuántos se descartan?",
    "¿Cuántos procesos usa ProcessPoolExecutor por defecto según la documentación? En una máquina de 8 núcleos, "
    "si tengo 100 tareas CPU-bound de 2 segundos cada una, estimá el tiempo total secuencial y con ese pool.",
    "Según la documentación de asyncio, ¿a partir de cuántos milisegundos el modo debug considera lento un "
    "callback? Si tengo 40 callbacks de 250 ms, ¿cuánto tiempo en total bloquean el event loop y cuántas veces "
    "supera cada uno ese umbral?",
    "Según la documentación de logging, ¿cuál es el nivel por defecto del logger raíz? Si mi aplicación emite "
    "1200 mensajes por hora (70% INFO, 20% WARNING y 10% ERROR), ¿cuántos se ven por hora y por día con esa "
    "configuración por defecto?",
]


def percentil(valores: list[float], p: float) -> float:
    if not valores:
        return 0.0
    if len(valores) == 1:
        return valores[0]
    return statistics.quantiles(valores, n=100, method="inclusive")[round(p) - 1]


async def correr_job(cliente: httpx.AsyncClient, indice: int, tarea: str, publicar: bool, intervalo: float, timeout: float) -> dict[str, Any]:
    inicio = time.perf_counter()
    r = await cliente.post("/tasks", json={"tarea": tarea, "publicar": publicar})
    latencia_post_ms = (time.perf_counter() - inicio) * 1000
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"  [{indice}] POST /tasks -> 202 en {latencia_post_ms:6.1f} ms  job_id={job_id}")

    estados_vistos: list[str] = []
    aprobado = False
    while True:
        job = (await cliente.get(f"/tasks/{job_id}")).json()
        if not estados_vistos or estados_vistos[-1] != job["status"]:
            estados_vistos.append(job["status"])
            print(f"  [{indice}] {job['status']}  (+{time.perf_counter() - inicio:5.1f} s)")
        if job["status"] == "AWAITING_APPROVAL" and not aprobado:
            motivos = "; ".join(job["aprobacion_pendiente"]["motivos"])
            print(f"  [{indice}] HITL: {motivos}\n  [{indice}] -> POST /tasks/{job_id}/approve (aprobado)")
            await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "prueba-carga", "comentario": "OK"})
            aprobado = True
        if job["status"] in TERMINALES:
            break
        if time.perf_counter() - inicio > timeout:
            job["status"] = "TIMEOUT_CLIENTE"
            break
        await asyncio.sleep(intervalo)

    return {
        "indice": indice,
        "job_id": job_id,
        "status": job["status"],
        "estados": estados_vistos,
        "publicar": publicar,
        "latencia_post_ms": round(latencia_post_ms, 1),
        "latencia_total_s": round(time.perf_counter() - inicio, 2),
        "ejecucion_worker_s": job.get("duracion_ejecucion_segundos"),
        "uso": job.get("uso"),
        "error": job.get("error"),
        "respuesta": ((job.get("resultado") or {}).get("respuesta_final") or "")[:300],
    }


async def main(args: argparse.Namespace) -> int:
    tareas = [TAREAS[i % len(TAREAS)] for i in range(args.n)]
    async with httpx.AsyncClient(base_url=args.url, timeout=30) as cliente:
        salud = (await cliente.get("/health")).json()
        print(f"API OK: {salud}\nLanzando {args.n} peticiones concurrentes...\n")
        inicio = time.perf_counter()
        resultados = await asyncio.gather(*(
            correr_job(cliente, i + 1, t, publicar=args.con_hitl and i == 0, intervalo=args.intervalo, timeout=args.timeout)
            for i, t in enumerate(tareas)
        ))
        total = time.perf_counter() - inicio

    posts = [r["latencia_post_ms"] for r in resultados]
    totales = [r["latencia_total_s"] for r in resultados if r["status"] == "DONE"]
    ejec = [r["ejecucion_worker_s"] for r in resultados if r["status"] == "DONE" and r["ejecucion_worker_s"]]
    tokens = [r["uso"]["tokens_totales"] for r in resultados if r["uso"]]
    costos = [r["uso"]["costo_estimado_usd"] for r in resultados if r["uso"]]

    print("\n" + "=" * 96)
    print(f"{'#':>2} {'status':<10} {'POST ms':>8} {'total s':>8} {'worker s':>9} {'tokens':>8} {'USD est.':>9}  job_id")
    for r in resultados:
        uso = r["uso"] or {}
        print(f"{r['indice']:>2} {r['status']:<10} {r['latencia_post_ms']:>8.1f} {r['latencia_total_s']:>8.1f} "
              f"{(r['ejecucion_worker_s'] or 0):>9.1f} {uso.get('tokens_totales', 0):>8} {uso.get('costo_estimado_usd', 0):>9.4f}  {r['job_id']}")
        if r["error"]:
            print(f"   error: {r['error']}")
    resumen = {
        "n": args.n,
        "ok": sum(r["status"] == "DONE" for r in resultados),
        "pared_total_s": round(total, 2),
        "post_ms_p50": round(percentil(posts, 50), 1),
        "post_ms_p95": round(percentil(posts, 95), 1),
        "total_s_p50": round(percentil(totales, 50), 2),
        "total_s_p95": round(percentil(totales, 95), 2),
        "worker_s_p50": round(percentil(ejec, 50), 2),
        "worker_s_p95": round(percentil(ejec, 95), 2),
        "tokens_promedio": round(statistics.mean(tokens)) if tokens else 0,
        "costo_estimado_promedio_usd": round(statistics.mean(costos), 5) if costos else 0,
        "costo_estimado_total_usd": round(sum(costos), 5),
    }
    print("-" * 96)
    for k, v in resumen.items():
        print(f"  {k:<30} {v}")

    salida = Path(__file__).resolve().parent.parent / "resultados"
    salida.mkdir(exist_ok=True)
    archivo = salida / f"carga_{datetime.now():%Y%m%d_%H%M%S}.json"
    archivo.write_text(json.dumps({"generado": datetime.now().isoformat(timespec="seconds"), "url": args.url,
                                   "resumen": resumen, "jobs": resultados}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultado guardado en {archivo}")
    return 0 if resumen["ok"] == args.n else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Prueba de carga de la API del orquestador.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=5, help="peticiones concurrentes (default 5)")
    parser.add_argument("--con-hitl", action="store_true", help="la primera tarea pide publicar y el script la aprueba")
    parser.add_argument("--intervalo", type=float, default=2.0, help="segundos entre polls")
    parser.add_argument("--timeout", type=float, default=900, help="segundos máximos por job")
    sys.exit(asyncio.run(main(parser.parse_args())))
