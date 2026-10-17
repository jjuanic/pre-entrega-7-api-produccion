"""
Lee de Phoenix (API GraphQL) las métricas de una corrida de carga.

    python scripts/metricas_phoenix.py                       # ventana de la última prueba de carga
    python scripts/metricas_phoenix.py --minutos 30          # últimos 30 minutos
    python scripts/metricas_phoenix.py --desde 2026-10-18T10:00:00 --hasta 2026-10-18T10:15:00

Reporta lo mismo que se ve en el dashboard, calculado por Phoenix:
- trazas, latencia p50 / p95 / p99 por traza;
- costo total y COSTO POR EJECUCIÓN (Phoenix lo calcula con los tokens de
  entrada y salida y su tabla de precios por modelo);
- desglose por nodo del grafo: tokens acumulados, costo y latencia, para ver
  dónde se concentra la latencia y qué nodo consume más tokens.

El resultado queda en `resultados/metricas_phoenix_<fecha>.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

RAIZ = Path(__file__).resolve().parent.parent
NODOS = ("supervisor", "investigador", "analista", "validador", "sintesis", "evaluar_riesgo", "aprobacion_humana", "publicar")

Q_PROYECTO = """
query($rango: TimeRange!) {
  projects { edges { node {
    id name
    traceCount(timeRange: $rango)
    p50: latencyMsQuantile(probability: 0.5, timeRange: $rango)
    p95: latencyMsQuantile(probability: 0.95, timeRange: $rango)
    p99: latencyMsQuantile(probability: 0.99, timeRange: $rango)
    tokenCountTotal(timeRange: $rango)
    tokenCountPrompt(timeRange: $rango)
    tokenCountCompletion(timeRange: $rango)
    costSummary(timeRange: $rango) { total { cost tokens } prompt { cost tokens } completion { cost tokens } }
  } } }
}"""

Q_TRAZAS = """
query($rango: TimeRange!, $after: String) {
  projects { edges { node { name
    spans(first: 200, after: $after, timeRange: $rango, filterCondition: "name == 'job'") {
      pageInfo { hasNextPage endCursor }
      edges { node {
        statusCode
        attributes
        trace {
          traceId latencyMs
          costSummary { total { cost tokens } }
          spans(first: 1000) { edges { node {
            name spanKind latencyMs parentId context { spanId }
            cumulativeTokenCountTotal
            cumulativeCostDetailSummaryEntries { value { cost } }
          } } }
        }
      } }
    }
  } } }
}"""


def gql(url: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    r = httpx.post(f"{url}/graphql", json={"query": query, "variables": variables}, timeout=60)
    r.raise_for_status()
    datos = r.json()
    if datos.get("errors"):
        raise RuntimeError(datos["errors"])
    return datos["data"]


def percentil(valores: list[float], p: int) -> float:
    if not valores:
        return 0.0
    if len(valores) == 1:
        return valores[0]
    return statistics.quantiles(valores, n=100, method="inclusive")[p - 1]


def ventana_ultima_carga() -> tuple[datetime, datetime, str]:
    cargas = sorted((RAIZ / "resultados").glob("carga_*.json"))
    if not cargas:
        raise SystemExit("No hay resultados/carga_*.json: pasá --minutos o --desde/--hasta.")
    datos = json.loads(cargas[-1].read_text(encoding="utf-8"))
    fin = datetime.fromisoformat(datos["generado"]).astimezone(timezone.utc)
    inicio = fin - timedelta(seconds=datos["resumen"]["pared_total_s"] + 30)
    return inicio, fin + timedelta(seconds=30), cargas[-1].name


def main(args: argparse.Namespace) -> None:
    if args.desde:
        inicio = datetime.fromisoformat(args.desde).astimezone(timezone.utc)
        fin = datetime.fromisoformat(args.hasta).astimezone(timezone.utc) if args.hasta else datetime.now(timezone.utc)
        origen = "argumentos"
    elif args.minutos:
        fin = datetime.now(timezone.utc)
        inicio = fin - timedelta(minutes=args.minutos)
        origen = f"últimos {args.minutos} min"
    else:
        inicio, fin, origen = ventana_ultima_carga()
    rango = {"start": inicio.isoformat(), "end": fin.isoformat()}

    proyectos = [e["node"] for e in gql(args.url, Q_PROYECTO, {"rango": rango})["projects"]["edges"]]
    proyecto = next((p for p in proyectos if p["name"] == args.proyecto), None)
    if proyecto is None:
        raise SystemExit(f"No existe el proyecto '{args.proyecto}' en Phoenix ({[p['name'] for p in proyectos]}).")

    # Trazas de jobs en la ventana (paginado) y desglose por nodo.
    trazas: list[dict[str, Any]] = []
    cursor = None
    while True:
        datos = gql(args.url, Q_TRAZAS, {"rango": rango, "after": cursor})
        nodo_p = next(e["node"] for e in datos["projects"]["edges"] if e["node"]["name"] == args.proyecto)
        trazas += [e["node"] for e in nodo_p["spans"]["edges"]]
        if not nodo_p["spans"]["pageInfo"]["hasNextPage"]:
            break
        cursor = nodo_p["spans"]["pageInfo"]["endCursor"]

    por_nodo: dict[str, dict[str, float]] = defaultdict(lambda: {"invocaciones": 0, "tokens": 0, "costo_usd": 0.0, "latencia_ms": 0.0})
    ejecuciones = []
    for raiz in trazas:
        traza = raiz["trace"]
        spans = [e["node"] for e in traza["spans"]["edges"]]
        grafo = next((s for s in spans if s["name"].startswith("orquestador:")), None)
        segmento = grafo["name"].split(":", 1)[1] if grafo else "?"
        for s in spans:
            if grafo and s["parentId"] == grafo["context"]["spanId"] and s["name"] in NODOS:
                d = por_nodo[s["name"]]
                d["invocaciones"] += 1
                d["tokens"] += s["cumulativeTokenCountTotal"] or 0
                d["costo_usd"] += sum((x["value"]["cost"] or 0) for x in s["cumulativeCostDetailSummaryEntries"] or [])
                d["latencia_ms"] += s["latencyMs"] or 0
        costo = (traza["costSummary"]["total"] or {}).get("cost") or 0.0
        ejecuciones.append({
            "trace_id": traza["traceId"], "segmento": segmento, "status": raiz["statusCode"],
            "latencia_s": round(traza["latencyMs"] / 1000, 2), "costo_usd": round(costo, 6),
            "tokens": (traza["costSummary"]["total"] or {}).get("tokens") or 0,
        })

    corridas = [e for e in ejecuciones if e["segmento"] == "run" and e["status"] == "OK"]
    costos = [e["costo_usd"] for e in corridas]
    latencias = [e["latencia_s"] for e in corridas]
    total_tokens_nodos = sum(d["tokens"] for d in por_nodo.values()) or 1
    total_lat_nodos = sum(d["latencia_ms"] for d in por_nodo.values()) or 1

    resumen = {
        "proyecto": args.proyecto,
        "ventana_utc": rango,
        "origen_ventana": origen,
        "phoenix": {
            "trazas": proyecto["traceCount"],
            "latencia_p50_s": round((proyecto["p50"] or 0) / 1000, 2),
            "latencia_p95_s": round((proyecto["p95"] or 0) / 1000, 2),
            "latencia_p99_s": round((proyecto["p99"] or 0) / 1000, 2),
            "tokens_total": proyecto["tokenCountTotal"],
            "tokens_entrada": proyecto["tokenCountPrompt"],
            "tokens_salida": proyecto["tokenCountCompletion"],
            "costo_total_usd": round(proyecto["costSummary"]["total"]["cost"] or 0, 6),
            "costo_entrada_usd": round(proyecto["costSummary"]["prompt"]["cost"] or 0, 6),
            "costo_salida_usd": round(proyecto["costSummary"]["completion"]["cost"] or 0, 6),
        },
        "por_ejecucion": {
            "ejecuciones_ok": len(corridas),
            "costo_promedio_usd": round(statistics.mean(costos), 6) if costos else 0,
            "costo_min_usd": round(min(costos), 6) if costos else 0,
            "costo_max_usd": round(max(costos), 6) if costos else 0,
            "latencia_p50_s": round(percentil(latencias, 50), 2),
            "latencia_p95_s": round(percentil(latencias, 95), 2),
        },
        "por_nodo": {
            nombre: {
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in d.items()},
                "pct_tokens": round(100 * d["tokens"] / total_tokens_nodos, 1),
                "pct_latencia": round(100 * d["latencia_ms"] / total_lat_nodos, 1),
            }
            for nombre, d in sorted(por_nodo.items(), key=lambda kv: -kv[1]["tokens"])
        },
        "ejecuciones": ejecuciones,
    }

    ph, pe = resumen["phoenix"], resumen["por_ejecucion"]
    print(f"Proyecto Phoenix: {args.proyecto}  |  ventana {inicio:%H:%M:%S}–{fin:%H:%M:%S} UTC ({origen})")
    print(f"  Trazas: {ph['trazas']}   Latencia por traza  p50={ph['latencia_p50_s']} s  p95={ph['latencia_p95_s']} s  p99={ph['latencia_p99_s']} s")
    print(f"  Tokens: {ph['tokens_total']} (entrada {ph['tokens_entrada']}, salida {ph['tokens_salida']})")
    print(f"  Costo total: USD {ph['costo_total_usd']:.4f}  (entrada {ph['costo_entrada_usd']:.4f} + salida {ph['costo_salida_usd']:.4f})")
    print(f"  Costo por ejecución ({pe['ejecuciones_ok']} corridas OK): promedio USD {pe['costo_promedio_usd']:.4f}  "
          f"[min {pe['costo_min_usd']:.4f} – max {pe['costo_max_usd']:.4f}]")
    print("\n  Nodo               invoc.   tokens  %tok   USD      latencia s  %lat")
    for nombre, d in resumen["por_nodo"].items():
        print(f"  {nombre:<18} {d['invocaciones']:>5} {d['tokens']:>8} {d['pct_tokens']:>5}  {d['costo_usd']:.4f}  "
              f"{d['latencia_ms'] / 1000:>9.1f}  {d['pct_latencia']:>5}")

    salida = RAIZ / "resultados" / f"metricas_phoenix_{datetime.now():%Y%m%d_%H%M%S}.json"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado en {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Métricas de la corrida de carga leídas de Phoenix.")
    parser.add_argument("--url", default=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"))
    parser.add_argument("--proyecto", default=os.getenv("PHOENIX_PROJECT_NAME", "orquestador-agentes"))
    parser.add_argument("--minutos", type=int)
    parser.add_argument("--desde")
    parser.add_argument("--hasta")
    main(parser.parse_args())
