"""
Capturas del dashboard de Phoenix para la carpeta `screenshots/` (herramienta de desarrollo).

    pip install playwright            # usa el Edge/Chrome instalado: no descarga navegador
    python scripts/capturas_phoenix.py

Toma la ventana de tiempo de la última prueba de carga (`resultados/carga_*.json`)
y captura: proyectos, trazas de la corrida, detalle de una traza, latencia p95,
costo, la sesión del job con Human-in-the-loop y la latencia de la API.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

import httpx
from playwright.sync_api import Page, sync_playwright

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "screenshots"
PHOENIX = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")
PROYECTO = os.getenv("PHOENIX_PROJECT_NAME", "orquestador-agentes")


def ultima_carga() -> dict:
    archivo = sorted((RAIZ / "resultados").glob("carga_*.json"))[-1]
    return json.loads(archivo.read_text(encoding="utf-8"))


def ultimas_metricas() -> dict:
    archivo = sorted((RAIZ / "resultados").glob("metricas_phoenix_*.json"))[-1]
    return json.loads(archivo.read_text(encoding="utf-8"))


def proyectos() -> dict[str, str]:
    r = httpx.post(f"{PHOENIX}/graphql", json={"query": "{ projects { edges { node { id name } } } }"}).json()
    return {e["node"]["name"]: e["node"]["id"] for e in r["data"]["projects"]["edges"]}


def esperar(page: Page, ms: int = 2500) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def captura(page: Page, nombre: str, selector: str | None = None, pagina_completa: bool = False) -> None:
    destino = SALIDA / nombre
    if selector:
        page.locator(selector).first.screenshot(path=destino)
    else:
        page.screenshot(path=destino, full_page=pagina_completa)
    print(f"  {destino.relative_to(RAIZ)}")


def tarjeta(page: Page, titulo: str):
    """La tarjeta del tab Metrics cuyo título es `titulo` (el contenedor más chico que tiene el título y un gráfico)."""
    return page.locator(f"xpath=//*[normalize-space(text())='{titulo}']/ancestor::*[.//*[local-name()='svg' or local-name()='canvas']][1]").first


def mostrar_tooltip(page: Page, contenedor, selector: str) -> None:
    """Mueve el mouse sobre el elemento con datos más alto del gráfico (punto o barra)."""
    elementos = [e for e in contenedor.locator(selector).all() if e.bounding_box()]
    if not elementos:
        return
    mas_alto = min(elementos, key=lambda e: e.bounding_box()["y"])
    caja = mas_alto.bounding_box()
    page.mouse.move(caja["x"] + caja["width"] / 2, caja["y"] + max(caja["height"] / 2, 2))
    page.wait_for_timeout(900)


def componer_costos(page: Page, recortes: list, destino: Path) -> None:
    """Junta los tooltips de costo de cada ejecución en una sola imagen."""
    import base64

    celdas = "".join(
        f'<figure><img src="data:image/png;base64,{base64.b64encode(ruta.read_bytes()).decode()}">'
        f'<figcaption>Ejecución {i} · trace {c["trace_id"][:12]}… · {c["latencia_s"]} s · Phoenix: USD {c["costo_usd"]:.4f}</figcaption></figure>'
        for i, (ruta, c) in enumerate(recortes, 1)
    )
    page.set_viewport_size({"width": 1600, "height": 900})
    page.set_content(
        "<html><body style='background:#111;color:#ddd;font:15px system-ui;margin:16px'>"
        "<h2 style='margin:0 0 12px'>Costo por ejecución (Phoenix · tooltip de Total Cost por traza) — prueba de 5 peticiones concurrentes</h2>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px'>{celdas}</div></body></html>"
    )
    page.wait_for_timeout(500)
    page.screenshot(path=destino, full_page=True)
    for ruta, _ in recortes:
        ruta.unlink()
    print(f"  {destino.relative_to(RAIZ)}")


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    carga, metricas = ultima_carga(), ultimas_metricas()
    fin = datetime.fromisoformat(carga["generado"]).astimezone(timezone.utc) + timedelta(seconds=40)
    inicio = fin - timedelta(seconds=carga["resumen"]["pared_total_s"] + 100)
    rango = urlencode({
        "timeRangeStart": inicio.strftime("%Y-%m-%dT%H:%M:00.000Z"),
        "timeRangeEnd": fin.strftime("%Y-%m-%dT%H:%M:59.000Z"),
    })
    ids = proyectos()
    pid, pid_api = ids[PROYECTO], ids.get(f"{PROYECTO}-api")
    job_hitl = next((j["job_id"] for j in carga["jobs"] if j["publicar"]), None)
    corridas = [e for e in metricas["ejecuciones"] if e["segmento"] == "run" and e["status"] == "OK"]
    traza_lenta = max(corridas, key=lambda e: e["latencia_s"])["trace_id"]

    with sync_playwright() as p:
        navegador = p.chromium.launch(channel=os.getenv("NAVEGADOR", "msedge"))
        page = navegador.new_page(viewport={"width": 1680, "height": 1000}, device_scale_factor=1)

        print("Capturando Phoenix...")
        page.goto(f"{PHOENIX}/projects?timeRangeKey=1h")
        esperar(page)
        captura(page, "01_proyectos_phoenix.png")

        # Tabla de trazas en un viewport ancho para que entren latencia, tokens y costo.
        page.set_viewport_size({"width": 3000, "height": 1000})
        page.goto(f"{PHOENIX}/projects/{quote(pid)}/traces?{rango}")
        esperar(page, 3500)
        captura(page, "02_trazas_prueba_carga.png")

        # Detalle de la traza más lenta: árbol job -> grafo -> nodos -> LLM / tools.
        page.set_viewport_size({"width": 1680, "height": 1000})
        page.goto(f"{PHOENIX}/projects/{quote(pid)}/traces?{rango}")
        esperar(page, 3000)
        page.locator(f'a[href*="/traces/{traza_lenta}"]').first.click()
        esperar(page, 4000)
        captura(page, "03_detalle_traza_job.png")

        page.goto(f"{PHOENIX}/projects/{quote(pid)}/metrics?{rango}")
        esperar(page, 4000)
        captura(page, "04_metricas_dashboard.png", pagina_completa=True)
        latencia = tarjeta(page, "Trace latency")
        latencia.scroll_into_view_if_needed()
        # Pasar el mouse por el punto con datos para que aparezca el tooltip con P50…P99.
        mostrar_tooltip(page, latencia, "circle, .recharts-dot")
        caja = latencia.bounding_box()
        # Recorte un poco más alto que la tarjeta: el tooltip (p50…max) se sale por abajo.
        page.screenshot(path=SALIDA / "05_latencia_p95.png",
                        clip={"x": caja["x"], "y": caja["y"], "width": caja["width"], "height": caja["height"] + 90})
        print("  screenshots/05_latencia_p95.png")
        costo = tarjeta(page, "Cost")
        costo.scroll_into_view_if_needed()
        mostrar_tooltip(page, costo, ".recharts-bar-rectangle path, .recharts-rectangle")
        costo.screenshot(path=SALIDA / "06_costo_estimado.png")
        print("  screenshots/06_costo_estimado.png")
        modelos = tarjeta(page, "Top models by cost")
        modelos.scroll_into_view_if_needed()
        modelos.screenshot(path=SALIDA / "07_costo_por_modelo.png")
        print("  screenshots/07_costo_por_modelo.png")

        page.goto(f"{PHOENIX}/projects/{quote(pid)}/sessions?{rango}")
        esperar(page, 3000)
        captura(page, "08_sesiones_por_job.png")
        if job_hitl:
            page.get_by_text(job_hitl, exact=True).first.click()
            esperar(page, 3500)
            captura(page, "09_sesion_hitl_run_y_resume.png")

        # Costo de CADA ejecución: tooltip de "Total Cost" en el detalle de cada traza.
        recortes = []
        for i, corrida in enumerate(sorted(corridas, key=lambda e: e["latencia_s"]), 1):
            page.goto(f"{PHOENIX}/projects/{quote(pid)}/traces?{rango}")
            esperar(page, 2500)
            page.locator(f'a[href*="/traces/{corrida["trace_id"]}"]').first.click()
            page.wait_for_timeout(3000)
            total = page.get_by_text("Total Cost", exact=True).first
            caja = total.bounding_box()
            page.mouse.move(caja["x"] + 20, caja["y"] + 30)
            page.wait_for_timeout(1200)
            destino = SALIDA / f"_costo_{i}.png"
            page.screenshot(path=destino, clip={"x": max(caja["x"] - 300, 0), "y": caja["y"] - 15, "width": 760, "height": 290})
            recortes.append((destino, corrida))
        componer_costos(page, recortes, SALIDA / "11_costo_por_ejecucion.png")

        if pid_api:
            # Latencia de la API: el POST /tasks solo encola, responde en milisegundos.
            page.set_viewport_size({"width": 1680, "height": 1000})
            page.goto(f"{PHOENIX}/projects/{quote(pid_api)}/spans?{rango}")
            esperar(page, 3000)
            filtro = page.get_by_role("textbox", name=re.compile("filter", re.I)).first
            filtro.click()
            page.keyboard.press("Control+A")
            page.keyboard.type("name == 'POST /tasks'")
            page.keyboard.press("Enter")
            esperar(page, 3000)
            captura(page, "10_api_post_tasks_latencia.png")

        navegador.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
