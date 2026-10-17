"""Tests del grafo completo con un LLM guionado (sin API keys ni red)."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage

import config
from agents.common import FALLAS
from app.graph import compilar
from tracing import ejecutar_tarea
from conftest import LLMGuionado, analisis, decidir, guion, investigacion_ok

TAREA = "¿Cuántos hilos usa ThreadPoolExecutor por defecto en una máquina de 8 núcleos?"
SINTESIS = AIMessage(content="Usa 12 hilos (concurrent_futures.md).")


def correr(llm: LLMGuionado, tarea: str = TAREA) -> dict[str, Any]:
    return asyncio.run(ejecutar_tarea(compilar(llm), tarea))


def test_flujo_de_delegacion_investigador_analista_fin() -> None:
    llm = guion(
        decidir("investigador", instruccion="Buscá el max_workers por defecto de ThreadPoolExecutor."),
        *investigacion_ok(),
        decidir("analista", instruccion="Calculá min(32, 8 + 4)."),
        *analisis(),
        decidir("FINISH", razon="Investigación y análisis válidos."),
        SINTESIS,
    )
    traza = correr(llm)

    assert traza["recorrido"] == [
        "supervisor", "investigador", "validador",
        "supervisor", "analista", "validador",
        "supervisor", "sintesis", "evaluar_riesgo",  # cola HITL (PE7): no crítica -> END
    ]
    assert traza["task_completed"] and not traza["requiere_revision_humana"]
    assert traza["validaciones"]["investigador"]["valido"]
    assert traza["validaciones"]["analista"]["valido"]
    assert [d["destino"] for d in traza["decisiones"]] == ["investigador", "analista", "sintesis"]
    assert all(d["regla_aplicada"] is None for d in traza["decisiones"])
    # Provenance: la evidencia la capturó el nodo desde la herramienta, no el LLM.
    fuentes = {e["fuente"] for e in traza["research_output"]["evidencia"]}
    assert "concurrent_futures.md" in fuentes
    assert traza["analysis_output"]["calculos_ejecutados"][0]["resultado"] == 12.0
    # Trazabilidad: quién aportó qué, en orden.
    assert [c["agente"] for c in traza["contribuciones"]] == ["investigador", "validador", "analista", "validador", "sintesis"]
    assert traza["respuesta_final"].startswith("Usa 12 hilos")
    assert "`concurrent_futures.md`" in traza["respuesta_final"]


def test_especialista_recibe_contexto_acotado() -> None:
    llm = guion(
        decidir("investigador", instruccion="INSTRUCCION-INV"),
        *investigacion_ok(),
        decidir("analista", instruccion="INSTRUCCION-ANA"),
        *analisis(),
        decidir("FINISH"),
        SINTESIS,
    )
    correr(llm)
    # Primera llamada del analista: system prompt + UN mensaje humano armado por el nodo.
    primera_del_analista = next(m for m in llm.recibidos if "INSTRUCCION-ANA" in str(m[-1].content))
    contenido = str(primera_del_analista[-1].content)
    assert len(primera_del_analista) == 2
    assert "Datos verificados por el Investigador" in contenido
    assert "concurrent_futures.md" in contenido
    assert TAREA not in contenido  # no recibe la conversación del usuario
    assert "fuente: concurrent_futures.md | sección" not in contenido  # ni los fragmentos crudos del RAG


def test_calculo_alucinado_se_detecta_y_se_refina() -> None:
    llm = guion(
        decidir("investigador", instruccion="Buscá max_workers."),
        *investigacion_ok(),
        decidir("analista", instruccion="Calculá min(32, 8 + 4)."),
        *analisis(resultado_declarado=25.0),  # declara 25 pero la expresión da 12
        decidir("analista", razon="El validador marcó el cálculo como inválido.", instruccion="Corregí el cálculo."),
        *analisis(),
        decidir("FINISH"),
        SINTESIS,
    )
    traza = correr(llm)

    validaciones_analista = [c for c in traza["contribuciones"] if c["agente"] == "validador" and "analista" in c["detalle"]]
    assert "INVÁLIDO" in validaciones_analista[0]["detalle"]
    assert "declara resultado=25" in validaciones_analista[0]["detalle"] and "da 12" in validaciones_analista[0]["detalle"]
    assert "VÁLIDO" in validaciones_analista[1]["detalle"]
    assert traza["intentos"]["analista"] == 2
    # El reintento recibió el error ESPECÍFICO del validador (bucle de reparación).
    reintento = next(m for m in llm.recibidos if "NO pasó la validación" in str(m[-1].content))
    assert "declara resultado=25" in str(reintento[-1].content)
    assert traza["task_completed"]


def test_guarda_de_suficiencia_impide_cerrar_con_output_invalido() -> None:
    llm = guion(
        decidir("investigador", instruccion="Buscá max_workers."),
        *investigacion_ok(),
        decidir("analista", instruccion="Calculá."),
        *analisis(resultado_declarado=99.0),
        decidir("FINISH", razon="Parece completo."),  # el LLM quiere cerrar igual
        *analisis(),  # la guarda lo manda de vuelta al analista
        decidir("FINISH"),
        SINTESIS,
    )
    traza = correr(llm)

    tercera = traza["decisiones"][2]
    assert tercera["propuesta_llm"] == "FINISH"
    assert tercera["destino"] == "analista"
    assert tercera["regla_aplicada"] == "SUFICIENCIA"
    assert traza["validaciones"]["analista"]["valido"]


def test_falla_de_un_especialista_no_rompe_al_supervisor() -> None:
    FALLAS.programar("investigador", "excepcion")
    llm = guion(
        decidir("investigador", instruccion="Buscá max_workers."),
        # La falla ocurre antes de llamar al LLM del investigador.
        decidir("investigador", razon="La investigación falló; reintento.", instruccion="Buscá max_workers."),
        *investigacion_ok(),
        decidir("analista", instruccion="Calculá."),
        *analisis(),
        decidir("FINISH"),
        SINTESIS,
    )
    traza = correr(llm)

    primera = traza["contribuciones"][0]
    assert primera["agente"] == "investigador" and "ConnectionError" in primera["detalle"]
    assert traza["intentos"]["investigador"] == 2
    assert traza["task_completed"]


def test_fuente_inventada_agota_intentos_y_escala_a_revision_humana() -> None:
    inventada = lambda: investigacion_ok(fuente="manual_inventado.md")  # noqa: E731
    llm = guion(
        decidir("investigador", instruccion="Buscá max_workers."),
        *inventada(),
        decidir("investigador", instruccion="Corregí la fuente."),
        *inventada(),
        decidir("investigador", instruccion="Corregí la fuente."),
        *inventada(),
        decidir("investigador", instruccion="Otra vez."),  # 4.º pedido: la guarda lo bloquea
        SINTESIS,
    )
    traza = correr(llm)

    assert traza["intentos"]["investigador"] == config.MAX_INTENTOS_POR_AGENTE
    assert "manual_inventado.md" in traza["validaciones"]["investigador"]["errores"][0]
    ultima = traza["decisiones"][-1]
    assert ultima["regla_aplicada"] == "MAX_INTENTOS_POR_AGENTE" and ultima["destino"] == "sintesis"
    assert traza["requiere_revision_humana"] and not traza["task_completed"]
    assert "Requiere revisión humana" in traza["respuesta_final"]


def test_supervisor_usa_fallback_si_su_llm_falla() -> None:
    class LLMRoto(LLMGuionado):
        def with_structured_output(self, *a: Any, **k: Any) -> Any:  # type: ignore[override]

            class Router:
                async def ainvoke(self, *_: Any, **__: Any) -> Any:
                    raise ValueError("respuesta no parseable")

            return Router()

    mensajes = iter([*investigacion_ok(), *analisis(), SINTESIS])
    llm = LLMRoto(messages=mensajes, recibidos=[])
    traza = correr(llm)

    assert [d["destino"] for d in traza["decisiones"]] == ["investigador", "analista", "sintesis"]
    assert all(d["regla_aplicada"] == "FALLBACK" for d in traza["decisiones"])
    assert traza["task_completed"]
