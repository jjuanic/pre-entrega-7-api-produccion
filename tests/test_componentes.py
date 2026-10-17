"""Tests unitarios: calculadora segura, validador, estado y aislamiento de parcelas."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

import config
from agents.analyst_agent import calculadora, evaluar_expresion
from agents.research_agent import buscar_documentacion
from agents.supervisor import aplicar_guardas
from agents.validator import validar_analisis, validar_investigacion
from app.graph import ViolacionDeParcela, con_parcela, enrutar
from state import (
    AnalysisArtifact, AnalysisOutput, Calculo, CalculoEjecutado, DecisionSupervisor, Evidencia,
    Hallazgo, ResearchArtifact, ResearchOutput, ValidationReport, estado_inicial, sumar_conteos,
)


# --- Calculadora segura ------------------------------------------------------

@pytest.mark.parametrize(("expresion", "esperado"), [
    ("240 * 0.5", 120.0),
    ("ceil(240 / 12) * 0.5", 10.0),
    ("min(32, 8 + 4)", 12.0),
    ("2 ^ 10", 1024.0),
    ("round(120 / 7, 2)", 17.14),
])
def test_calculadora_evalua_aritmetica(expresion: str, esperado: float) -> None:
    assert evaluar_expresion(expresion) == pytest.approx(esperado)


@pytest.mark.parametrize("maliciosa", [
    "__import__('os').system('dir')",
    "open('secreto.txt').read()",
    "(1).__class__",
    "9 ** 9 ** 9",
    "1 / 0",
])
def test_calculadora_rechaza_codigo_arbitrario(maliciosa: str) -> None:
    with pytest.raises(ValueError):
        evaluar_expresion(maliciosa)


def test_herramienta_calculadora_devuelve_artifact_y_no_lanza() -> None:
    mensaje = calculadora.invoke({"type": "tool_call", "id": "1", "name": "calculadora", "args": {"expresion": "1/0"}})
    assert "Error" in mensaje.content and mensaje.artifact["resultado"] is None


def test_buscar_documentacion_encuentra_el_dato_en_bm25_local() -> None:
    mensaje = buscar_documentacion.invoke(
        {"type": "tool_call", "id": "1", "name": "buscar_documentacion", "args": {"consulta": "max_workers ThreadPoolExecutor por defecto"}}
    )
    assert any(e["fuente"] == "concurrent_futures.md" for e in mensaje.artifact)
    assert "fuente: concurrent_futures.md" in mensaje.content


# --- Contratos Pydantic -------------------------------------------------

def test_confidence_score_fuera_de_rango_se_rechaza_en_el_borde() -> None:
    with pytest.raises(ValidationError):
        ResearchArtifact(hallazgos=[Hallazgo(dato="abc", fuente="x.md", cita="abc")], resumen="resumen largo", confidence_score=1.7)


def test_artefacto_sin_hallazgos_se_rechaza() -> None:
    with pytest.raises(ValidationError):
        ResearchArtifact(hallazgos=[], resumen="resumen largo", confidence_score=0.9)


# --- Validador determinístico -----------------------------------------------

def _investigacion(fuente: str, cita: str = "usa min(32, os.cpu_count() + 4) hilos", confianza: float = 0.9) -> ResearchOutput:
    return ResearchOutput(
        intento=1,
        instruccion="x",
        artefacto=ResearchArtifact(
            hallazgos=[Hallazgo(dato="Workers por defecto", fuente=fuente, cita=cita)],
            resumen="Resumen de prueba.",
            confidence_score=confianza,
        ),
        evidencia=[Evidencia(fuente="concurrent_futures.md", texto="ThreadPoolExecutor usa min(32, os.cpu_count() + 4) hilos")],
    )


def test_validador_acepta_investigacion_con_fuente_recuperada() -> None:
    assert validar_investigacion(_investigacion("concurrent_futures.md")).valido


def test_validador_detecta_fuente_inventada() -> None:
    reporte = validar_investigacion(_investigacion("asyncio_avanzado.pdf"))
    assert not reporte.valido and "asyncio_avanzado.pdf" in reporte.errores[0]


def test_validador_advierte_cita_parafraseada() -> None:
    reporte = validar_investigacion(_investigacion("concurrent_futures.md", cita="el pool escala según el procesador disponible"))
    assert reporte.valido and reporte.advertencias


def test_validador_rechaza_confianza_baja() -> None:
    reporte = validar_investigacion(_investigacion("concurrent_futures.md", confianza=0.2))
    assert not reporte.valido and "confidence_score" in reporte.errores[0]


def test_validador_recalcula_y_detecta_numero_alucinado() -> None:
    salida = AnalysisOutput(
        intento=1,
        instruccion="x",
        artefacto=AnalysisArtifact(
            calculos=[Calculo(descripcion="Secuencial", expresion="240 * 0.5", resultado=150)],
            conclusion="Conclusión de prueba.",
            confidence_score=0.9,
        ),
        calculos_ejecutados=[CalculoEjecutado(expresion="240 * 0.5", resultado=120)],
    )
    reporte = validar_analisis(salida)
    assert not reporte.valido
    assert reporte.errores == ["El cálculo 'Secuencial' declara resultado=150, pero la expresión '240 * 0.5' da 120."]


def test_validador_advierte_calculo_que_no_paso_por_la_calculadora() -> None:
    salida = AnalysisOutput(
        intento=1,
        instruccion="x",
        artefacto=AnalysisArtifact(
            calculos=[Calculo(descripcion="Secuencial", expresion="240*0.5", resultado=120)],
            conclusion="Conclusión de prueba.",
            confidence_score=0.9,
        ),
    )
    reporte = validar_analisis(salida)
    assert reporte.valido and "no se ejecutó con la calculadora" in reporte.advertencias[0]


# --- Estado, ruteo y guardas --------------------------------------------

def test_reducer_suma_intentos_por_agente() -> None:
    assert sumar_conteos({"investigador": 1}, {"investigador": 1, "analista": 1}) == {"investigador": 2, "analista": 1}


def test_enrutar_mapea_a_nombres_de_nodos() -> None:
    estado = estado_inicial("t")
    for destino in ("investigador", "analista", "sintesis"):
        estado["next_agent"] = destino  # type: ignore[typeddict-item]
        assert enrutar(estado) == destino
    estado["next_agent"] = None
    assert enrutar(estado) == "sintesis"


def test_guarda_max_intentos_bloquea_al_agente_agotado() -> None:
    estado = estado_inicial("t")
    estado["intentos"] = {"investigador": config.MAX_INTENTOS_POR_AGENTE}
    estado["validaciones"] = {"investigador": ValidationReport(agente="investigador", intento=3, valido=False, errores=["e"])}
    destino, _, _, regla, motivo = aplicar_guardas(DecisionSupervisor(next="investigador", razon="r"), estado)
    assert (destino, regla) == ("sintesis", "MAX_INTENTOS_POR_AGENTE")
    assert motivo and "investigador" in motivo


def test_parcela_impide_escribir_estado_ajeno() -> None:
    def analista_tramposo(state: dict) -> dict:
        return {"analysis_output": None, "research_output": None}  # pisa la parcela del investigador

    nodo = con_parcela("analista", analista_tramposo)  # type: ignore[arg-type]
    with pytest.raises(ViolacionDeParcela, match="research_output"):
        asyncio.run(nodo(estado_inicial("t")))


def test_validador_tolera_redondeo_pero_no_otro_numero() -> None:
    def con_resultado(valor: float) -> AnalysisOutput:
        return AnalysisOutput(
            intento=1,
            instruccion="x",
            artefacto=AnalysisArtifact(
                calculos=[Calculo(descripcion="Speedup", expresion="10 / 6", resultado=valor)],
                conclusion="Conclusión de prueba.",
                confidence_score=0.9,
            ),
            calculos_ejecutados=[CalculoEjecutado(expresion="10 / 6", resultado=10 / 6)],
        )

    assert validar_analisis(con_resultado(1.6667)).valido
    assert validar_analisis(con_resultado(1.67)).valido
    assert not validar_analisis(con_resultado(1.8)).valido
