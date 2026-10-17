"""
Punto de entrada del RAG para el Agente Investigador.

Dos backends con la misma interfaz (`Buscador.buscar`):

- "pinecone": el `RAGSystem` híbrido de la Pre-entrega 4 (BM25 + Pinecone
  combinados con RRF), conectado al índice ya cargado.
- "local": BM25 en memoria sobre los mismos documentos de `data/docs/`,
  chunkeados con el mismo pipeline. No necesita red ni API keys (se usa en
  los tests y como fallback si Pinecone no está disponible).

`RESEARCH_BACKEND=auto` (default) intenta Pinecone y, si falla la conexión o
faltan claves, cae a "local" avisando por log.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from rag.documents import chunk_documents, load_documents
from rag.rag_system import RAGSystem
from rag.retrievers import build_bm25_retriever

logger = logging.getLogger("orquestador")

Backend = Literal["pinecone", "local"]


@dataclass
class Buscador:
    rag: RAGSystem
    backend: Backend

    @property
    def modo(self) -> Literal["hybrid", "bm25"]:
        return "hybrid" if self.backend == "pinecone" else "bm25"

    def buscar(self, consulta: str, k: int = 4, categoria: str | None = None) -> list[dict[str, Any]]:
        filtro = {"category": {"$eq": categoria}} if categoria else None
        return self.rag.search(consulta, k=k, mode=self.modo, metadata_filter=filtro)

    def documentos(self) -> list[dict[str, str]]:
        """Catálogo de documentos indexados (uno por doc_id)."""
        vistos: dict[str, dict[str, str]] = {}
        for r in self.rag.corpus:
            vistos.setdefault(r["doc_id"], {
                "doc_id": r["doc_id"],
                "fuente": r.get("source", ""),
                "titulo": r.get("title", ""),
                "categoria": r.get("category", ""),
            })
        return sorted(vistos.values(), key=lambda d: d["doc_id"])


def _buscador_local() -> Buscador:
    corpus = [dict(c.metadata) for c in chunk_documents(load_documents())]
    bm25 = build_bm25_retriever(corpus)
    # En modo local el "vector_retriever" es el mismo BM25: solo se usa mode="bm25".
    return Buscador(rag=RAGSystem(corpus=corpus, vector_retriever=bm25), backend="local")


def _buscador_pinecone() -> Buscador:
    return Buscador(rag=RAGSystem.from_pinecone(), backend="pinecone")


@lru_cache(maxsize=1)
def obtener_buscador() -> Buscador:
    """Construye el buscador una sola vez por proceso (descargar el corpus de
    Pinecone para BM25 es costoso)."""
    preferido = os.getenv("RESEARCH_BACKEND", "auto").strip().lower()
    if preferido == "local":
        return _buscador_local()
    try:
        buscador = _buscador_pinecone()
        logger.info("RAG: backend Pinecone híbrido (%d chunks).", len(buscador.rag.corpus))
        return buscador
    except Exception as exc:
        if preferido == "pinecone":
            raise
        logger.warning("RAG: Pinecone no disponible (%s). Se usa BM25 local sobre data/docs.", exc)
        return _buscador_local()
