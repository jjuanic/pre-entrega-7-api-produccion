"""
RAGSystem: recuperador híbrido (BM25 + Pinecone) encapsulado en una clase.

Flujo de una consulta:

    consulta ──┬─> BM25Retriever (léxico, en memoria)  ── top-10 ─┐
               └─> PineconeRetriever (semántico, nube) ── top-10 ─┤
                                                                   v
                         EnsembleRetriever: Reciprocal Rank Fusion ponderado
                                                                   v
                                                             top-5 final

¿Por qué RRF y no sumar scores? Un score BM25 de 12.3 y una similitud
coseno de 0.81 no están en la misma escala; sumarlos favorece
arbitrariamente a uno de los dos. RRF (lo que implementa EnsembleRetriever)
combina POSICIONES: score(d) = sum_i w_i / (60 + rank_i(d)), sin necesidad
de normalizar nada.

El corpus de BM25 se descarga del propio namespace de Pinecone (el texto
original vive en la metadata), así que ambos retrievers ven exactamente los
mismos chunks, con los mismos IDs.
"""

from __future__ import annotations

from typing import Literal

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag import config
from rag.retrievers import PineconeRetriever, build_bm25_retriever, matches_filter

Mode = Literal["hybrid", "bm25", "vector"]


class RAGSystem:
    """Recibe una consulta y devuelve los top-k chunks combinando
    resultados léxicos (BM25) y semánticos (Pinecone) con un EnsembleRetriever."""

    def __init__(
        self,
        corpus: list[dict],
        vector_retriever: BaseRetriever,
        k: int = config.TOP_K,
        weights: tuple[float, float] = (config.BM25_WEIGHT, config.VECTOR_WEIGHT),
        candidates_per_retriever: int = config.CANDIDATES_PER_RETRIEVER,
    ) -> None:
        """
        Args:
            corpus: registros (metadata de cada chunk, con `text` y `chunk_id`).
            vector_retriever: retriever semántico (PineconeRetriever en producción).
            k: cantidad de documentos que devuelve `retrieve`.
            weights: pesos [BM25, vectorial] para la fusión RRF.
            candidates_per_retriever: candidatos que aporta cada retriever a la fusión.
        """
        self.corpus = corpus
        self.k = k
        self.weights = list(weights)
        self.candidates = max(candidates_per_retriever, k)
        self.vector_retriever = vector_retriever
        self.bm25_retriever = build_bm25_retriever(corpus, k=self.candidates)
        self.ensemble = self._build_ensemble(self.bm25_retriever, self.vector_retriever)

    # --- Construcción -------------------------------------------------

    @classmethod
    def from_pinecone(cls, **kwargs) -> RAGSystem:
        """Arma el sistema completo contra el índice/namespace configurados."""
        from rag.embeddings import get_embeddings
        from rag.pinecone_store import get_index, iter_records

        index = get_index()
        corpus = [{**metadata, "chunk_id": metadata.get("chunk_id", vid)} for vid, metadata in iter_records(index)]
        if not corpus:
            raise RuntimeError(
                f"El namespace '{config.NAMESPACE}' del índice '{config.INDEX_NAME}' está vacío. "
                "Corré primero: python ingest.py"
            )
        candidates = kwargs.get("candidates_per_retriever", config.CANDIDATES_PER_RETRIEVER)
        vector_retriever = PineconeRetriever(
            index=index,
            embeddings=get_embeddings(),
            namespace=config.NAMESPACE,
            k=max(candidates, kwargs.get("k", config.TOP_K)),
        )
        return cls(corpus=corpus, vector_retriever=vector_retriever, **kwargs)

    def _build_ensemble(self, bm25: BaseRetriever, vector: BaseRetriever) -> EnsembleRetriever:
        # id_key="chunk_id": deduplica por ID del chunk (no por texto), así un
        # mismo chunk encontrado por ambos retrievers suma sus dos aportes RRF.
        return EnsembleRetriever(retrievers=[bm25, vector], weights=self.weights, id_key="chunk_id")

    # --- Consulta ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        mode: Mode = "hybrid",
        metadata_filter: dict | None = None,
    ) -> list[Document]:
        """Top-k documentos para `query`.

        Args:
            mode: "hybrid" (BM25 + Pinecone vía EnsembleRetriever), o cada
                retriever por separado ("bm25" / "vector") para comparar.
            metadata_filter: filtro estilo Pinecone (p. ej.
                `{"category": "concurrencia"}` o `{"tags": {"$in": ["cache"]}}`).
                Se aplica como hard filter en Pinecone y, en memoria, en BM25.
        """
        k = k or self.k
        bm25, vector = self.bm25_retriever, self.vector_retriever
        if metadata_filter:
            bm25, vector = self._filtered_retrievers(metadata_filter)
            if bm25 is None:
                return []

        if mode == "bm25":
            docs = bm25.invoke(query)
        elif mode == "vector":
            docs = vector.invoke(query)
        elif mode == "hybrid":
            ensemble = self.ensemble if not metadata_filter else self._build_ensemble(bm25, vector)
            docs = ensemble.invoke(query)
        else:
            raise ValueError(f"mode inválido: {mode!r}")

        # Copias: BM25 devuelve los mismos objetos que tiene indexados, y no
        # queremos que el rank de una consulta "contamine" la siguiente.
        top = [doc.model_copy(deep=True) for doc in docs[:k]]
        for rank, doc in enumerate(top, start=1):
            doc.metadata["rank"] = rank
            doc.metadata["retrieval_mode"] = mode
        return top

    def search(self, query: str, **kwargs) -> list[dict]:
        """Versión "amigable" de `retrieve` para mostrar resultados."""
        return [
            {
                "rank": d.metadata.get("rank"),
                "chunk_id": d.metadata.get("chunk_id"),
                "doc_id": d.metadata.get("doc_id"),
                "fuente": d.metadata.get("source"),
                "seccion": d.metadata.get("section"),
                "categoria": d.metadata.get("category"),
                "pagina": d.metadata.get("page"),
                "contenido": d.page_content,
            }
            for d in self.retrieve(query, **kwargs)
        ]

    def _filtered_retrievers(self, metadata_filter: dict):
        subset = [r for r in self.corpus if matches_filter(r, metadata_filter)]
        if not subset:
            return None, None
        bm25 = build_bm25_retriever(subset, k=self.candidates)
        if isinstance(self.vector_retriever, PineconeRetriever):
            vector = self.vector_retriever.model_copy(update={"metadata_filter": metadata_filter})
        else:
            vector = self.vector_retriever
        return bm25, vector
