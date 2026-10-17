"""
Punto único de construcción del modelo de embeddings.

ingest.py (al indexar) y el retriever de Pinecone (al consultar) obtienen el
modelo desde acá, así se garantiza que ambos usen exactamente el mismo
modelo y la misma dimensión (`config.EMBEDDING_DIM`). Mezclar modelos o
dimensiones hace que la distancia vectorial deje de tener sentido (o que
Pinecone rechace el upsert).
"""

from __future__ import annotations

import time

from langchain_core.embeddings import Embeddings

from rag import config

# Los tiers gratuitos tienen pocas requests por minuto: lotes chicos + pausa
# entre lotes + reintento con backoff evitan el 429 al indexar todo de una.
_EMBED_BATCH_SIZE = 20
_EMBED_BATCH_DELAY_SECONDS = 1.0
_EMBED_MAX_RETRIES = 5
_EMBED_RETRY_BASE_DELAY_SECONDS = 15.0
_RATE_LIMIT_MARKERS = ("RESOURCE_EXHAUSTED", "429", "rate limit", "Rate limit")


def get_embeddings() -> Embeddings:
    """Modelo de embeddings de LangChain según `EMBEDDING_PROVIDER`."""
    api_key = config.require_embedding_key()
    if config.EMBEDDING_PROVIDER == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        # gemini-embedding-001 genera 3072 dims por defecto; con
        # output_dimensionality lo llevamos a 1536 (Matryoshka) para que
        # coincida con el índice. Para documentos usa la tarea
        # RETRIEVAL_DOCUMENT y para consultas RETRIEVAL_QUERY (automático).
        return GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=api_key,
            output_dimensionality=config.EMBEDDING_DIM,
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=api_key,
        dimensions=config.EMBEDDING_DIM,
    )


def embed_documents_batched(embeddings: Embeddings, texts: list[str]) -> list[list[float]]:
    """Embebe `texts` en lotes, con reintentos ante rate limit, y valida la
    dimensión de cada vector ANTES de mandarlo a Pinecone."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        if i > 0:
            time.sleep(_EMBED_BATCH_DELAY_SECONDS)
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        vectors.extend(_embed_batch_with_retry(embeddings, batch))
    for v in vectors:
        check_dimension(v)
    return vectors


def check_dimension(vector: list[float]) -> None:
    """Guarda contra el 'Mismatch de Dimensiones'."""
    if len(vector) != config.EMBEDDING_DIM:
        raise ValueError(
            f"El modelo devolvió vectores de {len(vector)} dimensiones pero el "
            f"índice espera {config.EMBEDDING_DIM}. Revisá EMBEDDING_MODEL / EMBEDDING_DIM."
        )


def _embed_batch_with_retry(embeddings: Embeddings, batch: list[str]) -> list[list[float]]:
    delay = _EMBED_RETRY_BASE_DELAY_SECONDS
    for attempt in range(_EMBED_MAX_RETRIES):
        try:
            return embeddings.embed_documents(batch)
        except Exception as exc:
            is_last_attempt = attempt == _EMBED_MAX_RETRIES - 1
            if is_last_attempt or not any(m in str(exc) for m in _RATE_LIMIT_MARKERS):
                raise
            print(f"  Rate limit alcanzado, esperando {delay:.0f}s antes de reintentar...")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")  # pragma: no cover
