"""
Acceso a Pinecone con el SDK nativo: cliente, creación idempotente del
índice Serverless y lectura del corpus guardado en la metadata.

Se usa el SDK nativo (permitido por la consigna) en lugar de
`langchain-pinecone` porque ese paquete todavía no publica versiones para
Python 3.14. La integración con LangChain se hace igual, con un retriever
propio (ver app/retrievers.py).
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from pinecone import Pinecone, ServerlessSpec

from rag import config

_client: Pinecone | None = None


def get_client() -> Pinecone:
    global _client
    if _client is None:
        _client = Pinecone(api_key=config.require_pinecone_key())
    return _client


def ensure_index() -> bool:
    """Verifica si el índice existe y, si no, lo crea en modo Serverless.

    Si ya existe, valida que su dimensión y métrica coincidan con la
    configuración: es preferible fallar acá con un mensaje claro que recibir
    un error críptico a mitad de la ingesta.

    Devuelve True si el índice se creó en esta llamada.
    """
    pc = get_client()
    if pc.has_index(config.INDEX_NAME):
        description = pc.describe_index(config.INDEX_NAME)
        if description.dimension != config.EMBEDDING_DIM:
            raise RuntimeError(
                f"Mismatch de dimensiones: el índice '{config.INDEX_NAME}' existe con "
                f"dimensión {description.dimension}, pero EMBEDDING_DIM={config.EMBEDDING_DIM}. "
                "Usá otro INDEX_NAME o borrá el índice desde la consola de Pinecone."
            )
        if description.metric != config.PINECONE_METRIC:
            raise RuntimeError(
                f"El índice '{config.INDEX_NAME}' usa la métrica '{description.metric}', "
                f"se esperaba '{config.PINECONE_METRIC}'."
            )
        return False

    pc.create_index(
        name=config.INDEX_NAME,
        dimension=config.EMBEDDING_DIM,
        metric=config.PINECONE_METRIC,
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        tags={"project": "pre-entrega-6", "embedding-model": _tag_safe(config.EMBEDDING_MODEL)},
    )
    _wait_until_ready()
    return True


def get_index():
    """Handle del índice (data plane). Requiere que exista."""
    pc = get_client()
    if not pc.has_index(config.INDEX_NAME):
        raise RuntimeError(
            f"El índice '{config.INDEX_NAME}' no existe. Corré primero: python setup_index.py"
        )
    return pc.Index(config.INDEX_NAME)


def namespace_vector_count(index, namespace: str = config.NAMESPACE) -> int:
    stats = index.describe_index_stats()
    summary = stats.namespaces.get(namespace)
    return summary.vector_count if summary else 0


def list_ids(index, prefix: str | None = None, namespace: str = config.NAMESPACE) -> list[str]:
    """IDs del namespace (opcionalmente filtrados por prefijo), paginando."""
    ids: list[str] = []
    for page in index.list(prefix=prefix, namespace=namespace):
        ids.extend(_page_ids(page))
    return ids


def iter_records(index, namespace: str = config.NAMESPACE, page_size: int = 100) -> Iterator[tuple[str, dict]]:
    """Recorre todos los vectores del namespace devolviendo (id, metadata).

    Como el texto original vive en la metadata (`text`), esto alcanza para
    reconstruir el corpus completo sin una base relacional aparte: es lo que
    usa el retriever BM25 para indexar exactamente lo mismo que está en
    Pinecone.
    """
    for page in index.list(namespace=namespace, limit=page_size):
        ids = _page_ids(page)
        if not ids:
            continue
        fetched = index.fetch(ids=ids, namespace=namespace)
        for vid, vec in fetched.vectors.items():
            yield vid, dict(vec.metadata or {})


def _page_ids(page) -> list[str]:
    # Según la versión del SDK, list() produce páginas (ListResponse) o
    # listas planas de IDs.
    if isinstance(page, (list, tuple)):
        return [str(x) for x in page]
    return [item.id for item in page.vectors]


def _wait_until_ready(timeout_seconds: int = 120) -> None:
    pc = get_client()
    start = time.time()
    while not pc.describe_index(config.INDEX_NAME).status.ready:
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"El índice '{config.INDEX_NAME}' no quedó listo a tiempo.")
        time.sleep(2)


def _tag_safe(value: str) -> str:
    """Los tags de Pinecone solo aceptan letras, números, '_' y '-'."""
    return "".join(c if c.isalnum() or c in "_-" else "-" for c in value)[:120]
