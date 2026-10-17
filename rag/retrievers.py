"""
Los dos recuperadores que combina el sistema híbrido:

- `PineconeRetriever`: búsqueda semántica (similitud coseno) en Pinecone,
  implementado como un `BaseRetriever` de LangChain sobre el SDK nativo.
- `build_bm25_retriever`: búsqueda léxica BM25 (`BM25Retriever` de LangChain)
  con un tokenizador pensado para documentación técnica en español.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from typing import Any

with warnings.catch_warnings():
    # langchain-community avisa que está en "sunset"; BM25Retriever sigue
    # siendo la implementación que pide la consigna.
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

from rag import config
from rag.documents import searchable_text

# --- Recuperador vectorial (Pinecone) --------------------------------------


class PineconeRetriever(BaseRetriever):
    """Retriever semántico sobre un índice de Pinecone.

    Embebe la consulta con el MISMO modelo usado en la ingesta, busca los
    `k` vecinos más cercanos dentro del namespace (opcionalmente con un
    filtro de metadatos) y reconstruye cada `Document` a partir de la
    metadata: el texto original viaja en `metadata["text"]`, así que no
    hace falta ninguna consulta extra a otra base de datos.
    """

    index: Any
    embeddings: Embeddings
    namespace: str = config.NAMESPACE
    k: int = config.CANDIDATES_PER_RETRIEVER
    metadata_filter: dict | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        vector = self.embeddings.embed_query(query)
        response = self.index.query(
            vector=vector,
            top_k=self.k,
            namespace=self.namespace,
            filter=self.metadata_filter or None,
            include_metadata=True,
        )
        docs = []
        for match in response.matches:
            metadata = dict(match.metadata or {})
            text = metadata.pop("text", "")
            metadata["chunk_id"] = metadata.get("chunk_id", match.id)
            metadata["vector_score"] = float(match.score)
            docs.append(Document(page_content=text, metadata=metadata, id=match.id))
        return docs


# --- Recuperador léxico (BM25) -------------------------------------------

# Stopwords en español: palabras demasiado frecuentes para aportar señal en
# BM25 (en inglés se dejan pasar, porque en docs técnicas suelen ser parte
# de identificadores o mensajes de error).
SPANISH_STOPWORDS = frozenset(
    """a al algo algun alguna algunas alguno algunos ante antes aqui asi aun cada como con
    contra cual cuales cuando cuanto de del desde donde dos e el ella ellas ellos en entre era
    es esa esas ese eso esos esta estan estas este esto estos fue hay la las le les lo los mas
    me mi mientras muy no nos o otra otras otro otros para pero poco por porque que quien se
    sea segun ser si sin sobre solo son su sus tambien te tiene tienen todo todos tu tus un una
    uno unos usa usar y ya yo puede pueden hace hacer cuanta cuantas cuantos debe deben""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Tokenizador para BM25 sobre documentación técnica.

    - minúsculas y sin acentos ("¿Qué función...?" == "que funcion"),
    - conserva identificadores de Python completos (`lru_cache`,
      `ensure_ascii`) y además agrega sus partes (`lru`, `cache`), para que
      matcheen tanto la consulta exacta como una parcial,
    - separa nombres con punto (`functools.partial` -> `functools`, `partial`),
    - descarta stopwords en español.
    """
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(_strip_accents(text.lower())):
        if token in SPANISH_STOPWORDS:
            continue
        tokens.append(token)
        if "_" in token.strip("_"):
            tokens.extend(part for part in token.split("_") if part and part not in SPANISH_STOPWORDS)
    return tokens


def build_bm25_retriever(records: list[dict], k: int = config.CANDIDATES_PER_RETRIEVER) -> BM25Retriever:
    """Construye el BM25Retriever a partir de los registros (metadata) del corpus.

    BM25 indexa el mismo `searchable_text` que se embebió (título + sección
    + chunk), pero los `Document` que devuelve tienen como contenido solo el
    texto original del chunk, igual que el retriever de Pinecone.
    """
    if not records:
        raise ValueError("El corpus para BM25 está vacío. ¿Corriste ingest.py?")
    docs = []
    for record in records:
        metadata = {key: value for key, value in record.items() if key != "text"}
        docs.append(Document(page_content=record["text"], metadata=metadata, id=record["chunk_id"]))
    vectorizer = BM25Okapi([tokenize(searchable_text(r)) for r in records])
    return BM25Retriever(vectorizer=vectorizer, docs=docs, k=k, preprocess_func=tokenize)


def matches_filter(metadata: dict, metadata_filter: dict | None) -> bool:
    """Aplica en memoria (para BM25) el mismo subconjunto de filtros de
    Pinecone que se usa en la búsqueda vectorial: igualdad, $eq, $ne, $in y
    $nin, incluyendo campos lista como `tags`."""
    if not metadata_filter:
        return True
    for field_name, condition in metadata_filter.items():
        value = metadata.get(field_name)
        values = value if isinstance(value, list) else [value]
        if not isinstance(condition, dict):
            condition = {"$eq": condition}
        for op, expected in condition.items():
            if op == "$eq" and expected not in values:
                return False
            if op == "$ne" and expected in values:
                return False
            if op == "$in" and not any(v in expected for v in values):
                return False
            if op == "$nin" and any(v in expected for v in values):
                return False
            if op not in {"$eq", "$ne", "$in", "$nin"}:
                raise ValueError(f"Operador de filtro no soportado para BM25: {op}")
    return True
