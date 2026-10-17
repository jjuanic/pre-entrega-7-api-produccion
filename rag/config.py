"""
Configuración del RAG híbrido reutilizado de la Pre-entrega 4.

Apunta al MISMO índice y namespace de Pinecone que se cargó en esa entrega
(`python-docs-rag` / `python-stdlib-es`), así que no hace falta reingestar:
el Investigador se conecta y consulta. Si no hay credenciales de Pinecone,
`rag.backend` cae a un BM25 local sobre los mismos documentos de `data/docs/`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Rutas --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"

# --- Pinecone -----------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "python-docs-rag")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "python-stdlib-es")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_METRIC = "cosine"
UPSERT_BATCH_SIZE = 100

# --- Embeddings (tienen que coincidir con los de la ingesta de la PE4) ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google").strip().lower()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_EMBEDDING_MODELS = {
    "google": "models/gemini-embedding-001",
    "openai": "text-embedding-3-small",
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or _EMBEDDING_MODELS.get(EMBEDDING_PROVIDER, "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# --- Chunking (solo se usa en el modo local) ----------------------------
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 100
TOKENIZER_ENCODING = "cl100k_base"

# --- Recuperación -------------------------------------------------------
TOP_K = 4
CANDIDATES_PER_RETRIEVER = 10
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.5"))


def require_pinecone_key() -> str:
    if not PINECONE_API_KEY:
        raise RuntimeError("Falta PINECONE_API_KEY en .env.")
    return PINECONE_API_KEY


def require_embedding_key() -> str:
    """Devuelve la API key del proveedor de embeddings configurado."""
    if EMBEDDING_PROVIDER == "google":
        key, name = GOOGLE_API_KEY, "GOOGLE_API_KEY"
    elif EMBEDDING_PROVIDER == "openai":
        key, name = OPENAI_API_KEY, "OPENAI_API_KEY"
    else:
        raise RuntimeError(f"EMBEDDING_PROVIDER='{EMBEDDING_PROVIDER}' no soportado. Usá 'google' u 'openai'.")
    if not key:
        raise RuntimeError(f"Falta {name} (EMBEDDING_PROVIDER={EMBEDDING_PROVIDER}) en .env.")
    return key
