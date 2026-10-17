"""
Carga, limpieza y chunking de documentos + esquema de metadatos.

Formatos soportados (los tres que pide la consigna):
  - Markdown (.md) con un front-matter simple (title, category, tags, ...).
  - JSON (.json) con una lista de entradas; cada entrada es una "página".
  - PDF (.pdf) leído página por página con pypdf.

Todos los formatos terminan en el MISMO esquema de metadatos (`ChunkMetadata`)
para evitar el "Schema Drift": si dos procesos nombraran distinto un mismo
campo (p. ej. `category` vs `categoria`) los filtros fallarían en silencio.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TypedDict

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config

SUPPORTED_EXTENSIONS = {".md", ".json", ".pdf"}


class ChunkMetadata(TypedDict):
    """Esquema ÚNICO de metadatos que se guarda junto a cada vector."""

    text: str  # texto original del chunk (evita una DB relacional aparte)
    chunk_id: str  # ID del vector en Pinecone: "<doc_id>#p<page>-c<chunk>"
    doc_id: str  # ID estable del documento (nombre de archivo sin extensión)
    source: str  # nombre de archivo de origen
    title: str  # título del documento
    section: str  # último encabezado Markdown antes del chunk ("" si no hay)
    page: int  # página (PDF), entrada (JSON) o 1 (Markdown)
    category: str  # categoría temática, p. ej. "concurrencia"
    tags: list[str]  # etiquetas libres, permiten filtros con $in
    doc_type: str  # "markdown" | "json" | "pdf"
    library: str  # librería documentada, p. ej. "asyncio"
    language: str  # idioma del contenido
    chunk_index: int  # posición del chunk dentro del documento
    n_tokens: int  # tamaño real del chunk en tokens
    content_hash: str  # hash del documento completo (para reindexar solo si cambió)
    ingested_at: str  # fecha de ingesta (ISO)


@dataclass
class SourceDocument:
    """Un documento ya cargado y limpio, antes de fragmentar."""

    doc_id: str
    source: str
    title: str
    category: str
    tags: list[str]
    doc_type: str
    library: str
    language: str
    pages: list[str] = field(default_factory=list)  # texto por página

    @property
    def content_hash(self) -> str:
        joined = "\n\f".join(self.pages)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


@dataclass
class Chunk:
    id: str
    metadata: ChunkMetadata

    @property
    def text(self) -> str:
        return self.metadata["text"]


# --- Tokenizador ---------------------------------------------------------


def _load_tokenizer():
    """tiktoken necesita descargar su tabla la primera vez; si no hay red,
    se usa una aproximación (~4 caracteres por token) en vez de romper."""
    try:
        return tiktoken.get_encoding(config.TOKENIZER_ENCODING)
    except Exception as exc:  # sin internet, proxy, etc.
        warnings.warn(f"No se pudo cargar tiktoken ({exc!r}); se aproxima el conteo de tokens.")
        return None


_encoding = _load_tokenizer()


def token_len(text: str) -> int:
    if _encoding is not None:
        return len(_encoding.encode(text))
    return max(1, len(text) // 4)


# --- Limpieza y normalización ------------------------------------------


def clean_text(text: str) -> str:
    """Normaliza Unicode (NFC), comillas tipográficas, espacios y saltos de
    línea múltiples, y elimina caracteres de control de una codificación rota."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = text.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c)[0] != "C")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _dedupe_paragraphs(pages: list[str]) -> list[str]:
    """Elimina párrafos repetidos dentro del documento (p. ej. encabezados o
    pies de página que un PDF repite en cada hoja)."""
    seen: set[str] = set()
    result = []
    for page in pages:
        kept = []
        for paragraph in page.split("\n\n"):
            key = paragraph.strip().lower()
            if len(key) > 40 and key in seen:
                continue
            seen.add(key)
            kept.append(paragraph)
        result.append("\n\n".join(kept))
    return result


# --- Loaders --------------------------------------------------------------

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Front-matter tipo YAML muy simple (`clave: valor` por línea)."""
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip().strip('"')
    return meta, raw[match.end() :]


def _split_tags(value) -> list[str]:
    if isinstance(value, list):
        items = value
    else:
        items = str(value or "").strip("[]").split(",")
    return [t.strip().strip("'\"").lower() for t in items if t.strip().strip("'\"")]


def _load_markdown(path: Path) -> SourceDocument:
    meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
    title = meta.get("title") or _first_heading(body) or path.stem
    return SourceDocument(
        doc_id=path.stem,
        source=path.name,
        title=title,
        category=meta.get("category", "general").lower(),
        tags=_split_tags(meta.get("tags")),
        doc_type="markdown",
        library=meta.get("library", path.stem).lower(),
        language=meta.get("language", "es"),
        pages=[clean_text(body)],
    )


def _load_json(path: Path) -> SourceDocument:
    """Formato esperado:
    {"title": ..., "category": ..., "tags": [...], "library": ...,
     "entries": [{"title": ..., "content": ..., "tags": [...]}, ...]}
    Cada entrada se trata como una página independiente."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    pages = []
    tags = _split_tags(data.get("tags"))
    for entry in entries:
        heading = entry.get("title", "").strip()
        content = entry.get("content", "").strip()
        pages.append(clean_text(f"## {heading}\n\n{content}" if heading else content))
        tags.extend(t for t in _split_tags(entry.get("tags")) if t not in tags)
    return SourceDocument(
        doc_id=path.stem,
        source=path.name,
        title=data.get("title", path.stem),
        category=data.get("category", "general").lower(),
        tags=tags,
        doc_type="json",
        library=data.get("library", "python").lower(),
        language=data.get("language", "es"),
        pages=pages,
    )


def _load_pdf(path: Path) -> SourceDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [clean_text(page.extract_text() or "") for page in reader.pages]
    info_title = (reader.metadata.title if reader.metadata else None) or path.stem
    return SourceDocument(
        doc_id=path.stem,
        source=path.name,
        title=str(info_title),
        category="pdf",
        tags=[],
        doc_type="pdf",
        library="python",
        language="es",
        pages=pages,
    )


_LOADERS = {".md": _load_markdown, ".json": _load_json, ".pdf": _load_pdf}


def load_documents(docs_dir: Path = config.DOCS_DIR) -> list[SourceDocument]:
    docs = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or path.name.startswith(("~$", ".")):
            continue
        doc = _LOADERS[path.suffix.lower()](path)
        doc.pages = _dedupe_paragraphs(doc.pages)
        if any(p.strip() for p in doc.pages):
            docs.append(doc)
    if not docs:
        raise FileNotFoundError(f"No hay documentos .md/.json/.pdf en {docs_dir}.")
    return docs


# --- Chunking -----------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _first_heading(text: str) -> str:
    match = _HEADING_RE.search(text)
    return match.group(1).strip() if match else ""


def _section_for_chunk(page_text: str, chunk_text: str, start_index: int) -> str:
    """Sección a la que pertenece un chunk:
    1) si el chunk arranca con un encabezado, ese encabezado;
    2) si no, el último encabezado anterior a la posición del chunk en la página.
    `start_index` del splitter puede venir en -1 cuando el splitter reescribió
    espacios, por eso se re-ubica el chunk buscando su comienzo."""
    first = _HEADING_RE.match(chunk_text)
    if first:
        return first.group(1).strip()
    position = start_index if start_index >= 0 else page_text.find(chunk_text[:80])
    section = ""
    for match in _HEADING_RE.finditer(page_text):
        if position < 0 or match.start() > position:
            break
        section = match.group(1).strip()
    return section


def build_splitter() -> RecursiveCharacterTextSplitter:
    """RecursiveCharacterTextSplitter midiendo tamaño en tokens reales.

    Los separadores priorizan la estructura Markdown (encabezados, bloques de
    código, párrafos) antes de cortar por oraciones o palabras, para no
    partir una función o una sección por la mitad.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        length_function=token_len,
        separators=["\n## ", "\n### ", "\n```\n", "\n\n", "\n", ". ", " ", ""],
        keep_separator="start",
        add_start_index=True,
        strip_whitespace=True,
    )


def chunk_document(doc: SourceDocument, splitter: RecursiveCharacterTextSplitter | None = None) -> list[Chunk]:
    splitter = splitter or build_splitter()
    today = date.today().isoformat()
    content_hash = doc.content_hash
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(doc.pages, start=1):
        if not page_text.strip():
            continue
        for piece in splitter.create_documents([page_text]):
            text = piece.page_content.strip()
            section = _section_for_chunk(page_text, text, piece.metadata.get("start_index", -1))
            chunk_id = f"{doc.doc_id}#p{page_number}-c{len(chunks)}"
            metadata: ChunkMetadata = {
                "text": text,
                "chunk_id": chunk_id,
                "doc_id": doc.doc_id,
                "source": doc.source,
                "title": doc.title,
                "section": section or doc.title,
                "page": page_number,
                "category": doc.category,
                "tags": doc.tags,
                "doc_type": doc.doc_type,
                "library": doc.library,
                "language": doc.language,
                "chunk_index": len(chunks),
                "n_tokens": token_len(text),
                "content_hash": content_hash,
                "ingested_at": today,
            }
            chunks.append(Chunk(id=chunk_id, metadata=metadata))
    return chunks


def chunk_documents(docs: list[SourceDocument]) -> list[Chunk]:
    splitter = build_splitter()
    return [chunk for doc in docs for chunk in chunk_document(doc, splitter)]


def searchable_text(metadata: dict) -> str:
    """Texto que se embebe y se indexa en BM25: el chunk con un encabezado
    contextual (título + sección). Un chunk suelto como "Se usa como
    decorador..." no dice de qué librería habla; con el encabezado, tanto el
    embedding como BM25 saben que es de `functools.lru_cache`."""
    header = metadata.get("title", "")
    section = metadata.get("section", "")
    if section and section != header:
        header = f"{header} — {section}"
    return f"{header}\n\n{metadata.get('text', '')}" if header else metadata.get("text", "")
