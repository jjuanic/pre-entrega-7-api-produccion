"""
Pipeline de Ingesta a Pinecone Serverless.

    data/docs/*.md|json|pdf
        -> carga + limpieza/normalización + deduplicación
        -> RecursiveCharacterTextSplitter (600 tokens, overlap 100)
        -> metadatos con esquema único (fuente, página, sección, categoría, tags, ...)
        -> embeddings (1536 dims) del texto con encabezado contextual
        -> batch upsert (100 vectores/request) en el namespace configurado

El texto original de cada chunk se guarda en `metadata["text"]`: una sola
consulta a Pinecone trae vector + texto + fuente, sin base relacional aparte.

Reindexación inteligente: los IDs son determinísticos ("<doc_id>#p<pag>-c<n>")
y cada chunk guarda el hash del documento. Si un documento no cambió desde
la última corrida, se saltea (no se paga de nuevo por sus embeddings). Si
cambió, se borran sus chunks viejos y se suben los nuevos.

Uso:
    python ingest.py             # sube solo documentos nuevos o modificados
    python ingest.py --force     # vacía el namespace y reindexa todo
    python ingest.py --dry-run   # solo chunking: muestra chunks y metadata (no requiere API keys)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from rag import config
from rag.documents import Chunk, SourceDocument, chunk_document, load_documents, searchable_text


def _print_chunking_report(docs: list[SourceDocument], chunks_by_doc: dict[str, list[Chunk]]) -> None:
    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    sizes = [c.metadata["n_tokens"] for c in all_chunks]
    print(f"Documentos: {len(docs)} | Chunks: {len(all_chunks)}")
    print(f"Tokens por chunk: min {min(sizes)} / prom {sum(sizes) / len(sizes):.0f} / max {max(sizes)}")
    print("Por categoría:", dict(Counter(c.metadata["category"] for c in all_chunks)))
    for doc in docs:
        print(f"  - {doc.source:<34} [{doc.doc_type:<8}] {len(chunks_by_doc[doc.doc_id])} chunks")


def _existing_hash(index, doc_id: str) -> tuple[list[str], str | None]:
    """IDs ya indexados del documento y el hash con el que se indexaron."""
    from rag.pinecone_store import list_ids

    ids = list_ids(index, prefix=f"{doc_id}#")
    if not ids:
        return [], None
    fetched = index.fetch(ids=ids[:1], namespace=config.NAMESPACE)
    vec = next(iter(fetched.vectors.values()), None)
    return ids, (vec.metadata or {}).get("content_hash") if vec else None


def ingest(force: bool = False) -> int:
    from rag.embeddings import embed_documents_batched, get_embeddings
    from rag.pinecone_store import ensure_index, get_index, namespace_vector_count

    if ensure_index():
        print(f"Índice '{config.INDEX_NAME}' creado (no existía).")
    index = get_index()
    embeddings = get_embeddings()

    if force and namespace_vector_count(index) > 0:
        index.delete(delete_all=True, namespace=config.NAMESPACE)
        print(f"--force: se vació el namespace '{config.NAMESPACE}'.")

    docs = load_documents()
    chunks_by_doc = {doc.doc_id: chunk_document(doc) for doc in docs}
    _print_chunking_report(docs, chunks_by_doc)
    print()

    to_upload: list[Chunk] = []
    for doc in docs:
        old_ids, old_hash = ([], None) if force else _existing_hash(index, doc.doc_id)
        if old_hash == doc.content_hash:
            print(f"  = {doc.source}: sin cambios, se saltea.")
            continue
        if old_ids:
            index.delete(ids=old_ids, namespace=config.NAMESPACE)
            print(f"  ~ {doc.source}: cambió, se borran {len(old_ids)} chunks viejos.")
        else:
            print(f"  + {doc.source}: nuevo.")
        to_upload.extend(chunks_by_doc[doc.doc_id])

    if not to_upload:
        print(f"\nNada para indexar: el namespace '{config.NAMESPACE}' ya está al día.")
        return 0

    print(f"\nGenerando {len(to_upload)} embeddings ({config.EMBEDDING_PROVIDER}: {config.EMBEDDING_MODEL})...")
    vectors = embed_documents_batched(embeddings, [searchable_text(c.metadata) for c in to_upload])

    records = [
        {"id": chunk.id, "values": vector, "metadata": dict(chunk.metadata)}
        for chunk, vector in zip(to_upload, vectors, strict=True)
    ]
    # Batch Upsert: de a UPSERT_BATCH_SIZE vectores por request.
    for start in range(0, len(records), config.UPSERT_BATCH_SIZE):
        batch = records[start : start + config.UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch, namespace=config.NAMESPACE, show_progress=False)
        print(f"  upsert lote {start // config.UPSERT_BATCH_SIZE + 1}: {len(batch)} vectores")

    print(
        f"\nListo: {len(records)} chunks subidos al índice '{config.INDEX_NAME}', "
        f"namespace '{config.NAMESPACE}'. (Pinecone puede tardar unos segundos en "
        "reflejarlos en describe_index_stats.)"
    )
    return len(records)


def dry_run() -> None:
    docs = load_documents()
    chunks_by_doc = {doc.doc_id: chunk_document(doc) for doc in docs}
    _print_chunking_report(docs, chunks_by_doc)
    sample = next(iter(chunks_by_doc.values()))[0]
    preview = {**sample.metadata, "text": sample.metadata["text"][:160] + "..."}
    print("\nMetadata de ejemplo (tal como se sube a Pinecone):")
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de documentos a Pinecone Serverless.")
    parser.add_argument("--force", action="store_true", help="vacía el namespace y reindexa todo")
    parser.add_argument("--dry-run", action="store_true", help="solo chunking, sin llamar a APIs")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        ingest(force=args.force)
