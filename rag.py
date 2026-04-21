"""RAG module: ChromaDB-backed vector store with pluggable embedding backend."""

import hashlib
import logging
from typing import TYPE_CHECKING, Any

import chromadb

from config import RagConfig

if TYPE_CHECKING:
    from llm_backend import LLMBackend

logger = logging.getLogger(__name__)


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split *text* into overlapping character-based chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += size - overlap
    return chunks


class RagManager:
    """Manages document ingestion and retrieval using ChromaDB + a pluggable embedding backend."""

    def __init__(self, cfg: RagConfig, embed_backend: "LLMBackend") -> None:
        self._cfg = cfg
        self._embed_backend = embed_backend
        if cfg.enabled:
            try:
                self._client = chromadb.PersistentClient(path=cfg.db_path)
            except Exception as exc:
                logger.warning("ChromaDB unavailable, RAG disabled: %s", exc)
                self._cfg = RagConfig(enabled=False)

    async def _embed(self, text: str) -> list[float]:
        """Return embedding vector for *text* using the configured embed backend."""
        return await self._embed_backend.embed(self._cfg.embed_model, text)

    def _get_collection(self, name: str) -> Any:
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def _source_id(self, source: str) -> str:
        return hashlib.md5(source.encode()).hexdigest()

    async def ingest(
        self, collection_name: str, source: str, text: str
    ) -> int:
        """
        Ingest *text* from *source* into *collection_name*.

        Returns the number of chunks added (0 if already ingested).
        """
        if not self._cfg.enabled:
            return 0

        col = self._get_collection(collection_name)
        source_id = self._source_id(source)

        existing = col.get(where={"source": source}, include=[])
        if existing["ids"]:
            logger.info("Source already ingested, skipping: %s", source)
            return 0

        chunks = chunk_text(text)
        if not chunks:
            return 0

        ids, embeddings, documents, metadatas = [], [], [], []
        for i, chunk in enumerate(chunks):
            emb = await self._embed(chunk)
            ids.append(f"{source_id}-{i}")
            embeddings.append(emb)
            documents.append(chunk)
            metadatas.append({"source": source, "chunk_index": i})

        col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.info("Ingested %d chunks from %s into '%s'", len(chunks), source, collection_name)
        return len(chunks)

    async def search(self, query: str) -> list[str]:
        """
        Search all collections for chunks relevant to *query*.

        Returns a list of formatted strings: '[source: X, chunk N]\\ntext'.
        """
        if not self._cfg.enabled:
            return []

        try:
            collection_names = [c.name for c in self._client.list_collections()]
        except Exception as exc:
            logger.warning("Failed to list collections: %s", exc)
            return []

        if not collection_names:
            return []

        try:
            query_emb = await self._embed(query)
        except Exception as exc:
            logger.warning("Embedding failed, skipping RAG: %s", exc)
            return []

        all_results: list[tuple[float, str]] = []
        for name in collection_names:
            col = self._get_collection(name)
            try:
                results = col.query(
                    query_embeddings=[query_emb],
                    n_results=min(self._cfg.top_k, col.count() or 1),
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as exc:
                logger.warning("Query failed on collection '%s': %s", name, exc)
                continue

            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = 1.0 - dist
                if similarity >= self._cfg.similarity_threshold:
                    label = f"[source: {meta['source']}, chunk {meta['chunk_index']}]"
                    all_results.append((similarity, f"{label}\n{doc}"))

        all_results.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in all_results[: self._cfg.top_k]]

    def list_collections(self) -> list[dict]:
        """Return list of {name, count} dicts for all collections."""
        if not self._cfg.enabled:
            return []
        try:
            cols = self._client.list_collections()
            result = []
            for c in cols:
                col = self._get_collection(c.name)
                result.append({"name": c.name, "count": col.count()})
            return result
        except Exception as exc:
            logger.warning("Failed to list collections: %s", exc)
            return []
