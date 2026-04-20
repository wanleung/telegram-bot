# RAG Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add always-on Retrieval-Augmented Generation so every user message is enriched with relevant chunks from a local ChromaDB knowledge base populated from PDFs, text files, RFC URLs, and crawled web pages.

**Architecture:** A new `rag.py` (`RagManager`) handles embedding via Ollama and ChromaDB storage/retrieval. `agent.py` accepts an optional `context` string prepended to the system prompt. `bot.py` calls `RagManager.search()` before every `agent.run()` call and adds `/ingest` + `/collections` commands. A standalone `ingest.py` CLI handles bulk ingestion and web crawling.

**Tech Stack:** `chromadb`, `pypdf`, `beautifulsoup4`, `httpx` (already used via `ollama`), Ollama embedding API (`/api/embed`), Python 3.12, pytest-asyncio.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `rag.py` | Create | `RagManager`: embed, ingest chunks, search all collections, list, delete |
| `ingest.py` | Create | CLI: single source, URL-file batch, recursive web crawl |
| `config.py` | Modify | Add `RagConfig` Pydantic model; add to `Config` |
| `config.example.yaml` | Modify | Add `rag:` section |
| `config.yaml` | Modify | Add `rag:` section (user's live config) |
| `agent.py` | Modify | Accept `context: str \| None` kwarg; prepend `### Context` block |
| `bot.py` | Modify | Call RAG search before `agent.run()`; add `/ingest`, `/collections` commands; update `/start` help |
| `requirements.txt` | Modify | Add `chromadb`, `pypdf`, `beautifulsoup4` |
| `tests/test_rag.py` | Create | Unit tests for `RagManager` (mocked ChromaDB + Ollama) |
| `tests/test_ingest.py` | Create | Unit tests for chunking, source detection, web crawl helpers |
| `tests/test_agent_rag.py` | Create | Tests for `agent.run()` with `context` kwarg |
| `tests/test_bot_rag.py` | Create | Tests for `/ingest` and `/collections` bot commands |

---

## Task 1: Add dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add new packages**

Edit `requirements.txt` to be:
```
python-telegram-bot==21.*
ollama
mcp[cli]
aiosqlite
pydantic>=2.0
pyyaml
markdown
chromadb
pypdf
beautifulsoup4
pytest
pytest-asyncio
```

- [ ] **Step 2: Install**

```bash
cd /home/wanleung/Projects/telegram-bot
source venv/bin/activate
pip install chromadb pypdf beautifulsoup4
```

Expected: packages install without error.

- [ ] **Step 3: Verify existing tests still pass**

```bash
python -m pytest --tb=short -q
```

Expected: `52 passed`.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add chromadb, pypdf, beautifulsoup4 for RAG support"
```

---

## Task 2: Add `RagConfig` to config

**Files:**
- Modify: `config.py`
- Modify: `config.example.yaml`
- Modify: `config.yaml`

- [ ] **Step 1: Write failing test**

Create `tests/test_config_rag.py`:
```python
import pytest
from config import load_config, RagConfig


def test_rag_config_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "telegram:\n  token: tok\n"
        "ollama:\n  default_model: llama3.2\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg.rag.enabled is False
    assert cfg.rag.embed_model == "nomic-embed-text"
    assert cfg.rag.db_path == "data/chroma"
    assert cfg.rag.top_k == 4
    assert cfg.rag.similarity_threshold == 0.5


def test_rag_config_custom(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "telegram:\n  token: tok\n"
        "ollama:\n  default_model: llama3.2\n"
        "rag:\n  enabled: true\n  embed_model: mxbai-embed-large\n  top_k: 6\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg.rag.enabled is True
    assert cfg.rag.embed_model == "mxbai-embed-large"
    assert cfg.rag.top_k == 6
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_config_rag.py -v
```

Expected: `ImportError` or `AttributeError` — `RagConfig` not yet defined.

- [ ] **Step 3: Add `RagConfig` to `config.py`**

Add after `HistoryConfig` class and update `Config`:

```python
class RagConfig(BaseModel):
    enabled: bool = False
    embed_model: str = "nomic-embed-text"
    db_path: str = "data/chroma"
    top_k: int = 4
    similarity_threshold: float = 0.5


class Config(BaseModel):
    telegram: TelegramConfig
    ollama: OllamaConfig
    history: HistoryConfig = HistoryConfig()
    rag: RagConfig = RagConfig()
    mcp_servers: dict[str, MCPServerConfig] = {}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_config_rag.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Update `config.example.yaml`**

Add after the `history:` block:
```yaml
rag:
  enabled: false
  embed_model: "nomic-embed-text"
  db_path: "data/chroma"
  top_k: 4
  similarity_threshold: 0.5
```

- [ ] **Step 6: Update `config.yaml` (live config)**

Add the same `rag:` block to `config.yaml`. Set `enabled: true` if Ollama has `nomic-embed-text` pulled; otherwise leave `false`.

- [ ] **Step 7: Run all tests**

```bash
python -m pytest --tb=short -q
```

Expected: `54 passed`.

- [ ] **Step 8: Commit**

```bash
git add config.py config.example.yaml config.yaml tests/test_config_rag.py
git commit -m "feat: add RagConfig to config with defaults"
```

---

## Task 3: Create `rag.py` — RagManager

**Files:**
- Create: `rag.py`
- Create: `tests/test_rag.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_rag.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from config import RagConfig
from rag import RagManager


@pytest.fixture
def rag_cfg(tmp_path):
    return RagConfig(
        enabled=True,
        embed_model="nomic-embed-text",
        db_path=str(tmp_path / "chroma"),
        top_k=2,
        similarity_threshold=0.0,
    )


@pytest.fixture
def mock_chroma(monkeypatch):
    """Patch chromadb.PersistentClient to return a mock."""
    col = MagicMock()
    col.count.return_value = 0
    col.get.return_value = {"ids": []}
    col.query.return_value = {
        "ids": [["id1"]],
        "documents": [["chunk text"]],
        "metadatas": [[{"source": "test.txt", "chunk_index": 0}]],
        "distances": [[0.1]],
    }
    client = MagicMock()
    client.get_or_create_collection.return_value = col
    client.list_collections.return_value = [MagicMock(name="test")]
    monkeypatch.setattr("rag.chromadb.PersistentClient", lambda path: client)
    return client, col


@pytest.mark.asyncio
async def test_ingest_text(rag_cfg, mock_chroma):
    client, col = mock_chroma
    manager = RagManager(rag_cfg)

    with patch.object(manager, "_embed", new=AsyncMock(return_value=[0.1] * 768)):
        count = await manager.ingest("mycol", "test.txt", "Hello world " * 100)

    assert count > 0
    assert col.add.called


@pytest.mark.asyncio
async def test_ingest_skips_duplicate(rag_cfg, mock_chroma):
    client, col = mock_chroma
    col.get.return_value = {"ids": ["existing-id"]}  # source already present
    manager = RagManager(rag_cfg)

    with patch.object(manager, "_embed", new=AsyncMock(return_value=[0.1] * 768)):
        count = await manager.ingest("mycol", "test.txt", "Hello world " * 100)

    assert count == 0
    col.add.assert_not_called()


@pytest.mark.asyncio
async def test_search_returns_chunks(rag_cfg, mock_chroma):
    client, col = mock_chroma
    manager = RagManager(rag_cfg)

    with patch.object(manager, "_embed", new=AsyncMock(return_value=[0.1] * 768)):
        chunks = await manager.search("who am I?")

    assert len(chunks) == 1
    assert "chunk text" in chunks[0]
    assert "test.txt" in chunks[0]


@pytest.mark.asyncio
async def test_search_empty_when_disabled(tmp_path):
    cfg = RagConfig(enabled=False, db_path=str(tmp_path / "chroma"))
    manager = RagManager(cfg)
    chunks = await manager.search("anything")
    assert chunks == []


def test_chunk_text_splits_correctly():
    from rag import chunk_text
    text = "a" * 1200
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) == 3
    assert chunks[0] == "a" * 500
    assert chunks[1] == "a" * 500  # starts at offset 450
    assert len(chunks[2]) > 0


def test_chunk_text_short_input():
    from rag import chunk_text
    chunks = chunk_text("hello", size=500, overlap=50)
    assert chunks == ["hello"]


def test_chunk_text_empty():
    from rag import chunk_text
    chunks = chunk_text("", size=500, overlap=50)
    assert chunks == []


def test_list_collections(rag_cfg, mock_chroma):
    client, col = mock_chroma
    manager = RagManager(rag_cfg)
    cols = manager.list_collections()
    assert isinstance(cols, list)
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_rag.py -v
```

Expected: `ModuleNotFoundError: No module named 'rag'`.

- [ ] **Step 3: Create `rag.py`**

```python
"""RAG module: ChromaDB-backed vector store with Ollama embeddings."""

import hashlib
import logging
from typing import Any

import chromadb
import ollama

from config import RagConfig

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
    """Manages document ingestion and retrieval using ChromaDB + Ollama embeddings."""

    def __init__(self, cfg: RagConfig) -> None:
        self._cfg = cfg
        if cfg.enabled:
            try:
                self._client = chromadb.PersistentClient(path=cfg.db_path)
            except Exception as exc:
                logger.warning("ChromaDB unavailable, RAG disabled: %s", exc)
                self._cfg = RagConfig(enabled=False)
        self._ollama = ollama.AsyncClient()

    async def _embed(self, text: str) -> list[float]:
        """Return embedding vector for *text* using the configured embed model."""
        response = await self._ollama.embed(
            model=self._cfg.embed_model,
            input=text,
        )
        return response.embeddings[0]

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
                similarity = 1.0 - dist  # cosine distance → similarity
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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_rag.py -v
```

Expected: `8 passed`.

- [ ] **Step 5: Run all tests**

```bash
python -m pytest --tb=short -q
```

Expected: `62 passed` (54 + 8).

- [ ] **Step 6: Commit**

```bash
git add rag.py tests/test_rag.py
git commit -m "feat: add RagManager with ChromaDB + Ollama embedding support"
```

---

## Task 4: Create `ingest.py` — ingestion helpers and CLI

**Files:**
- Create: `ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ingest.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ingest import (
    detect_source_type,
    extract_text_from_pdf,
    fetch_url_text,
    collect_links,
)


def test_detect_source_type_pdf():
    assert detect_source_type("/path/to/file.pdf") == "pdf"


def test_detect_source_type_text():
    assert detect_source_type("/path/to/readme.txt") == "text"
    assert detect_source_type("/path/to/spec.md") == "text"


def test_detect_source_type_url():
    assert detect_source_type("https://example.com/page") == "url"
    assert detect_source_type("http://rfc-editor.org/rfc/rfc9110.txt") == "url"


def test_collect_links_same_domain():
    html = """<html><body>
    <a href="/page1">P1</a>
    <a href="https://example.com/page2">P2</a>
    <a href="https://other.com/page3">P3</a>
    </body></html>"""
    links = collect_links(html, base_url="https://example.com")
    assert "https://example.com/page1" in links
    assert "https://example.com/page2" in links
    assert "https://other.com/page3" not in links


def test_collect_links_deduplication():
    html = """<html><body>
    <a href="/page">A</a>
    <a href="/page">B</a>
    </body></html>"""
    links = collect_links(html, base_url="https://example.com")
    assert links.count("https://example.com/page") == 1


@pytest.mark.asyncio
async def test_fetch_url_text_plain():
    mock_response = MagicMock()
    mock_response.text = "RFC content here"
    mock_response.headers = {"content-type": "text/plain"}
    mock_response.raise_for_status = MagicMock()

    with patch("ingest.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        text = await fetch_url_text("https://example.com/rfc.txt")
    assert text == "RFC content here"


@pytest.mark.asyncio
async def test_fetch_url_text_html_strips_tags():
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Hello world</p></body></html>"
    mock_response.headers = {"content-type": "text/html"}
    mock_response.raise_for_status = MagicMock()

    with patch("ingest.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        text = await fetch_url_text("https://example.com/page.html")
    assert "Hello world" in text
    assert "<p>" not in text
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_ingest.py -v
```

Expected: `ModuleNotFoundError: No module named 'ingest'`.

- [ ] **Step 3: Create `ingest.py`**

```python
"""Ingestion CLI and helper functions for the RAG knowledge base."""

import argparse
import asyncio
import logging
import os
import sys
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config import load_config
from rag import RagManager

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def detect_source_type(source: str) -> str:
    """Return 'pdf', 'text', or 'url' based on the source string."""
    if source.startswith("http://") or source.startswith("https://"):
        return "url"
    if source.lower().endswith(".pdf"):
        return "pdf"
    return "text"


def extract_text_from_pdf(path: str) -> str:
    """Extract plain text from a PDF file using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


async def fetch_url_text(url: str) -> str:
    """Fetch a URL and return plain text (strips HTML if necessary)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        return response.text


def collect_links(html: str, base_url: str) -> list[str]:
    """Extract same-domain absolute links from an HTML string."""
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(base_url).netloc
    seen: set[str] = set()
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.netloc == base_domain and absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


async def ingest_source(
    manager: RagManager, collection: str, source: str
) -> int:
    """Ingest a single source (path or URL) into *collection*. Returns chunk count."""
    source_type = detect_source_type(source)
    try:
        if source_type == "pdf":
            text = extract_text_from_pdf(source)
        elif source_type == "url":
            text = await fetch_url_text(source)
        else:
            with open(source, encoding="utf-8") as f:
                text = f.read()
    except Exception as exc:
        logger.error("Failed to read %s: %s", source, exc)
        return 0

    return await manager.ingest(collection, source, text)


async def crawl(
    manager: RagManager,
    collection: str,
    start_url: str,
    max_depth: int,
) -> None:
    """Recursively crawl *start_url* up to *max_depth* and ingest all pages."""
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(start_url, 0)]
    base_domain = urlparse(start_url).netloc

    while queue:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"[{len(visited)}/~] {url}")
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                response = await client.get(url)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                html = response.text
        except Exception as exc:
            logger.warning("Skipping %s: %s", url, exc)
            continue

        if "html" in content_type:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

            if depth < max_depth:
                for link in collect_links(html, url):
                    if link not in visited and urlparse(link).netloc == base_domain:
                        queue.append((link, depth + 1))
        else:
            text = html

        await manager.ingest(collection, url, text)


async def main_async(args: argparse.Namespace) -> None:
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    cfg = load_config(config_path)
    manager = RagManager(cfg.rag)

    if not cfg.rag.enabled:
        print("RAG is disabled in config. Set rag.enabled: true to use ingest.")
        sys.exit(1)

    collection = args.collection

    if args.crawl:
        depth = args.depth if hasattr(args, "depth") and args.depth else 2
        await crawl(manager, collection, args.crawl, max_depth=depth)

    elif args.url_file:
        with open(args.url_file) as f:
            urls = [line.strip() for line in f if line.strip()]
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {url}")
            count = await ingest_source(manager, collection, url)
            print(f"  → {count} chunks ingested")

    elif args.source:
        count = await ingest_source(manager, collection, args.source)
        print(f"Ingested {count} chunks from {args.source} into '{collection}'")

    else:
        print("Specify a source, --url-file, or --crawl. See --help.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base")
    parser.add_argument("--collection", required=True, help="Collection name")
    parser.add_argument("source", nargs="?", help="URL or file path to ingest")
    parser.add_argument("--url-file", metavar="FILE", help="File with one URL per line")
    parser.add_argument("--crawl", metavar="URL", help="Start URL for recursive crawl")
    parser.add_argument("--depth", type=int, default=2, help="Max crawl depth (default: 2)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_ingest.py -v
```

Expected: `7 passed`.

- [ ] **Step 5: Run all tests**

```bash
python -m pytest --tb=short -q
```

Expected: `69 passed`.

- [ ] **Step 6: Commit**

```bash
git add ingest.py tests/test_ingest.py
git commit -m "feat: add ingest CLI with URL-file batch and recursive crawl support"
```

---

## Task 5: Update `agent.py` to accept RAG context

**Files:**
- Modify: `agent.py` (lines 102–130: `run()` signature and system prompt construction)
- Create: `tests/test_agent_rag.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_agent_rag.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from config import Config, TelegramConfig, OllamaConfig, HistoryConfig, RagConfig
from agent import Agent
from mcp_manager import MCPManager


def make_config():
    return Config(
        telegram=TelegramConfig(token="tok"),
        ollama=OllamaConfig(default_model="llama3.2"),
        history=HistoryConfig(db_path=":memory:"),
        rag=RagConfig(enabled=False),
    )


@pytest.mark.asyncio
async def test_agent_run_injects_context(tmp_path):
    cfg = make_config()
    cfg.history.db_path = str(tmp_path / "h.db")
    mcp = MagicMock(spec=MCPManager)
    mcp.get_tool_definitions.return_value = []

    agent = Agent(cfg, mcp)

    fake_response = MagicMock()
    fake_response.message.content = "42"
    fake_response.message.tool_calls = None

    with patch("agent.get_history", new=AsyncMock(return_value=[])), \
         patch("agent.save_messages", new=AsyncMock()), \
         patch.object(agent._client, "chat", new=AsyncMock(return_value=fake_response)):

        result = await agent.run(
            chat_id=1,
            user_message="What is the answer?",
            context="### Context\n[source: test.txt, chunk 0]\nThe answer is 42.",
        )

    assert result == "42"
    call_args = agent._client.chat.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0] if call_args.args else call_args.kwargs["messages"]
    # Find the system message
    system_msgs = [m for m in messages if m.get("role") == "system"]
    assert any("### Context" in m.get("content", "") for m in system_msgs)


@pytest.mark.asyncio
async def test_agent_run_no_context_no_system_message(tmp_path):
    cfg = make_config()
    cfg.history.db_path = str(tmp_path / "h.db")
    mcp = MagicMock(spec=MCPManager)
    mcp.get_tool_definitions.return_value = []

    agent = Agent(cfg, mcp)

    fake_response = MagicMock()
    fake_response.message.content = "hello"
    fake_response.message.tool_calls = None

    with patch("agent.get_history", new=AsyncMock(return_value=[])), \
         patch("agent.save_messages", new=AsyncMock()), \
         patch.object(agent._client, "chat", new=AsyncMock(return_value=fake_response)):

        result = await agent.run(chat_id=1, user_message="hi", context=None)

    assert result == "hello"
    call_args = agent._client.chat.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0] if call_args.args else call_args.kwargs["messages"]
    system_msgs = [m for m in messages if m.get("role") == "system"]
    assert not any("### Context" in m.get("content", "") for m in system_msgs)
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_agent_rag.py -v
```

Expected: `TypeError: run() got an unexpected keyword argument 'context'`.

- [ ] **Step 3: Update `agent.py` `run()` signature and message construction**

Change the `run()` method signature from:
```python
async def run(
    self,
    chat_id: int,
    user_message: str,
    images: list[str] | None = None,
) -> str:
```

To:
```python
async def run(
    self,
    chat_id: int,
    user_message: str,
    images: list[str] | None = None,
    context: str | None = None,
) -> str:
```

Then in the body, after fetching `history` and `tools`, add context injection before building `messages`:

```python
        history = await get_history(
            self._cfg.history.db_path, chat_id, self._cfg.history.max_messages
        )
        tools = self._mcp.get_tool_definitions()

        # Prepend RAG context as a system message when provided.
        prefix: list[dict] = []
        if context:
            prefix = [{"role": "system", "content": context}]

        # Build the outgoing user message; attach images for vision models.
        user_msg: dict = {"role": "user", "content": user_message}
        if images:
            user_msg["images"] = images

        # History placeholder — store caption text only, never raw base64.
        history_user_content = (
            f"[image] {user_message}" if images else user_message
        )

        messages = prefix + history + [user_msg]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_agent_rag.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Run all tests**

```bash
python -m pytest --tb=short -q
```

Expected: `71 passed`.

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/test_agent_rag.py
git commit -m "feat: agent accepts optional RAG context injected as system message"
```

---

## Task 6: Update `bot.py` — wire RAG into message handler and add commands

**Files:**
- Modify: `bot.py`
- Create: `tests/test_bot_rag.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_bot_rag.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, Message, Chat, User
from telegram.ext import ContextTypes
from config import Config, TelegramConfig, OllamaConfig, HistoryConfig, RagConfig
from rag import RagManager
from bot import cmd_ingest, cmd_collections


def make_update(text: str, args: list[str] | None = None):
    user = MagicMock(spec=User)
    chat = MagicMock(spec=Chat)
    chat.id = 42
    msg = MagicMock(spec=Message)
    msg.text = text
    msg.reply_text = AsyncMock()
    update = MagicMock(spec=Update)
    update.message = msg
    update.effective_chat = chat
    return update


def make_context(args: list[str] | None = None):
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    ctx.args = args or []
    rag = MagicMock(spec=RagManager)
    rag.list_collections.return_value = [{"name": "rfcs", "count": 42}]
    rag.ingest = AsyncMock(return_value=5)
    ctx.bot_data = {"rag": rag}
    return ctx


@pytest.mark.asyncio
async def test_cmd_collections_lists(monkeypatch):
    update = make_update("/collections")
    ctx = make_context()
    await cmd_collections(update, ctx)
    update.message.reply_text.assert_called_once()
    text = update.message.reply_text.call_args.args[0]
    assert "rfcs" in text
    assert "42" in text


@pytest.mark.asyncio
async def test_cmd_collections_empty(monkeypatch):
    update = make_update("/collections")
    ctx = make_context()
    ctx.bot_data["rag"].list_collections.return_value = []
    await cmd_collections(update, ctx)
    text = update.message.reply_text.call_args.args[0]
    assert "No collections" in text


@pytest.mark.asyncio
async def test_cmd_ingest_requires_args(monkeypatch):
    update = make_update("/ingest")
    ctx = make_context(args=[])
    await cmd_ingest(update, ctx)
    text = update.message.reply_text.call_args.args[0]
    assert "Usage" in text


@pytest.mark.asyncio
async def test_cmd_ingest_calls_manager(monkeypatch):
    update = make_update("/ingest rfcs https://example.com/rfc.txt")
    ctx = make_context(args=["rfcs", "https://example.com/rfc.txt"])

    with patch("bot.ingest_source", new=AsyncMock(return_value=7)):
        await cmd_ingest(update, ctx)

    text = update.message.reply_text.call_args.args[0]
    assert "7" in text
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_bot_rag.py -v
```

Expected: `ImportError` — `cmd_ingest`, `cmd_collections` not defined in `bot.py`.

- [ ] **Step 3: Update `bot.py`**

Add import at the top (after existing imports):
```python
from rag import RagManager
from ingest import ingest_source
```

Add two new command handlers after `cmd_tools`:
```python
async def cmd_ingest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ingest <collection> <url-or-path> — ingest a document into RAG."""
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /ingest <collection> <url-or-path>\n"
            "Example: /ingest rfcs https://rfc-editor.org/rfc/rfc9110.txt"
        )
        return
    collection = context.args[0]
    source = context.args[1]
    rag: RagManager = context.bot_data["rag"]
    thinking = await update.message.reply_text(f"⏳ Ingesting {source}…")
    try:
        count = await ingest_source(rag, collection, source)
        if count == 0:
            await thinking.edit_text(f"ℹ️ Already ingested or empty: {source}")
        else:
            await thinking.edit_text(f"✅ Ingested {count} chunks into '{collection}'.")
    except Exception as exc:
        await thinking.edit_text(f"⚠️ Ingest failed: {exc}")


async def cmd_collections(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /collections — list all RAG collections with chunk counts."""
    rag: RagManager = context.bot_data["rag"]
    cols = rag.list_collections()
    if not cols:
        await update.message.reply_text("No collections found. Use /ingest to add documents.")
        return
    lines = "\n".join(f"  • <b>{c['name']}</b> — {c['count']} chunks" for c in cols)
    await update.message.reply_text(f"<b>RAG Collections:</b>\n{lines}", parse_mode="HTML")
```

Update `handle_message` to call RAG search before `agent.run()`. Replace the existing `handle_message` function body:

```python
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle user text messages and photo uploads.

    Searches the RAG knowledge base and injects relevant context into the
    agent prompt before calling Ollama.
    """
    agent: Agent = context.bot_data["agent"]
    rag: RagManager = context.bot_data["rag"]
    thinking = await update.message.reply_text("⏳ Thinking…")

    images: list[str] | None = None
    user_text = update.message.text or update.message.caption or ""

    if update.message.photo:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        image_bytes = await file.download_as_bytearray()
        images = [base64.b64encode(image_bytes).decode()]

    # RAG: retrieve relevant chunks and format as context block
    rag_context: str | None = None
    try:
        chunks = await rag.search(user_text)
        if chunks:
            rag_context = "### Context\n" + "\n\n".join(chunks)
    except Exception as exc:
        logger.warning("RAG search failed, continuing without context: %s", exc)

    reply = await agent.run(
        chat_id=update.effective_chat.id,
        user_message=user_text,
        images=images,
        context=rag_context,
    )
    formatted = md_to_telegram_html(reply) if reply else "_(no response)_"
    parse_mode = "HTML" if reply else "Markdown"
    await thinking.edit_text(formatted, parse_mode=parse_mode)
```

Update `_post_init` to initialise `RagManager` and store in `bot_data`:
```python
async def _post_init(application: Application) -> None:
    """Run async startup tasks after the bot is initialized."""
    cfg = application.bot_data["config"]
    os.makedirs(os.path.dirname(cfg.history.db_path) or ".", exist_ok=True)
    await init_db(cfg.history.db_path)
    mcp: MCPManager = application.bot_data["mcp"]
    await mcp.start()
    application.bot_data["rag"] = RagManager(cfg.rag)
```

Update `main()` to register new commands and update `/start` help text.

Replace the `/start` reply text in `cmd_start`:
```python
    await update.message.reply_text(
        f"👋 Hello! I'm your Ollama-powered assistant.\n"
        f"Current model: `{agent.active_model}`\n\n"
        f"Commands:\n"
        f"  /models — list available models\n"
        f"  /model <name> — switch model\n"
        f"  /clear — clear conversation history\n"
        f"  /tools — list available MCP tools\n"
        f"  /ingest <collection> <url-or-path> — add document to RAG\n"
        f"  /collections — list RAG knowledge base collections",
        parse_mode="Markdown",
    )
```

Add handlers in `main()` after existing ones:
```python
    app.add_handler(CommandHandler("ingest", cmd_ingest))
    app.add_handler(CommandHandler("collections", cmd_collections))
```

- [ ] **Step 4: Run bot RAG tests**

```bash
python -m pytest tests/test_bot_rag.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Run all tests**

```bash
python -m pytest --tb=short -q
```

Expected: `75 passed`.

- [ ] **Step 6: Commit**

```bash
git add bot.py tests/test_bot_rag.py
git commit -m "feat: wire RAG into bot — search on every message, add /ingest and /collections commands"
```

---

## Task 7: Update README and push

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add RAG section to README**

Add a new `## RAG (Retrieval-Augmented Generation)` section documenting:
- Prerequisites: `ollama pull nomic-embed-text`
- Config: the `rag:` block with field descriptions
- Ingestion commands: single source, URL-file batch, recursive crawl CLI examples
- Bot commands: `/ingest`, `/collections`
- How always-on retrieval works

- [ ] **Step 2: Run all tests one final time**

```bash
python -m pytest --tb=short -q
```

Expected: `75 passed`.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: document RAG support — config, ingestion CLI, bot commands"
git push origin master
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - ChromaDB + Ollama embeddings ✅ (Task 3)
  - Named collections ✅ (`RagManager.list_collections`, `/collections`)
  - PDF, text, URL, HTML web page sources ✅ (Task 4, `detect_source_type`)
  - URL-file batch ingestion ✅ (`--url-file`)
  - Recursive web crawl ✅ (`--crawl`, `--depth`, `collect_links`)
  - Always-on retrieval ✅ (`handle_message` calls `rag.search`)
  - `/ingest` bot command ✅ (Task 6)
  - `/collections` bot command ✅ (Task 6)
  - Graceful degradation when RAG disabled or unavailable ✅ (Tasks 3, 6)
  - Docker: `data/chroma` inside existing volume ✅ (no new Docker changes needed)
  - Config `rag:` section ✅ (Task 2)

- [x] **No placeholders:** all code blocks are complete and runnable
- [x] **Type consistency:** `RagManager.ingest(collection, source, text)` and `ingest_source(rag, collection, source)` used consistently across tasks 3, 4, and 6
- [x] **`context` kwarg:** defined in Task 5, used in Task 6 `agent.run(..., context=rag_context)`
