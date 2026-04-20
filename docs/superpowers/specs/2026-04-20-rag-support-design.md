# RAG Support Design

**Date:** 2026-04-20  
**Status:** Approved  
**Topic:** Retrieval-Augmented Generation (RAG) for the Telegram bot

---

## Overview

Add RAG support so the bot can answer questions grounded in a local knowledge base of documents. Documents include local PDFs, plain text files, RFCs fetched from URLs, and web pages crawled from a start URL. Retrieval is always-on — every user message silently searches the vector store and injects relevant chunks into the system prompt before Ollama responds.

---

## Architecture

A new `rag.py` module (`RagManager` class) sits between `bot.py` and `agent.py`. A standalone `ingest.py` CLI script handles bulk ingestion outside the bot.

```
User message
    → bot.py: message handler
    → rag.py: embed query → search collections → return top-k chunks
    → agent.py: inject chunks as system prompt context → Ollama
    → bot.py: reply to user
```

**New files:**
- `rag.py` — `RagManager`: embed, ingest, search, list collections, delete
- `ingest.py` — CLI for bulk ingestion (URL list, recursive crawl, local files)

**Modified files:**
- `bot.py` — add `/ingest` and `/collections` commands; pass RAG context to agent
- `agent.py` — accept optional `context: str` kwarg; prepend as `### Context` block in system prompt
- `config.py` — add `RagConfig` Pydantic model
- `config.example.yaml` — add `rag:` section
- `requirements.txt` — add `chromadb`, `pypdf`, `beautifulsoup4`

---

## Configuration

New `rag:` block in `config.yaml` / `config.example.yaml`:

```yaml
rag:
  enabled: true
  embed_model: "nomic-embed-text"
  db_path: "data/chroma"
  top_k: 4
  similarity_threshold: 0.5
```

- `enabled` — set to `false` to disable RAG entirely; bot continues normally
- `embed_model` — Ollama model used for embeddings; must be pulled before use
- `db_path` — path to ChromaDB persistent storage; mapped to Docker volume
- `top_k` — number of chunks retrieved per query
- `similarity_threshold` — minimum cosine similarity to include a chunk (0–1)

---

## Ingestion Pipeline

The same pipeline is used by both the CLI and the `/ingest` bot command.

**Source types:**

| Source | Detection | Extraction |
|--------|-----------|------------|
| Local PDF | `.pdf` extension | `pypdf` |
| Local text/markdown | `.txt`, `.md` | read directly |
| Plain-text URL (RFC etc.) | URL, no HTML `<html>` tag | `httpx` fetch |
| Web page | URL with HTML content | `httpx` + `beautifulsoup4` |

**Steps:**
1. Fetch/read source → extract plain text
2. Chunk into ~500-token segments with 50-token overlap (character-based approximation: 500 chars, 50-char overlap)
3. Embed each chunk via Ollama embedding API (`/api/embed`)
4. Store in ChromaDB under the named collection with metadata: `{source, chunk_index, timestamp}`
5. Skip chunks whose source URL/path is already present in the collection (deduplication)

---

## Ingestion Sources

### Bot command
```
/ingest <collection> <url-or-path>
```
Examples:
```
/ingest rfcs https://rfc-editor.org/rfc/rfc9110.txt
/ingest work-docs /data/docs/spec.pdf
```

### CLI — single source
```bash
python ingest.py --collection rfcs https://rfc-editor.org/rfc/rfc9110.txt
python ingest.py --collection work-docs /path/to/file.pdf
```

### CLI — URL list (batch)
```bash
python ingest.py --collection rfcs --url-file urls.txt
```
`urls.txt` contains one URL per line. URLs are processed sequentially with progress output.

### CLI — recursive web crawl
```bash
python ingest.py --collection docs --crawl https://example.com --depth 2
```
- Follows `<a href>` links, same domain only
- Respects `--depth` limit (default: 2)
- Skips already-ingested URLs
- Prints progress: `[N/M] https://...`

---

## Query Pipeline

Runs on every user message, transparently.

1. Embed the user message using the same `embed_model`
2. Query all ChromaDB collections (no collection targeting by default)
3. Merge results, rank by similarity score, take top `top_k`
4. Discard any chunk below `similarity_threshold`
5. If chunks remain, prepend to system prompt:

```
### Context
[source: rfc9110.txt, chunk 3]
HTTP defines a stateless request/response protocol...

[source: spec.pdf, chunk 12]
The retry policy must include exponential backoff...
```

6. If no chunks exceed the threshold, query proceeds without injected context (no hallucinated "context")

---

## Bot Commands

| Command | Description |
|---------|-------------|
| `/ingest <collection> <url-or-path>` | Ingest a single document into a named collection |
| `/collections` | List all collections with document and chunk counts |

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| `rag.enabled: false` | RAG skipped entirely; bot works as before |
| ChromaDB unavailable at startup | Log warning; continue without RAG |
| Embedding model not pulled | `/ingest` returns user-facing error: "Pull `nomic-embed-text` first with `ollama pull nomic-embed-text`" |
| PDF parse failure | Skip document; report error to user |
| URL fetch failure (timeout/4xx/5xx) | Skip URL; report error to user or CLI |
| No chunks above threshold | Query proceeds with no injected context |
| Unknown collection name in `/ingest` | Collection is created automatically |

---

## Docker / Deployment

- `data/chroma` directory is stored inside the existing `bot-data` Docker volume (same mount as `data/history.db`)
- No additional Docker services required
- `nomic-embed-text` must be pulled on the Ollama host: `ollama pull nomic-embed-text`

---

## Testing

- Unit tests for `RagManager`: ingest, search, list collections, delete (mocked ChromaDB + Ollama embed API)
- Unit tests for chunking: correct chunk size, overlap, edge cases (empty input, very short text)
- Unit tests for source detection: PDF, text, plain-text URL, HTML URL
- Unit tests for web crawl: link extraction, depth limiting, same-domain filtering, deduplication
- Integration test: ingest a small in-memory text → query → verify chunk appears in result
- All existing 52 tests must remain green

---

## Out of Scope

- Multi-user collection isolation (single-user bot)
- Reranking (cross-encoder reranking)
- Streaming ingestion progress to Telegram
- Automatic RFC discovery or sitemap parsing
