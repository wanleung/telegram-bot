# Telegram Ollama Bot

A single-user Telegram bot powered by a local [Ollama](https://ollama.com) instance with [MCP](https://modelcontextprotocol.io) tool-calling and [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) support.

## Features

- 🤖 Chat with any locally running Ollama model
- ⚡ **Streaming responses** — replies appear token-by-token as the model generates them
- 🧠 **Thinking mode** — toggle chain-of-thought reasoning per chat via `/think` (shown as a spoiler)
- 🔧 MCP tool-calling — connect stdio, SSE, and HTTP MCP servers
- 📚 RAG — ground responses in local documents (RFCs, PDFs, web pages)
- 🖼️ Image upload support for vision models (llava, llama3.2-vision, etc.)
- 💾 Persistent per-chat conversation history (SQLite)
- 🔄 Switch models at runtime via `/model`
- ⚙️ YAML config with environment variable interpolation

## Quick Start

```bash
# 1. Clone and install
git clone git@github.com:wanleung/telegram-bot.git
cd telegram-bot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp config.example.yaml config.yaml
# Edit config.yaml: set telegram.token and ollama.default_model

# 3. Pull a model and run
ollama pull llama3.2
python3 bot.py
```

To enable RAG:

```bash
ollama pull nomic-embed-text
# Set rag.enabled: true in config.yaml
python ingest.py --collection docs /path/to/file.pdf
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally (`http://localhost:11434`)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)

## Setup

### 1. Clone and install dependencies

```bash
git clone git@github.com:wanleung/telegram-bot.git
cd telegram-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml`:

```yaml
telegram:
  token: "your-telegram-bot-token"   # or set TELEGRAM_TOKEN env var

ollama:
  base_url: "http://localhost:11434"
  default_model: "llama3.2"          # any model you have pulled
  timeout: 120
  think: false                       # set true to enable thinking mode by default

history:
  max_messages: 50
  db_path: "data/history.db"

mcp_servers:
  filesystem:
    type: stdio
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
    enabled: true

  # SSE-based MCP server
  web_search:
    type: sse
    url: "http://localhost:8080/sse"
    enabled: false

  # Streamable HTTP MCP server
  custom_http:
    type: http
    url: "http://localhost:9000/mcp"
    enabled: false
```

Environment variables are interpolated using `${VAR_NAME}` syntax anywhere in the config.

### 3. Pull an Ollama model

```bash
ollama pull llama3.2
```

### 4. Run

```bash
source venv/bin/activate
python3 bot.py
```

The bot will connect to all enabled MCP servers on startup and begin polling Telegram.

You can override the config file path:

```bash
CONFIG_PATH=/path/to/my-config.yaml python3 bot.py
```

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Show welcome message and current model |
| `/model <name>` | Switch to a different Ollama/vLLM model |
| `/models` | List all available models |
| `/tools` | List all available MCP tools |
| `/clear` | Clear conversation history for this chat |
| `/think` | Toggle thinking mode (shows model reasoning as a spoiler) |
| `/ingest <collection> <url-or-path>` | Ingest a document into RAG |
| `/collections` | List all RAG collections with chunk counts |

Any other text message is sent to the agent, which may use MCP tools before responding.

### Streaming

Responses are streamed token-by-token. The bot edits a placeholder message as tokens arrive, giving instant feedback. The final edit applies full Markdown→HTML formatting.

### Thinking mode

Type `/think` to toggle per-chat reasoning mode. When on, the model's chain-of-thought is shown as a collapsible spoiler before the answer. Requires a model that supports thinking (e.g. DeepSeek R1, QwQ).

The default can be set in `config.yaml` via `ollama.think: true` or `vllm.think: true`.

You can also **send a photo** (with optional caption) — the image is forwarded directly to the Ollama model. Switch to a vision-capable model first:

```
/model llama3.2-vision
```

Then send any photo. The caption becomes the prompt (defaults to empty if omitted).

## Markdown in Telegram

Telegram renders a limited subset of Markdown. The bot sends responses as plain text by default, so headers (`###`), bold (`**`), and tables from the model appear as raw characters.

To get cleaner output, instruct the model via a system prompt to avoid Markdown:

> "Respond in plain text without Markdown formatting. Use plain lists instead of tables."

If you enable `parse_mode=MarkdownV2` in `bot.py`, use Telegram-compatible syntax only:

| Formatting | Telegram syntax |
|---|---|
| Bold | `*bold*` |
| Italic | `_italic_` |
| Code | `` `code` `` |
| Pre | ` ```block``` ` |

Note: tables and `###` headings are not supported in Telegram Markdown.

## RAG (Retrieval-Augmented Generation)

Always-on retrieval: every user message triggers a search of the local ChromaDB knowledge base. Relevant chunks are injected as context into the Ollama prompt. Ideal for grounding responses in RFCs, documentation, or any local knowledge base.

### Prerequisites

```bash
ollama pull nomic-embed-text
```

### Configuration

Enable RAG in `config.yaml`:

```yaml
rag:
  enabled: true
  embed_model: nomic-embed-text
  db_path: data/chroma
  top_k: 5
  similarity_threshold: 0.4
```

**Field descriptions:**

- `enabled` — set to `true` to activate RAG (default: `false`)
- `embed_model` — Ollama embedding model (default: `nomic-embed-text`)
- `db_path` — ChromaDB persistence directory (default: `data/chroma`)
- `top_k` — number of chunks to retrieve per query (default: 5)
- `similarity_threshold` — minimum cosine similarity to include a chunk (default: 0.4)

### Ingesting documents (CLI)

**Single file or URL:**

```bash
python ingest.py --collection rfcs https://rfc-editor.org/rfc/rfc9110.txt
python ingest.py --collection docs /path/to/spec.pdf
python ingest.py --collection notes /path/to/notes.txt
```

**Batch from URL list file:**

```bash
# urls.txt: one URL per line
python ingest.py --collection rfcs --url-file urls.txt
```

**Recursive web crawl:**

```bash
python ingest.py --collection docs --crawl https://docs.example.com --depth 2
```

### Bot commands

- `/ingest <collection> <url-or-path>` — ingest a document from the bot
- `/collections` — list all RAG collections with chunk counts

### How it works

Each user message is embedded with `nomic-embed-text`, and the top-k most similar chunks are retrieved from ChromaDB across all collections. Results above the similarity threshold are prepended to the Ollama prompt as a `### Context` block. If RAG is disabled or unavailable, the bot falls back to standard operation.

## vLLM Backend

The bot supports [vLLM](https://github.com/vllm-project/vllm) as an alternative to Ollama for both chat and embeddings. vLLM must expose an OpenAI-compatible API (enabled by default).

### Quick switch

In `config.yaml`:

```yaml
backend: vllm

vllm:
  base_url: http://localhost:8000          # local vLLM
  # base_url: http://192.168.1.50:8000    # or any network address
  default_model: meta-llama/Llama-3.2-3B-Instruct
  timeout: 120
  think: false                             # set true for DeepSeek R1, QwQ, etc.
```

Set `ollama:` to `null` or remove it if you are using vLLM for both chat **and** embeddings (`rag.embed_backend: vllm`). If you keep Ollama for embeddings (the default), the `ollama:` block is still required.

### Embedding backend

By default, RAG embeddings use Ollama (`nomic-embed-text`). To use vLLM's embedding endpoint instead:

```yaml
rag:
  embed_backend: vllm
  embed_model: intfloat/e5-mistral-7b-instruct
```

vLLM must be serving an embedding-capable model. If the embed endpoint fails, RAG degrades gracefully (returns empty context).

### Starting vLLM

```bash
pip install vllm
vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8000
```

### Ollama vs vLLM

| Feature | Ollama | vLLM |
|---|---|---|
| API | Native Ollama | OpenAI-compatible |
| Text tool-call fallback | ✅ (`<\|python_start\|>`) | ❌ (not needed) |
| Embedding models | ✅ (`nomic-embed-text`) | ✅ (if model loaded) |
| Model management | `ollama pull <model>` | CLI at startup |

## MCP Server Types

| Type | Config field | Description |
|---|---|---|
| `stdio` | `command` | Subprocess (e.g. npx, python script) |
| `sse` | `url` | Server-Sent Events endpoint |
| `http` | `url` | Streamable HTTP endpoint |

All three types are supported simultaneously. Disabled servers (`enabled: false`) are skipped at startup.

## Project Structure

```
telegram-bot/
├── bot.py            # Telegram entry point, command handlers, streaming
├── agent.py          # Agentic loop: history → LLM → tools → reply (streaming)
├── mcp_manager.py    # MCP server connections and tool registry
├── config.py         # Pydantic config loader with env var resolution
├── history.py        # Async SQLite conversation history
├── llm_backend.py    # LLM backend abstraction (Ollama + vLLM)
├── rag.py            # RAG retrieval logic using ChromaDB
├── ingest.py         # CLI for ingesting documents into RAG
├── config.example.yaml
├── requirements.txt
└── tests/
    ├── test_agent.py
    ├── test_bot.py
    ├── test_config.py
    ├── test_history.py
    ├── test_llm_backend.py
    ├── test_mcp_manager.py
    ├── test_rag.py
    └── test_ingest.py
```

## Deployment

### Docker

```bash
# Build and start
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

`config.yaml` is mounted read-only into the container. The `bot-data` Docker volume persists both conversation history (`data/history.db`) and the RAG knowledge base (`data/chroma`) across container restarts.

> **Connecting to a local Ollama:** `docker-compose.yml` uses `network_mode: host` (Linux). On macOS/Windows, remove that line and set `ollama.base_url: "http://host.docker.internal:11434"` in `config.yaml`.

> **RAG with Docker:** Run `ingest.py` inside the container to populate the knowledge base into the `bot-data` volume:
> ```bash
> docker compose run --rm bot python ingest.py --collection rfcs https://rfc-editor.org/rfc/rfc9110.txt
> ```

---

### Daemon (systemd)

```bash
# 1. Copy files to /opt/telegram-bot
sudo cp -r . /opt/telegram-bot
sudo python3 -m venv /opt/telegram-bot/venv
sudo /opt/telegram-bot/venv/bin/pip install -r /opt/telegram-bot/requirements.txt

# 2. Create the environment file with your token
sudo tee /etc/telegram-bot.env <<EOF
TELEGRAM_TOKEN=your-token-here
EOF
sudo chmod 600 /etc/telegram-bot.env

# 3. Install and enable the service (replace 'youruser' with the user to run as)
sudo cp telegram-bot.service /etc/systemd/system/telegram-bot@youruser.service
sudo systemctl daemon-reload
sudo systemctl enable --now telegram-bot@youruser

# View logs
journalctl -u telegram-bot@youruser -f
```

The service restarts automatically on failure (`Restart=on-failure`).

---

## Running Tests

```bash
source venv/bin/activate
python3 -m pytest
```

## Configuration Reference

| Key | Default | Description |
|---|---|---|
| `telegram.token` | — | Bot token from BotFather |
| `ollama.base_url` | `http://localhost:11434` | Ollama API base URL |
| `ollama.default_model` | `llama3.2` | Model used on startup |
| `ollama.timeout` | `120` | Request timeout in seconds |
| `ollama.think` | `false` | Enable thinking mode by default for Ollama |
| `history.max_messages` | `50` | Max messages kept per chat |
| `history.db_path` | `data/history.db` | SQLite database path |
| `mcp_servers.<name>.type` | — | `stdio`, `sse`, or `http` |
| `mcp_servers.<name>.enabled` | `true` | Enable/disable server |
| `mcp_servers.<name>.command` | — | Command list (stdio only) |
| `mcp_servers.<name>.url` | — | Endpoint URL (sse/http only) |
| `backend` | `ollama` | Chat backend: `ollama` or `vllm` |
| `vllm.base_url` | `http://localhost:8000` | vLLM API endpoint |
| `vllm.default_model` | — | Model name served by vLLM |
| `vllm.timeout` | `120` | Request timeout in seconds |
| `vllm.think` | `false` | Enable thinking mode by default for vLLM |
| `rag.enabled` | `false` | Enable RAG retrieval |
| `rag.embed_backend` | `ollama` | Embedding backend: `ollama` or `vllm` |
| `rag.embed_model` | `nomic-embed-text` | Ollama embedding model |
| `rag.db_path` | `data/chroma` | ChromaDB persistence directory |
| `rag.top_k` | `5` | Number of chunks to retrieve per query |
| `rag.similarity_threshold` | `0.4` | Minimum cosine similarity for chunk inclusion |
