# Telegram Ollama Bot

A single-user Telegram bot powered by a local [Ollama](https://ollama.com) instance with [MCP](https://modelcontextprotocol.io) (Model Context Protocol) tool-calling support.

## Features

- 🤖 Chat with any locally running Ollama model
- 🔧 MCP tool-calling — connect stdio, SSE, and HTTP MCP servers
- 💾 Persistent per-chat conversation history (SQLite)
- 🔄 Switch models at runtime via `/model`
- ⚙️ YAML config with environment variable interpolation

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
| `/model <name>` | Switch to a different Ollama model |
| `/tools` | List all available MCP tools |
| `/clear` | Clear conversation history for this chat |

Any other text message is sent to the Ollama agent, which may use MCP tools before responding.

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
├── bot.py            # Telegram entry point and command handlers
├── agent.py          # Agentic loop: history → Ollama → tools → reply
├── mcp_manager.py    # MCP server connections and tool registry
├── config.py         # Pydantic config loader with env var resolution
├── history.py        # Async SQLite conversation history
├── config.example.yaml
├── requirements.txt
└── tests/
    ├── test_agent.py
    ├── test_config.py
    ├── test_history.py
    └── test_mcp_manager.py
```

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
| `history.max_messages` | `50` | Max messages kept per chat |
| `history.db_path` | `data/history.db` | SQLite database path |
| `mcp_servers.<name>.type` | — | `stdio`, `sse`, or `http` |
| `mcp_servers.<name>.enabled` | `true` | Enable/disable server |
| `mcp_servers.<name>.command` | — | Command list (stdio only) |
| `mcp_servers.<name>.url` | — | Endpoint URL (sse/http only) |
