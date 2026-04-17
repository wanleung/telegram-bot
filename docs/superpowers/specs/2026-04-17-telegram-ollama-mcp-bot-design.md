# Telegram Bot with Ollama + MCP — Design Spec

**Date:** 2026-04-17  
**Status:** Approved

---

## Overview

A single-user Telegram bot that uses a locally running Ollama instance as its LLM backend. The bot supports the Model Context Protocol (MCP) so Ollama can invoke external tools (filesystem, web search, code execution, custom tools). MCP servers are configured via a YAML config file and support both `stdio` (local subprocess) and `sse`/`http` (remote HTTP) transports.

---

## Architecture

Single Python service (no microservices). All components are async Python modules.

### Modules

| Module | Responsibility |
|---|---|
| `bot.py` | Telegram entry point. Handles updates, routes messages to the agent, exposes bot commands. |
| `agent.py` | Ollama tool-call agent loop. Loads conversation history, retrieves MCP tool definitions, calls Ollama, dispatches tool calls, loops until a final text response is produced. |
| `mcp_manager.py` | Manages MCP server connections. Spawns stdio processes and connects to SSE/HTTP servers on startup. Exposes a unified async tool registry. |
| `history.py` | SQLite-backed conversation history (via `aiosqlite`). Per-chat storage, sliding window trim, async read/write. |
| `config.py` | Pydantic v2 model for `config.yaml`. Validates all fields at startup; resolves `${ENV_VAR}` references in string fields. |

### External Dependencies

| Dependency | Purpose |
|---|---|
| `python-telegram-bot` v21+ | Async Telegram bot framework |
| `ollama` | Ollama Python client (wraps `/api/chat`) |
| `mcp` | Official MCP Python SDK (stdio + SSE client) |
| `aiosqlite` | Async SQLite driver |
| `pydantic` v2 | Config validation |
| `pyyaml` | Config file parsing |

---

## Data Flow

```
User message (Telegram)
  → bot.py receives update
  → agent.py
      1. Load conversation history from SQLite (history.py)
      2. Fetch available tools from MCP Manager (mcp_manager.py)
      3. POST /api/chat to Ollama with messages + tools
      4. If response contains tool_call:
           a. Call the appropriate MCP server tool
           b. Append tool result to message context
           c. Go back to step 3
         (max 10 iterations to prevent runaway loops)
      5. Return final text response
  → bot.py sends reply to Telegram
  → history.py saves user message + assistant reply to SQLite
```

---

## Configuration

Single `config.yaml` at the project root. Secrets can be provided as `${ENV_VAR}` references (resolved at runtime, never stored in plain text if using env vars).

```yaml
telegram:
  token: "${TELEGRAM_TOKEN}"

ollama:
  base_url: "http://localhost:11434"
  default_model: "llama3.2"
  timeout: 120                    # seconds

history:
  max_messages: 50                # sliding window; older messages trimmed
  db_path: "data/history.db"

mcp_servers:
  filesystem:
    type: stdio
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
    enabled: true

  web_search:
    type: sse
    url: "http://localhost:8080/sse"
    enabled: true

  code_exec:
    type: stdio
    command: ["python", "mcp_servers/code_exec.py"]
    enabled: true

  my_custom_tool:
    type: http
    url: "http://localhost:9000/mcp"
    enabled: false
```

**MCP server types:**
- `stdio` — bot spawns the process; `command` is the argv list
- `sse` — bot connects to a remote SSE endpoint
- `http` — bot connects to a remote Streamable HTTP endpoint

---

## Bot Commands

| Command | Behaviour |
|---|---|
| `/start` | Greeting message showing current model name |
| `/model <name>` | Switch active Ollama model in memory; reverts to `default_model` on bot restart |
| `/clear` | Delete conversation history for this chat |
| `/tools` | List all enabled MCP tools and their descriptions |

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Ollama unreachable at startup | Log error, bot starts but replies with error message on each chat |
| Ollama request timeout | Reply with timeout error message; history not saved |
| MCP server fails to start/connect | Log warning, server skipped; bot continues without that server's tools |
| MCP tool call returns error | Error string appended as tool result; Ollama continues reasoning |
| Tool-call loop exceeds 10 iterations | Loop aborted; partial response returned with a note |
| Telegram API errors | Handled by `python-telegram-bot` built-in retry/backoff |
| Invalid config.yaml | Startup fails immediately with a clear Pydantic validation error |

---

## Persistence

- **Database:** SQLite at `history.db_path` (default: `data/history.db`)
- **Schema:** `messages(id, chat_id, role, content, timestamp)` — `role` is `user` or `assistant` only; tool call intermediates are not persisted
- **Trim:** When message count for a chat exceeds `max_messages`, oldest messages are deleted
- **Gitignore:** `data/` directory is gitignored

---

## Project Layout

```
telegram-bot/
├── bot.py                  # Telegram bot entry point
├── agent.py                # Ollama agent loop
├── mcp_manager.py          # MCP server manager
├── history.py              # Conversation history (SQLite)
├── config.py               # Config model (Pydantic)
├── config.yaml             # User configuration
├── config.example.yaml     # Example config (committed to git)
├── requirements.txt
├── data/                   # SQLite DB — gitignored
├── tests/
│   ├── test_agent.py
│   ├── test_history.py
│   ├── test_config.py
│   └── test_mcp_manager.py
├── docs/
│   └── superpowers/specs/
│       └── 2026-04-17-telegram-ollama-mcp-bot-design.md
└── .gitignore
```

---

## Testing

- **Framework:** `pytest` + `pytest-asyncio`
- **Mocking:** Ollama client and MCP SDK clients are mocked; no live services required for unit tests
- **Coverage areas:**
  - Agent loop: single response, tool call → response, max-iteration guard
  - History: save, retrieve, trim at max_messages boundary
  - Config: valid config, missing required fields, env var resolution
  - MCP Manager: tool registry population, unknown tool dispatch error
