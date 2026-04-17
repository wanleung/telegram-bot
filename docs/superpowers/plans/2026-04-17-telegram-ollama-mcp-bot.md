# Telegram Ollama MCP Bot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-user Telegram bot that uses a local Ollama instance as the LLM backend, with MCP tool-calling support (stdio and SSE/HTTP servers) configured via `config.yaml`.

**Architecture:** Single async Python service. `bot.py` receives Telegram updates and delegates to `agent.py`, which runs the Ollama tool-call loop using tools from `mcp_manager.py`. Conversation history is persisted in SQLite via `history.py`. Config is validated at startup by `config.py`.

**Tech Stack:** Python 3.11+, `python-telegram-bot` v21, `ollama`, `mcp`, `aiosqlite`, `pydantic` v2, `pyyaml`, `pytest`, `pytest-asyncio`

---

## File Map

| File | Role |
|---|---|
| `config.py` | Pydantic v2 config model; loads + validates `config.yaml`; resolves `${ENV_VAR}` in strings |
| `config.yaml` | Runtime config (gitignored); token, Ollama URL, model, MCP servers |
| `config.example.yaml` | Committed example config with placeholder values |
| `history.py` | SQLite conversation history: init, get, save, clear, trim |
| `mcp_manager.py` | Starts MCP servers (stdio/SSE/HTTP), unified tool registry, tool dispatch |
| `agent.py` | Ollama tool-call loop: load history → get tools → chat → dispatch tool calls → reply |
| `bot.py` | Telegram bot entry point: handlers for messages and commands |
| `requirements.txt` | All dependencies |
| `pytest.ini` | pytest config (`asyncio_mode = auto`) |
| `data/` | SQLite DB directory — gitignored |
| `tests/test_config.py` | Config loading + env var tests |
| `tests/test_history.py` | History CRUD + trim tests |
| `tests/test_mcp_manager.py` | MCP manager unit tests (mocked sessions) |
| `tests/test_agent.py` | Agent loop unit tests (mocked Ollama + MCP) |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `.gitignore`
- Create: `config.example.yaml`
- Create: `data/.gitkeep`

- [ ] **Step 1: Create `requirements.txt`**

```
python-telegram-bot==21.*
ollama
mcp[cli]
aiosqlite
pydantic>=2.0
pyyaml
pytest
pytest-asyncio
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 3: Create `.gitignore`**

```
config.yaml
data/
__pycache__/
*.pyc
.env
.superpowers/
```

- [ ] **Step 4: Create `config.example.yaml`**

```yaml
telegram:
  token: "${TELEGRAM_TOKEN}"   # export TELEGRAM_TOKEN=your-token-here

ollama:
  base_url: "http://localhost:11434"
  default_model: "llama3.2"
  timeout: 120

history:
  max_messages: 50
  db_path: "data/history.db"

mcp_servers:
  filesystem:
    type: stdio
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
    enabled: true

  web_search:
    type: sse
    url: "http://localhost:8080/sse"
    enabled: false

  custom_http:
    type: http
    url: "http://localhost:9000/mcp"
    enabled: false
```

- [ ] **Step 5: Create `data/.gitkeep`**

```bash
mkdir -p data && touch data/.gitkeep
```

- [ ] **Step 6: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pytest.ini .gitignore config.example.yaml data/.gitkeep
git commit -m "chore: project scaffold

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: Config Module

**Files:**
- Create: `config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/__init__.py` (empty).

Create `tests/test_config.py`:

```python
import os
import pytest
from pydantic import ValidationError
from config import load_config


def write_cfg(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return str(p)


def test_load_valid_config(tmp_path):
    path = write_cfg(tmp_path, """
telegram:
  token: "test-token"
ollama:
  base_url: "http://localhost:11434"
  default_model: "llama3.2"
  timeout: 60
history:
  max_messages: 20
  db_path: "data/test.db"
mcp_servers: {}
""")
    cfg = load_config(path)
    assert cfg.telegram.token == "test-token"
    assert cfg.ollama.default_model == "llama3.2"
    assert cfg.ollama.timeout == 60
    assert cfg.history.max_messages == 20
    assert cfg.mcp_servers == {}


def test_missing_telegram_token_raises(tmp_path):
    path = write_cfg(tmp_path, """
telegram: {}
ollama:
  default_model: "llama3.2"
""")
    with pytest.raises(ValidationError):
        load_config(path)


def test_missing_ollama_model_raises(tmp_path):
    path = write_cfg(tmp_path, """
telegram:
  token: "tok"
ollama:
  base_url: "http://localhost:11434"
""")
    with pytest.raises(ValidationError):
        load_config(path)


def test_env_var_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "resolved-token")
    path = write_cfg(tmp_path, """
telegram:
  token: "${MY_TOKEN}"
ollama:
  default_model: "llama3.2"
""")
    cfg = load_config(path)
    assert cfg.telegram.token == "resolved-token"


def test_unset_env_var_left_as_is(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    path = write_cfg(tmp_path, """
telegram:
  token: "${MISSING_VAR}"
ollama:
  default_model: "llama3.2"
""")
    cfg = load_config(path)
    assert cfg.telegram.token == "${MISSING_VAR}"


def test_mcp_server_stdio_config(tmp_path):
    path = write_cfg(tmp_path, """
telegram:
  token: "tok"
ollama:
  default_model: "llama3.2"
mcp_servers:
  fs:
    type: stdio
    command: ["npx", "server-fs"]
    enabled: true
""")
    cfg = load_config(path)
    assert cfg.mcp_servers["fs"].type == "stdio"
    assert cfg.mcp_servers["fs"].command == ["npx", "server-fs"]
    assert cfg.mcp_servers["fs"].enabled is True


def test_mcp_server_disabled_by_default_is_false(tmp_path):
    path = write_cfg(tmp_path, """
telegram:
  token: "tok"
ollama:
  default_model: "llama3.2"
mcp_servers:
  remote:
    type: sse
    url: "http://localhost:8080/sse"
    enabled: false
""")
    cfg = load_config(path)
    assert cfg.mcp_servers["remote"].enabled is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: `ImportError` — `config` module not found.

- [ ] **Step 3: Implement `config.py`**

```python
import os
import re
import yaml
from pydantic import BaseModel


class TelegramConfig(BaseModel):
    token: str


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    default_model: str
    timeout: int = 120


class HistoryConfig(BaseModel):
    max_messages: int = 50
    db_path: str = "data/history.db"


class MCPServerConfig(BaseModel):
    type: str  # "stdio", "sse", "http"
    command: list[str] | None = None
    url: str | None = None
    enabled: bool = True


class Config(BaseModel):
    telegram: TelegramConfig
    ollama: OllamaConfig
    history: HistoryConfig = HistoryConfig()
    mcp_servers: dict[str, MCPServerConfig] = {}


def _resolve(obj: object) -> object:
    """Recursively resolve ${VAR} references in string values."""
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(i) for i in obj]
    return obj


def load_config(path: str = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(_resolve(raw))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add config.py tests/__init__.py tests/test_config.py
git commit -m "feat: config module with env var resolution

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: History Module

**Files:**
- Create: `history.py`
- Create: `tests/test_history.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_history.py`:

```python
import pytest
from history import init_db, get_history, save_messages, clear_history


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


async def test_empty_history(db_path):
    await init_db(db_path)
    result = await get_history(db_path, chat_id=1, max_messages=50)
    assert result == []


async def test_save_and_retrieve(db_path):
    await init_db(db_path)
    await save_messages(db_path, chat_id=1, messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ], max_messages=50)
    history = await get_history(db_path, chat_id=1, max_messages=50)
    assert history == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


async def test_trim_at_max_messages(db_path):
    await init_db(db_path)
    # Save 3 batches of 2 messages each (6 total), max is 4
    for i in range(3):
        await save_messages(db_path, chat_id=1, messages=[
            {"role": "user", "content": f"user {i}"},
            {"role": "assistant", "content": f"reply {i}"},
        ], max_messages=4)
    history = await get_history(db_path, chat_id=1, max_messages=4)
    assert len(history) == 4
    # Oldest messages trimmed, newest retained
    assert history[-1] == {"role": "assistant", "content": "reply 2"}


async def test_clear_history(db_path):
    await init_db(db_path)
    await save_messages(db_path, chat_id=1, messages=[
        {"role": "user", "content": "Hi"},
    ], max_messages=50)
    await clear_history(db_path, chat_id=1)
    result = await get_history(db_path, chat_id=1, max_messages=50)
    assert result == []


async def test_separate_chat_ids_isolated(db_path):
    await init_db(db_path)
    await save_messages(db_path, chat_id=1, messages=[
        {"role": "user", "content": "chat one"},
    ], max_messages=50)
    await save_messages(db_path, chat_id=2, messages=[
        {"role": "user", "content": "chat two"},
    ], max_messages=50)
    h1 = await get_history(db_path, chat_id=1, max_messages=50)
    h2 = await get_history(db_path, chat_id=2, max_messages=50)
    assert h1 == [{"role": "user", "content": "chat one"}]
    assert h2 == [{"role": "user", "content": "chat two"}]


async def test_clear_does_not_affect_other_chats(db_path):
    await init_db(db_path)
    await save_messages(db_path, chat_id=1, messages=[
        {"role": "user", "content": "keep me"},
    ], max_messages=50)
    await save_messages(db_path, chat_id=2, messages=[
        {"role": "user", "content": "delete me"},
    ], max_messages=50)
    await clear_history(db_path, chat_id=2)
    h1 = await get_history(db_path, chat_id=1, max_messages=50)
    h2 = await get_history(db_path, chat_id=2, max_messages=50)
    assert h1 == [{"role": "user", "content": "keep me"}]
    assert h2 == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_history.py -v
```

Expected: `ImportError` — `history` module not found.

- [ ] **Step 3: Implement `history.py`**

```python
import aiosqlite


async def init_db(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id   INTEGER NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_chat_id ON messages(chat_id)")
        await db.commit()


async def get_history(db_path: str, chat_id: int, max_messages: int) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, max_messages),
        ) as cursor:
            rows = await cursor.fetchall()
    # Reverse so oldest-first order is preserved for the LLM
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]


async def save_messages(
    db_path: str, chat_id: int, messages: list[dict], max_messages: int
) -> None:
    async with aiosqlite.connect(db_path) as db:
        for msg in messages:
            await db.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, msg["role"], msg["content"]),
            )
        # Trim: keep only the newest max_messages rows for this chat
        await db.execute(
            """
            DELETE FROM messages
            WHERE chat_id = ?
              AND id NOT IN (
                  SELECT id FROM messages
                  WHERE chat_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (chat_id, chat_id, max_messages),
        )
        await db.commit()


async def clear_history(db_path: str, chat_id: int) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        await db.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_history.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add history.py tests/test_history.py
git commit -m "feat: history module with SQLite persistence and trim

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: MCP Manager Module

**Files:**
- Create: `mcp_manager.py`
- Create: `tests/test_mcp_manager.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_mcp_manager.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_manager import MCPManager
from config import MCPServerConfig


def _make_tool(name: str, description: str, schema: dict):
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = schema
    return t


def _make_content(text: str):
    c = MagicMock()
    c.text = text
    return c


async def test_tool_registry_populated():
    servers = {
        "search": MCPServerConfig(type="stdio", command=["echo"], enabled=True)
    }
    mgr = MCPManager(servers)

    mock_session = AsyncMock()
    mock_tool = _make_tool("search_web", "Search the web", {"type": "object", "properties": {}})
    mock_session.list_tools.return_value = MagicMock(tools=[mock_tool])

    with patch.object(mgr, "_connect", return_value=mock_session):
        await mgr.start()

    defs = mgr.get_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "search_web"
    assert defs[0]["function"]["description"] == "Search the web"


async def test_call_tool_dispatches_correctly():
    servers = {
        "search": MCPServerConfig(type="stdio", command=["echo"], enabled=True)
    }
    mgr = MCPManager(servers)

    mock_session = AsyncMock()
    mock_tool = _make_tool("search_web", "Search", {"type": "object"})
    mock_session.list_tools.return_value = MagicMock(tools=[mock_tool])
    mock_session.call_tool.return_value = MagicMock(content=[_make_content("result text")])

    with patch.object(mgr, "_connect", return_value=mock_session):
        await mgr.start()

    result = await mgr.call_tool("search_web", {"query": "python"})
    assert result == "result text"
    mock_session.call_tool.assert_called_once_with("search_web", {"query": "python"})


async def test_unknown_tool_returns_error():
    mgr = MCPManager({})
    result = await mgr.call_tool("nonexistent", {})
    assert "unknown tool" in result.lower()


async def test_disabled_server_not_connected():
    servers = {
        "search": MCPServerConfig(type="stdio", command=["echo"], enabled=False)
    }
    mgr = MCPManager(servers)
    with patch.object(mgr, "_connect") as mock_connect:
        await mgr.start()
        mock_connect.assert_not_called()
    assert mgr.get_tool_definitions() == []


async def test_failed_server_is_skipped_gracefully():
    servers = {
        "broken": MCPServerConfig(type="stdio", command=["bad-cmd"], enabled=True)
    }
    mgr = MCPManager(servers)
    with patch.object(mgr, "_connect", side_effect=RuntimeError("connection failed")):
        await mgr.start()  # must not raise

    assert mgr.get_tool_definitions() == []


async def test_list_tools_summary_empty():
    mgr = MCPManager({})
    assert "No MCP tools" in mgr.list_tools_summary()


async def test_list_tools_summary_shows_tools():
    servers = {
        "fs": MCPServerConfig(type="stdio", command=["npx", "fs"], enabled=True)
    }
    mgr = MCPManager(servers)

    mock_session = AsyncMock()
    mock_tool = _make_tool("read_file", "Read a file", {"type": "object"})
    mock_session.list_tools.return_value = MagicMock(tools=[mock_tool])

    with patch.object(mgr, "_connect", return_value=mock_session):
        await mgr.start()

    summary = mgr.list_tools_summary()
    assert "read_file" in summary
    assert "Read a file" in summary
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_mcp_manager.py -v
```

Expected: `ImportError` — `mcp_manager` module not found.

- [ ] **Step 3: Implement `mcp_manager.py`**

```python
import logging
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPManager:
    def __init__(self, servers: dict[str, MCPServerConfig]) -> None:
        self._servers = {k: v for k, v in servers.items() if v.enabled}
        self._sessions: dict[str, ClientSession] = {}
        # tool_name -> (server_name, tool_object)
        self._tools: dict[str, tuple[str, object]] = {}
        self._exit_stack = AsyncExitStack()

    async def start(self) -> None:
        await self._exit_stack.__aenter__()
        for name, cfg in self._servers.items():
            try:
                session = await self._connect(name, cfg)
                self._sessions[name] = session
                result = await session.list_tools()
                for tool in result.tools:
                    self._tools[tool.name] = (name, tool)
                logger.info("MCP server '%s' connected with %d tool(s)", name, len(result.tools))
            except Exception as exc:
                logger.warning("MCP server '%s' failed to connect: %s", name, exc)

    async def _connect(self, name: str, cfg: MCPServerConfig) -> ClientSession:
        if cfg.type == "stdio":
            params = StdioServerParameters(command=cfg.command[0], args=cfg.command[1:])
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        elif cfg.type == "sse":
            read, write = await self._exit_stack.enter_async_context(sse_client(cfg.url))
        elif cfg.type == "http":
            read, write, _ = await self._exit_stack.enter_async_context(
                streamablehttp_client(cfg.url)
            )
        else:
            raise ValueError(f"Unknown MCP server type '{cfg.type}' for server '{name}'")

        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    def get_tool_definitions(self) -> list[dict]:
        """Return tools in the format Ollama's /api/chat expects."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for _, (_, tool) in self._tools.items()
        ]

    async def call_tool(self, name: str, arguments: dict) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        server_name, _ = self._tools[name]
        session = self._sessions[server_name]
        result = await session.call_tool(name, arguments)
        parts = [
            c.text if hasattr(c, "text") else str(c)
            for c in result.content
        ]
        return "\n".join(parts)

    def list_tools_summary(self) -> str:
        if not self._tools:
            return "No MCP tools available."
        lines = [
            f"• **{tool_name}** ({server_name}): {tool.description or '(no description)'}"
            for tool_name, (server_name, tool) in self._tools.items()
        ]
        return "\n".join(lines)

    async def stop(self) -> None:
        await self._exit_stack.__aexit__(None, None, None)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mcp_manager.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_manager.py tests/test_mcp_manager.py
git commit -m "feat: MCP manager with stdio/SSE/HTTP support

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: Agent Module

**Files:**
- Create: `agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agent.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent import Agent, MAX_ITERATIONS
from config import Config, TelegramConfig, OllamaConfig, HistoryConfig


def _make_config(tmp_path) -> Config:
    return Config(
        telegram=TelegramConfig(token="tok"),
        ollama=OllamaConfig(default_model="llama3.2"),
        history=HistoryConfig(max_messages=50, db_path=str(tmp_path / "h.db")),
    )


def _make_response(content: str | None = None, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    resp = MagicMock()
    resp.message = msg
    return resp


def _make_tool_call(name: str, arguments: dict):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    tc.model_dump.return_value = {"function": {"name": name, "arguments": arguments}}
    return tc


async def test_simple_text_response(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    agent = Agent(cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages") as mock_save, \
         patch.object(agent._client, "chat", return_value=_make_response(content="Hello!")):
        result = await agent.run(chat_id=1, user_message="Hi")

    assert result == "Hello!"
    mock_save.assert_called_once()
    saved_messages = mock_save.call_args[0][2]
    assert saved_messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


async def test_tool_call_then_text_response(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = AsyncMock()
    mcp.get_tool_definitions.return_value = [
        {"type": "function", "function": {"name": "search_web"}}
    ]
    mcp.call_tool.return_value = "Search result: Python docs"
    agent = Agent(cfg, mcp)

    tc = _make_tool_call("search_web", {"query": "python"})
    tool_response = _make_response(content="", tool_calls=[tc])
    final_response = _make_response(content="Here is what I found.")

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"), \
         patch.object(agent._client, "chat", side_effect=[tool_response, final_response]):
        result = await agent.run(chat_id=1, user_message="Search python")

    assert result == "Here is what I found."
    mcp.call_tool.assert_called_once_with("search_web", {"query": "python"})


async def test_max_iterations_guard(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = AsyncMock()
    mcp.get_tool_definitions.return_value = []
    mcp.call_tool.return_value = "result"
    agent = Agent(cfg, mcp)

    tc = _make_tool_call("loop_tool", {})
    always_tool = _make_response(content="", tool_calls=[tc])

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"), \
         patch.object(agent._client, "chat", return_value=always_tool):
        result = await agent.run(chat_id=1, user_message="loop")

    assert "maximum" in result.lower()


async def test_ollama_error_returns_error_message(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    agent = Agent(cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"), \
         patch.object(agent._client, "chat", side_effect=Exception("connection refused")):
        result = await agent.run(chat_id=1, user_message="Hi")

    assert "Ollama error" in result


async def test_history_passed_to_ollama(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    agent = Agent(cfg, mcp)

    existing_history = [
        {"role": "user", "content": "prev question"},
        {"role": "assistant", "content": "prev answer"},
    ]

    with patch("agent.get_history", return_value=existing_history), \
         patch("agent.save_messages"), \
         patch.object(agent._client, "chat", return_value=_make_response(content="ok")) as mock_chat:
        await agent.run(chat_id=1, user_message="new question")

    sent_messages = mock_chat.call_args[1]["messages"]
    assert sent_messages[0] == {"role": "user", "content": "prev question"}
    assert sent_messages[-1] == {"role": "user", "content": "new question"}


async def test_set_model_changes_active_model(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    agent = Agent(cfg, mcp)
    assert agent.active_model == "llama3.2"
    agent.set_model("mistral")
    assert agent.active_model == "mistral"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py -v
```

Expected: `ImportError` — `agent` module not found.

- [ ] **Step 3: Implement `agent.py`**

```python
import logging
import ollama

from config import Config
from history import get_history, save_messages
from mcp_manager import MCPManager

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10


class Agent:
    def __init__(self, config: Config, mcp_manager: MCPManager) -> None:
        self._cfg = config
        self._mcp = mcp_manager
        self._client = ollama.AsyncClient(host=config.ollama.base_url)
        self._active_model: str = config.ollama.default_model

    @property
    def active_model(self) -> str:
        return self._active_model

    def set_model(self, model: str) -> None:
        self._active_model = model

    async def run(self, chat_id: int, user_message: str) -> str:
        history = await get_history(
            self._cfg.history.db_path, chat_id, self._cfg.history.max_messages
        )
        tools = self._mcp.get_tool_definitions()
        messages = history + [{"role": "user", "content": user_message}]

        for _ in range(MAX_ITERATIONS):
            try:
                response = await self._client.chat(
                    model=self._active_model,
                    messages=messages,
                    tools=tools or None,
                )
            except Exception as exc:
                return f"⚠️ Ollama error: {exc}"

            msg = response.message

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                })
                for tc in msg.tool_calls:
                    result = await self._mcp.call_tool(
                        tc.function.name, dict(tc.function.arguments)
                    )
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "name": tc.function.name,
                    })
            else:
                reply = msg.content or ""
                await save_messages(
                    self._cfg.history.db_path,
                    chat_id,
                    [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": reply},
                    ],
                    self._cfg.history.max_messages,
                )
                return reply

        return "⚠️ Reached maximum tool call iterations. Please try again."
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: agent with Ollama tool-call loop and max-iteration guard

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 6: Telegram Bot Entry Point

**Files:**
- Create: `bot.py`

No unit tests for `bot.py` — it is a thin wiring layer tested manually. All logic lives in the already-tested modules.

- [ ] **Step 1: Implement `bot.py`**

```python
import asyncio
import logging
import os

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config import load_config
from history import init_db, clear_history
from mcp_manager import MCPManager
from agent import Agent

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    agent: Agent = context.bot_data["agent"]
    await update.message.reply_text(
        f"👋 Hello! I'm your Ollama-powered assistant.\n"
        f"Current model: `{agent.active_model}`\n\n"
        f"Commands:\n"
        f"  /model <name> — switch model\n"
        f"  /clear — clear conversation history\n"
        f"  /tools — list available MCP tools",
        parse_mode="Markdown",
    )


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    agent: Agent = context.bot_data["agent"]
    if not context.args:
        await update.message.reply_text(
            f"Current model: `{agent.active_model}`\nUsage: /model <name>",
            parse_mode="Markdown",
        )
        return
    new_model = context.args[0]
    agent.set_model(new_model)
    await update.message.reply_text(f"✅ Switched to model `{new_model}`", parse_mode="Markdown")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg = context.bot_data["config"]
    await clear_history(cfg.history.db_path, update.effective_chat.id)
    await update.message.reply_text("🗑️ Conversation history cleared.")


async def cmd_tools(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mcp: MCPManager = context.bot_data["mcp"]
    await update.message.reply_text(mcp.list_tools_summary(), parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    agent: Agent = context.bot_data["agent"]
    thinking = await update.message.reply_text("⏳ Thinking…")
    reply = await agent.run(
        chat_id=update.effective_chat.id,
        user_message=update.message.text,
    )
    await thinking.edit_text(reply)


async def main() -> None:
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    cfg = load_config(config_path)

    os.makedirs(os.path.dirname(cfg.history.db_path) or ".", exist_ok=True)
    await init_db(cfg.history.db_path)

    mcp = MCPManager(cfg.mcp_servers)
    await mcp.start()

    agent = Agent(cfg, mcp)

    app = Application.builder().token(cfg.telegram.token).build()
    app.bot_data["config"] = cfg
    app.bot_data["agent"] = agent
    app.bot_data["mcp"] = mcp

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("tools", cmd_tools))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting with model '%s'", agent.active_model)
    try:
        await app.run_polling(drop_pending_updates=True)
    finally:
        await mcp.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run full test suite to confirm nothing broken**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add bot.py
git commit -m "feat: Telegram bot entry point with command handlers

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 7: Wire Up and Run

**Files:**
- Create: `config.yaml` (from example; not committed)

- [ ] **Step 1: Copy example config**

```bash
cp config.example.yaml config.yaml
```

- [ ] **Step 2: Fill in your config**

Edit `config.yaml`:
- Set `telegram.token` to your bot token from [@BotFather](https://t.me/BotFather), **or** export it:
  ```bash
  export TELEGRAM_TOKEN="your-token-here"
  ```
- Set `ollama.default_model` to a model you have pulled (e.g. `llama3.2`, `qwen2.5-coder`).
- Verify Ollama is running:
  ```bash
  curl http://localhost:11434/api/tags
  ```
- Enable/disable MCP servers as needed in `mcp_servers`.

- [ ] **Step 3: Verify config loads cleanly**

```python
python -c "from config import load_config; cfg = load_config(); print('OK:', cfg.ollama.default_model)"
```

Expected: `OK: <your-model-name>`

- [ ] **Step 4: Start the bot**

```bash
python bot.py
```

Expected log output:
```
INFO agent: Bot starting with model 'llama3.2'
INFO httpx: HTTP Request: POST https://api.telegram.org/...
```

- [ ] **Step 5: Smoke test in Telegram**

1. Open your bot in Telegram and send `/start` — should reply with greeting and model name.
2. Send a plain message — should reply with `⏳ Thinking…` then the LLM response.
3. Send `/tools` — should list available MCP tools (or "No MCP tools available").
4. Send `/model mistral` — should confirm model switch.
5. Send `/clear` — should confirm history cleared.

- [ ] **Step 6: Final commit**

```bash
git add .
git commit -m "chore: add .gitignore and finalize project

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
