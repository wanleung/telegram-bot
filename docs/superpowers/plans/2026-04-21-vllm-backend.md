# vLLM Backend Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `LLMBackend` abstraction so the bot can use either a local Ollama instance or a local/network vLLM instance, selected via `config.yaml`.

**Architecture:** A new `llm_backend.py` defines a `LLMBackend` protocol with `chat()`, `list_models()`, `embed()`, and `format_tool_result()` methods. `OllamaBackend` wraps the existing `ollama` SDK; `VLLMBackend` wraps the `openai` SDK pointed at vLLM's OpenAI-compatible endpoint. `Agent` and `RagManager` accept a backend instance rather than creating their own clients.

**Tech Stack:** `ollama` SDK (existing), `openai>=1.0` (new), `chromadb` (existing), `pydantic v2`, `pytest-asyncio`.

---

## File Map

| File | Action | Change |
|---|---|---|
| `requirements.txt` | Modify | Add `openai>=1.0` |
| `config.py` | Modify | Add `VLLMConfig`; make `ollama` optional; add `backend` field; add cross-field validators; add `embed_backend` to `RagConfig` |
| `config.example.yaml` | Modify | Add `backend`, `vllm:` block, `embed_backend` |
| `llm_backend.py` | **Create** | `ToolCall`, `ChatResponse` dataclasses; `LLMBackend` Protocol; `OllamaBackend`; `VLLMBackend`; `create_backend()`; `create_embed_backend()` |
| `agent.py` | Modify | Accept `LLMBackend` + `initial_model`; remove ollama SDK usage; remove `_parse_text_tool_calls` (moved to `llm_backend.py`) |
| `rag.py` | Modify | Accept `LLMBackend` for embeddings; remove ollama SDK usage |
| `bot.py` | Modify | Wire factory functions; update `Agent` and `RagManager` construction |
| `tests/test_config_vllm.py` | **Create** | New config validator tests |
| `tests/test_llm_backend.py` | **Create** | Unit tests for both backends + factories |
| `tests/test_agent.py` | Modify | Update to use mock `LLMBackend` instead of patching `_client` |
| `tests/test_rag.py` | Modify | Pass mock embed backend to `RagManager` constructor |
| `README.md` | Modify | Add vLLM config section |

---

### Task 1: Config — add VLLMConfig, backend selector, embed_backend

**Files:**
- Modify: `config.py`
- Modify: `config.example.yaml`
- Modify: `requirements.txt`
- Create: `tests/test_config_vllm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_config_vllm.py`:

```python
import pytest
from pydantic import ValidationError
from config import load_config, Config, TelegramConfig, OllamaConfig, VLLMConfig, RagConfig


def _write_cfg(tmp_path, content: str):
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return str(p)


def test_vllm_config_parsed(tmp_path):
    path = _write_cfg(tmp_path,
        "telegram:\n  token: tok\n"
        "backend: vllm\n"
        "vllm:\n  base_url: http://192.168.1.50:8000\n  default_model: llama3\n"
    )
    cfg = load_config(path)
    assert cfg.backend == "vllm"
    assert cfg.vllm.base_url == "http://192.168.1.50:8000"
    assert cfg.vllm.default_model == "llama3"
    assert cfg.vllm.timeout == 120


def test_backend_ollama_requires_ollama_block(tmp_path):
    path = _write_cfg(tmp_path,
        "telegram:\n  token: tok\n"
        "backend: ollama\n"
    )
    with pytest.raises((ValidationError, ValueError)):
        load_config(path)


def test_backend_vllm_requires_vllm_block(tmp_path):
    path = _write_cfg(tmp_path,
        "telegram:\n  token: tok\n"
        "backend: vllm\n"
        "ollama:\n  default_model: llama3.2\n"
    )
    with pytest.raises((ValidationError, ValueError)):
        load_config(path)


def test_embed_backend_vllm_requires_vllm_block(tmp_path):
    path = _write_cfg(tmp_path,
        "telegram:\n  token: tok\n"
        "backend: ollama\n"
        "ollama:\n  default_model: llama3.2\n"
        "rag:\n  embed_backend: vllm\n"
    )
    with pytest.raises((ValidationError, ValueError)):
        load_config(path)


def test_full_vllm_config(tmp_path):
    path = _write_cfg(tmp_path,
        "telegram:\n  token: tok\n"
        "backend: vllm\n"
        "vllm:\n  base_url: http://localhost:8000\n  default_model: llama3\n"
        "rag:\n  embed_backend: vllm\n"
    )
    cfg = load_config(path)
    assert cfg.backend == "vllm"
    assert cfg.rag.embed_backend == "vllm"


def test_embed_backend_defaults_to_ollama(tmp_path):
    path = _write_cfg(tmp_path,
        "telegram:\n  token: tok\n"
        "ollama:\n  default_model: llama3.2\n"
    )
    cfg = load_config(path)
    assert cfg.rag.embed_backend == "ollama"


def test_vllm_config_default_timeout():
    cfg = VLLMConfig(base_url="http://localhost:8000", default_model="llama3")
    assert cfg.timeout == 120
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /home/wanleung/Projects/telegram-bot && source venv/bin/activate
python -m pytest tests/test_config_vllm.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'VLLMConfig'` or similar.

- [ ] **Step 3: Implement config changes**

Edit `config.py` — replace the entire file:

```python
import os
import re
import yaml
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class TelegramConfig(BaseModel):
    token: str


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    default_model: str
    timeout: int = 120


class VLLMConfig(BaseModel):
    base_url: str = "http://localhost:8000"
    default_model: str
    timeout: int = 120


class HistoryConfig(BaseModel):
    max_messages: int = 50
    db_path: str = "data/history.db"


class RagConfig(BaseModel):
    enabled: bool = False
    embed_backend: Literal["ollama", "vllm"] = "ollama"
    embed_model: str = Field(default="nomic-embed-text", min_length=1)
    db_path: str = Field(default="data/chroma", min_length=1)
    top_k: int = Field(default=4, gt=0)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class MCPServerConfig(BaseModel):
    type: Literal["stdio", "sse", "http"]
    command: list[str] | None = None
    url: str | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def check_fields_for_type(self) -> "MCPServerConfig":
        if self.type == "stdio" and not self.command:
            raise ValueError("MCP server type 'stdio' requires 'command'")
        if self.type in ("sse", "http") and not self.url:
            raise ValueError(f"MCP server type '{self.type}' requires 'url'")
        return self


class Config(BaseModel):
    telegram: TelegramConfig
    backend: Literal["ollama", "vllm"] = "ollama"
    ollama: OllamaConfig | None = None
    vllm: VLLMConfig | None = None
    history: HistoryConfig = HistoryConfig()
    rag: RagConfig = RagConfig()
    mcp_servers: dict[str, MCPServerConfig] = {}

    @model_validator(mode="after")
    def check_backend_config(self) -> "Config":
        if self.backend == "ollama" and self.ollama is None:
            raise ValueError("backend is 'ollama' but no ollama: block found in config")
        if self.backend == "vllm" and self.vllm is None:
            raise ValueError("backend is 'vllm' but no vllm: block found in config")
        if self.rag.embed_backend == "ollama" and self.ollama is None:
            raise ValueError("rag.embed_backend is 'ollama' but no ollama: block found in config")
        if self.rag.embed_backend == "vllm" and self.vllm is None:
            raise ValueError("rag.embed_backend is 'vllm' but no vllm: block found in config")
        return self


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

- [ ] **Step 4: Add openai to requirements.txt**

Add the line `openai>=1.0` after `ollama`:

```
python-telegram-bot==21.*
ollama
openai>=1.0
mcp[cli]
aiosqlite
pydantic>=2.0
pyyaml
markdown
chromadb
pypdf
beautifulsoup4
httpx
pytest
pytest-asyncio
```

Run `pip install openai` in the venv:

```bash
source venv/bin/activate && pip install "openai>=1.0" -q
```

- [ ] **Step 5: Update config.example.yaml**

Replace the `ollama:` block and add `backend` + `vllm:` section:

```yaml
telegram:
  token: "${TELEGRAM_TOKEN}"   # export TELEGRAM_TOKEN=your-token-here

# Chat backend: ollama (default) or vllm
backend: ollama

ollama:
  base_url: "http://localhost:11434"
  default_model: "llama3.2"
  timeout: 120

# vLLM backend (local or network). Required when backend: vllm or rag.embed_backend: vllm
# vllm:
#   base_url: "http://localhost:8000"      # or http://192.168.1.50:8000
#   default_model: "meta-llama/Llama-3.2-3B-Instruct"
#   timeout: 120

history:
  max_messages: 50
  db_path: "data/history.db"

rag:
  enabled: false
  embed_backend: ollama   # or vllm
  embed_model: "nomic-embed-text"
  db_path: "data/chroma"
  top_k: 4
  similarity_threshold: 0.5

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

- [ ] **Step 6: Run tests — expect new tests pass, existing suite intact**

```bash
python -m pytest tests/test_config_vllm.py tests/test_config_rag.py tests/test_config.py -v
```

Expected: all pass. (Existing config tests all provide `ollama:` block so the new validator doesn't reject them.)

- [ ] **Step 7: Commit**

```bash
git add config.py config.example.yaml requirements.txt tests/test_config_vllm.py
git commit -m "feat: add VLLMConfig and backend selector to config

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Create llm_backend.py

**Files:**
- Create: `llm_backend.py`
- Create: `tests/test_llm_backend.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_llm_backend.py`:

```python
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from config import Config, TelegramConfig, OllamaConfig, VLLMConfig, RagConfig
from llm_backend import (
    ChatResponse, ToolCall, OllamaBackend, VLLMBackend,
    create_backend, create_embed_backend, _parse_text_tool_calls,
)


def _ollama_cfg():
    return OllamaConfig(base_url="http://localhost:11434", default_model="llama3.2")


def _vllm_cfg():
    return VLLMConfig(base_url="http://localhost:8000", default_model="llama3")


def _make_config(backend="ollama", embed_backend="ollama"):
    return Config(
        telegram=TelegramConfig(token="tok"),
        backend=backend,
        ollama=_ollama_cfg() if (backend == "ollama" or embed_backend == "ollama") else None,
        vllm=_vllm_cfg() if (backend == "vllm" or embed_backend == "vllm") else None,
        rag=RagConfig(embed_backend=embed_backend),
    )


# --- _parse_text_tool_calls ---

def test_parse_text_tool_calls_python_tags():
    content = '<|python_start|>{"type": "function", "name": "get_status", "parameters": {}}<|python_end|>'
    calls = _parse_text_tool_calls(content)
    assert calls == [{"name": "get_status", "arguments": {}}]


def test_parse_text_tool_calls_with_arguments():
    content = '<|python_start|>{"name": "search", "parameters": {"query": "python"}}<|python_end|>'
    calls = _parse_text_tool_calls(content)
    assert calls == [{"name": "search", "arguments": {"query": "python"}}]


def test_parse_text_tool_calls_no_match():
    assert _parse_text_tool_calls("Hello, how can I help?") is None


def test_parse_text_tool_calls_multiple():
    content = (
        '<|python_start|>{"name": "tool_a", "parameters": {}}<|python_end|>'
        '<|python_start|>{"name": "tool_b", "parameters": {"x": 1}}<|python_end|>'
    )
    calls = _parse_text_tool_calls(content)
    assert len(calls) == 2
    assert calls[0]["name"] == "tool_a"
    assert calls[1] == {"name": "tool_b", "arguments": {"x": 1}}


# --- OllamaBackend ---

@pytest.mark.asyncio
async def test_ollama_backend_chat_plain_text():
    backend = OllamaBackend(_ollama_cfg())
    msg = MagicMock()
    msg.content = "Hello!"
    msg.tool_calls = []
    resp = MagicMock()
    resp.message = msg
    with patch.object(backend._client, "chat", return_value=resp):
        result = await backend.chat("llama3.2", [{"role": "user", "content": "hi"}], None)
    assert isinstance(result, ChatResponse)
    assert result.content == "Hello!"
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_ollama_backend_chat_structured_tool_calls():
    backend = OllamaBackend(_ollama_cfg())
    tc = MagicMock()
    tc.function.name = "search"
    tc.function.arguments = {"query": "python"}
    tc.model_dump.return_value = {"function": {"name": "search", "arguments": {"query": "python"}}}
    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tc]
    resp = MagicMock()
    resp.message = msg
    with patch.object(backend._client, "chat", return_value=resp):
        result = await backend.chat("llama3.2", [], [])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "python"}
    assert result.raw_assistant_message["role"] == "assistant"
    assert "tool_calls" in result.raw_assistant_message


@pytest.mark.asyncio
async def test_ollama_backend_chat_text_embedded_fallback():
    backend = OllamaBackend(_ollama_cfg())
    msg = MagicMock()
    msg.content = '<|python_start|>{"name": "get_status", "parameters": {}}<|python_end|>'
    msg.tool_calls = []
    resp = MagicMock()
    resp.message = msg
    with patch.object(backend._client, "chat", return_value=resp):
        result = await backend.chat("llama3.2", [], None)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_status"
    assert result.content == ""  # stripped


@pytest.mark.asyncio
async def test_ollama_backend_list_models():
    backend = OllamaBackend(_ollama_cfg())
    m1, m2 = MagicMock(), MagicMock()
    m1.model = "llava"
    m2.model = "llama3.2"
    resp = MagicMock()
    resp.models = [m1, m2]
    with patch.object(backend._client, "list", return_value=resp):
        result = await backend.list_models()
    assert result == ["llama3.2", "llava"]


@pytest.mark.asyncio
async def test_ollama_backend_list_models_error():
    backend = OllamaBackend(_ollama_cfg())
    with patch.object(backend._client, "list", side_effect=Exception("conn refused")):
        result = await backend.list_models()
    assert result == []


@pytest.mark.asyncio
async def test_ollama_backend_embed():
    backend = OllamaBackend(_ollama_cfg())
    resp = MagicMock()
    resp.embeddings = [[0.1, 0.2, 0.3]]
    with patch.object(backend._client, "embed", return_value=resp):
        result = await backend.embed("nomic-embed-text", "hello")
    assert result == [0.1, 0.2, 0.3]


def test_ollama_format_tool_result():
    backend = OllamaBackend(_ollama_cfg())
    tc = ToolCall(name="search", arguments={})
    msg = backend.format_tool_result(tc, "result text")
    assert msg == {"role": "tool", "content": "result text", "tool_name": "search"}


# --- VLLMBackend ---

@pytest.mark.asyncio
async def test_vllm_backend_chat_plain_text():
    backend = VLLMBackend(_vllm_cfg())
    choice = MagicMock()
    choice.message.content = "Hello from vLLM!"
    choice.message.tool_calls = None
    resp = MagicMock()
    resp.choices = [choice]
    with patch.object(backend._client.chat.completions, "create", return_value=resp):
        result = await backend.chat("llama3", [{"role": "user", "content": "hi"}], None)
    assert result.content == "Hello from vLLM!"
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_vllm_backend_chat_tool_calls():
    backend = VLLMBackend(_vllm_cfg())
    tc = MagicMock()
    tc.id = "call_abc"
    tc.function.name = "search"
    tc.function.arguments = json.dumps({"query": "python"})
    choice = MagicMock()
    choice.message.content = ""
    choice.message.tool_calls = [tc]
    resp = MagicMock()
    resp.choices = [choice]
    with patch.object(backend._client.chat.completions, "create", return_value=resp):
        result = await backend.chat("llama3", [], [])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "python"}
    assert result.tool_calls[0].id == "call_abc"


@pytest.mark.asyncio
async def test_vllm_backend_list_models():
    backend = VLLMBackend(_vllm_cfg())
    m1, m2 = MagicMock(), MagicMock()
    m1.id = "llama3"
    m2.id = "mistral"
    resp = MagicMock()
    resp.data = [m2, m1]
    with patch.object(backend._client.models, "list", return_value=resp):
        result = await backend.list_models()
    assert result == ["llama3", "mistral"]


@pytest.mark.asyncio
async def test_vllm_backend_list_models_error():
    backend = VLLMBackend(_vllm_cfg())
    with patch.object(backend._client.models, "list", side_effect=Exception("conn refused")):
        result = await backend.list_models()
    assert result == []


@pytest.mark.asyncio
async def test_vllm_backend_embed():
    backend = VLLMBackend(_vllm_cfg())
    emb = MagicMock()
    emb.embedding = [0.4, 0.5, 0.6]
    resp = MagicMock()
    resp.data = [emb]
    with patch.object(backend._client.embeddings, "create", return_value=resp):
        result = await backend.embed("e5-mistral", "hello")
    assert result == [0.4, 0.5, 0.6]


def test_vllm_format_tool_result():
    backend = VLLMBackend(_vllm_cfg())
    tc = ToolCall(name="search", arguments={}, id="call_xyz")
    msg = backend.format_tool_result(tc, "result text")
    assert msg == {"role": "tool", "tool_call_id": "call_xyz", "content": "result text"}


def test_vllm_format_tool_result_no_id():
    backend = VLLMBackend(_vllm_cfg())
    tc = ToolCall(name="search", arguments={}, id=None)
    msg = backend.format_tool_result(tc, "result")
    assert msg["tool_call_id"] == ""


# --- Factories ---

def test_create_backend_ollama():
    cfg = _make_config(backend="ollama")
    backend = create_backend(cfg)
    assert isinstance(backend, OllamaBackend)


def test_create_backend_vllm():
    cfg = _make_config(backend="vllm", embed_backend="vllm")
    backend = create_backend(cfg)
    assert isinstance(backend, VLLMBackend)


def test_create_embed_backend_ollama():
    cfg = _make_config(embed_backend="ollama")
    backend = create_embed_backend(cfg)
    assert isinstance(backend, OllamaBackend)


def test_create_embed_backend_vllm():
    cfg = _make_config(backend="vllm", embed_backend="vllm")
    backend = create_embed_backend(cfg)
    assert isinstance(backend, VLLMBackend)
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
python -m pytest tests/test_llm_backend.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'llm_backend'`

- [ ] **Step 3: Create llm_backend.py**

Create `llm_backend.py` in the project root:

```python
"""LLM backend abstraction: Ollama and vLLM (OpenAI-compat) implementations."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import ollama
import openai

from config import Config, OllamaConfig, VLLMConfig

logger = logging.getLogger(__name__)

_TEXT_TOOL_CALL_RE = re.compile(
    r"<\|python_start\|>(.*?)<\|python_end\|>"
    r"|```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)


def _parse_text_tool_calls(content: str) -> list[dict] | None:
    """
    Extract tool call dicts from a model's plain-text tool-call format.

    Returns a list of {"name": str, "arguments": dict} dicts, or None if
    no tool-call markers were found in content.
    """
    matches = _TEXT_TOOL_CALL_RE.findall(content)
    if not matches:
        return None
    calls = []
    for python_block, json_block in matches:
        raw = (python_block or json_block).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Could not parse text tool call: %r", raw)
            continue
        name = data.get("name") or data.get("function", {}).get("name")
        arguments = data.get("parameters") or data.get("arguments") or {}
        if name:
            calls.append({"name": name, "arguments": arguments})
    return calls or None


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str | None = None  # tool_call_id for OpenAI/vLLM response correlation


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_assistant_message: dict = field(default_factory=dict)


@runtime_checkable
class LLMBackend(Protocol):
    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None
    ) -> ChatResponse: ...

    async def list_models(self) -> list[str]: ...

    async def embed(self, model: str, text: str) -> list[float]: ...

    def format_tool_result(self, tool_call: ToolCall, result: str) -> dict: ...


class OllamaBackend:
    """LLM backend backed by the Ollama native API."""

    def __init__(self, cfg: OllamaConfig) -> None:
        self._client = ollama.AsyncClient(host=cfg.base_url, timeout=cfg.timeout)

    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None
    ) -> ChatResponse:
        response = await self._client.chat(model=model, messages=messages, tools=tools)
        msg = response.message
        content = msg.content or ""

        if msg.tool_calls:
            tool_calls = [
                ToolCall(name=tc.function.name, arguments=dict(tc.function.arguments))
                for tc in msg.tool_calls
            ]
            raw_msg = {
                "role": "assistant",
                "content": content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
            return ChatResponse(content=content, tool_calls=tool_calls, raw_assistant_message=raw_msg)

        text_calls = _parse_text_tool_calls(content)
        if text_calls:
            tool_calls = [
                ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in text_calls
            ]
            clean_content = _TEXT_TOOL_CALL_RE.sub("", content).strip()
            raw_msg = {"role": "assistant", "content": clean_content}
            return ChatResponse(
                content=clean_content, tool_calls=tool_calls, raw_assistant_message=raw_msg
            )

        raw_msg = {"role": "assistant", "content": content}
        return ChatResponse(content=content, raw_assistant_message=raw_msg)

    async def list_models(self) -> list[str]:
        try:
            response = await self._client.list()
            return sorted(m.model for m in response.models)
        except Exception as exc:
            logger.exception("Failed to list Ollama models: %s", exc)
            return []

    async def embed(self, model: str, text: str) -> list[float]:
        response = await self._client.embed(model=model, input=text)
        return response.embeddings[0]

    def format_tool_result(self, tool_call: ToolCall, result: str) -> dict:
        return {"role": "tool", "content": result, "tool_name": tool_call.name}


class VLLMBackend:
    """LLM backend backed by vLLM's OpenAI-compatible API."""

    def __init__(self, cfg: VLLMConfig) -> None:
        self._client = openai.AsyncOpenAI(
            base_url=cfg.base_url,
            api_key="none",
            timeout=cfg.timeout,
        )

    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None
    ) -> ChatResponse:
        kwargs: dict = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = await self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        content = msg.content or ""

        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                    id=tc.id,
                )
                for tc in msg.tool_calls
            ]
            raw_msg = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            return ChatResponse(content=content, tool_calls=tool_calls, raw_assistant_message=raw_msg)

        raw_msg = {"role": "assistant", "content": content}
        return ChatResponse(content=content, raw_assistant_message=raw_msg)

    async def list_models(self) -> list[str]:
        try:
            response = await self._client.models.list()
            return sorted(m.id for m in response.data)
        except Exception as exc:
            logger.exception("Failed to list vLLM models: %s", exc)
            return []

    async def embed(self, model: str, text: str) -> list[float]:
        response = await self._client.embeddings.create(model=model, input=text)
        return response.data[0].embedding

    def format_tool_result(self, tool_call: ToolCall, result: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call.id or "", "content": result}


def create_backend(config: Config) -> LLMBackend:
    """Return the chat LLMBackend configured in config.backend."""
    if config.backend == "vllm":
        return VLLMBackend(config.vllm)
    return OllamaBackend(config.ollama)


def create_embed_backend(config: Config) -> LLMBackend:
    """Return the embedding LLMBackend configured in config.rag.embed_backend."""
    if config.rag.embed_backend == "vllm":
        return VLLMBackend(config.vllm)
    return OllamaBackend(config.ollama)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
python -m pytest tests/test_llm_backend.py -v
```

Expected: all pass.

- [ ] **Step 5: Run full suite to check no regressions**

```bash
python -m pytest --tb=short -q
```

Expected: existing tests still pass (some may fail because `agent.py` and `rag.py` still import `ollama` directly — that is fine and will be fixed in Tasks 3 and 4).

- [ ] **Step 6: Commit**

```bash
git add llm_backend.py tests/test_llm_backend.py
git commit -m "feat: add LLMBackend abstraction with Ollama and vLLM implementations

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Refactor agent.py to use LLMBackend

**Files:**
- Modify: `agent.py`
- Modify: `tests/test_agent.py`

- [ ] **Step 1: Update test_agent.py and test_agent_rag.py**

Replace `tests/test_agent.py` entirely (see below), then also replace `tests/test_agent_rag.py`:

```python
# tests/test_agent_rag.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from config import Config, TelegramConfig, OllamaConfig, HistoryConfig, RagConfig
from agent import Agent
from llm_backend import ChatResponse
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

    backend = MagicMock()
    backend.chat = AsyncMock(return_value=ChatResponse(content="42"))
    backend.format_tool_result = MagicMock()
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        result = await agent.run(
            chat_id=1,
            user_message="What is the answer?",
            context="### Context\n[source: test.txt, chunk 0]\nThe answer is 42.",
        )

    assert result == "42"
    messages = backend.chat.call_args[1]["messages"]
    system_msgs = [m for m in messages if m.get("role") == "system"]
    assert any("### Context" in m.get("content", "") for m in system_msgs)


@pytest.mark.asyncio
async def test_agent_run_no_context_no_system_message(tmp_path):
    cfg = make_config()
    cfg.history.db_path = str(tmp_path / "h.db")
    mcp = MagicMock(spec=MCPManager)
    mcp.get_tool_definitions.return_value = []

    backend = MagicMock()
    backend.chat = AsyncMock(return_value=ChatResponse(content="hello"))
    backend.format_tool_result = MagicMock()
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        result = await agent.run(chat_id=1, user_message="hi", context=None)

    assert result == "hello"
    messages = backend.chat.call_args[1]["messages"]
    system_msgs = [m for m in messages if m.get("role") == "system"]
    assert not any("### Context" in m.get("content", "") for m in system_msgs)
```

Now replace `tests/test_agent.py` entirely:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent import Agent, MAX_ITERATIONS
from config import Config, TelegramConfig, OllamaConfig, HistoryConfig
from llm_backend import ChatResponse, ToolCall


def _make_config(tmp_path) -> Config:
    return Config(
        telegram=TelegramConfig(token="tok"),
        ollama=OllamaConfig(default_model="llama3.2"),
        history=HistoryConfig(max_messages=50, db_path=str(tmp_path / "h.db")),
    )


def _make_backend(content: str = "Hello!", tool_calls: list | None = None):
    """Return a mock LLMBackend with chat returning the given ChatResponse."""
    backend = MagicMock()
    response = ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        raw_assistant_message={"role": "assistant", "content": content},
    )
    backend.chat = AsyncMock(return_value=response)
    backend.list_models = AsyncMock(return_value=[])
    backend.format_tool_result = MagicMock(
        return_value={"role": "tool", "content": "tool result", "tool_name": "mock_tool"}
    )
    return backend


@pytest.mark.asyncio
async def test_simple_text_response(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    backend = _make_backend(content="Hello!")
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages") as mock_save:
        result = await agent.run(chat_id=1, user_message="Hi")

    assert result == "Hello!"
    mock_save.assert_called_once()
    saved_messages = mock_save.call_args[0][2]
    assert saved_messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


@pytest.mark.asyncio
async def test_tool_call_then_text_response(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = [
        {"type": "function", "function": {"name": "search_web"}}
    ]
    mcp.call_tool = AsyncMock(return_value="Search result: Python docs")

    tc = ToolCall(name="search_web", arguments={"query": "python"})
    tool_response = ChatResponse(
        content="",
        tool_calls=[tc],
        raw_assistant_message={"role": "assistant", "content": "", "tool_calls": []},
    )
    final_response = ChatResponse(content="Here is what I found.")

    backend = MagicMock()
    backend.chat = AsyncMock(side_effect=[tool_response, final_response])
    backend.format_tool_result = MagicMock(
        return_value={"role": "tool", "content": "Search result: Python docs", "tool_name": "search_web"}
    )
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"), \
         patch("agent.save_messages"):
        result = await agent.run(chat_id=1, user_message="Search python")

    assert result == "Here is what I found."
    mcp.call_tool.assert_called_once_with("search_web", {"query": "python"})

    second_call_messages = backend.chat.call_args_list[1][1]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) >= 1
    assert tool_messages[0]["tool_name"] == "search_web"


@pytest.mark.asyncio
async def test_max_iterations_guard(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    mcp.call_tool = AsyncMock(return_value="result")

    tc = ToolCall(name="loop_tool", arguments={})
    always_tool = ChatResponse(
        content="",
        tool_calls=[tc],
        raw_assistant_message={"role": "assistant", "content": ""},
    )
    backend = MagicMock()
    backend.chat = AsyncMock(return_value=always_tool)
    backend.format_tool_result = MagicMock(
        return_value={"role": "tool", "content": "result", "tool_name": "loop_tool"}
    )
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages") as mock_save:
        result = await agent.run(chat_id=1, user_message="loop")

    assert "maximum" in result.lower()
    mock_save.assert_called_once()
    saved_messages = mock_save.call_args[0][2]
    assert saved_messages[0] == {"role": "user", "content": "loop"}


@pytest.mark.asyncio
async def test_backend_error_returns_error_message(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []

    backend = MagicMock()
    backend.chat = AsyncMock(side_effect=Exception("connection refused"))
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        result = await agent.run(chat_id=1, user_message="Hi")

    assert "error" in result.lower()


@pytest.mark.asyncio
async def test_history_passed_to_backend(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    backend = _make_backend(content="ok")
    agent = Agent(backend, "llama3.2", cfg, mcp)

    existing_history = [
        {"role": "user", "content": "prev question"},
        {"role": "assistant", "content": "prev answer"},
    ]

    with patch("agent.get_history", return_value=existing_history), \
         patch("agent.save_messages"):
        await agent.run(chat_id=1, user_message="new question")

    sent_messages = backend.chat.call_args[1]["messages"]
    assert sent_messages[0] == {"role": "user", "content": "prev question"}
    assert sent_messages[-1] == {"role": "user", "content": "new question"}


def test_set_model_changes_active_model(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    backend = _make_backend()
    agent = Agent(backend, "llama3.2", cfg, mcp)
    assert agent.active_model == "llama3.2"
    agent.set_model("mistral")
    assert agent.active_model == "mistral"


@pytest.mark.asyncio
async def test_list_models_delegates_to_backend(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    backend = MagicMock()
    backend.list_models = AsyncMock(return_value=["llama3.2", "llava"])
    agent = Agent(backend, "llama3.2", cfg, mcp)
    result = await agent.list_models()
    assert result == ["llama3.2", "llava"]


@pytest.mark.asyncio
async def test_image_passed_to_backend(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    backend = _make_backend(content="Nice image!")
    agent = Agent(backend, "llama3.2", cfg, mcp)

    fake_b64 = "aGVsbG8="

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        result = await agent.run(chat_id=1, user_message="What is this?", images=[fake_b64])

    assert result == "Nice image!"
    sent = backend.chat.call_args[1]["messages"]
    user_msg = next(m for m in sent if m["role"] == "user")
    assert user_msg["images"] == [fake_b64]
    assert user_msg["content"] == "What is this?"


@pytest.mark.asyncio
async def test_image_stored_as_placeholder_in_history(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    backend = _make_backend(content="ok")
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages") as mock_save:
        await agent.run(chat_id=1, user_message="describe it", images=["aGVsbG8="])

    saved = mock_save.call_args[0][2]
    assert saved[0]["role"] == "user"
    assert saved[0]["content"] == "[image] describe it"
    assert "aGVsbG8=" not in saved[0]["content"]


@pytest.mark.asyncio
async def test_empty_content_response(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    backend = _make_backend(content="")
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        result = await agent.run(chat_id=1, user_message="Hi")

    assert result == ""


@pytest.mark.asyncio
async def test_text_tool_call_fallback_executed(tmp_path):
    """Agent executes tool calls present in ChatResponse regardless of their source."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    mcp.call_tool = AsyncMock(return_value="All lines good")

    tc = ToolCall(name="get_status", arguments={})
    tool_response = ChatResponse(
        content="",
        tool_calls=[tc],
        raw_assistant_message={"role": "assistant", "content": ""},
    )
    final_response = ChatResponse(content="All lines are running normally.")

    backend = MagicMock()
    backend.chat = AsyncMock(side_effect=[tool_response, final_response])
    backend.format_tool_result = MagicMock(
        return_value={"role": "tool", "content": "All lines good", "tool_name": "get_status"}
    )
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        result = await agent.run(chat_id=1, user_message="list line status")

    assert result == "All lines are running normally."
    mcp.call_tool.assert_called_once_with("get_status", {})
```

- [ ] **Step 2: Run updated tests — expect failures (agent.py not yet changed)**

```bash
python -m pytest tests/test_agent.py -v 2>&1 | head -20
```

Expected: `TypeError: Agent.__init__() takes 3 positional arguments but 5 were given` or similar.

- [ ] **Step 3: Rewrite agent.py**

Replace `agent.py` entirely:

```python
"""Agent module for orchestrating LLM interactions with tool support."""

import logging

from config import Config
from history import get_history, save_messages
from llm_backend import LLMBackend, ChatResponse
from mcp_manager import MCPManager

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10


class Agent:
    """
    Agent that orchestrates LLM interactions with tool-use capabilities.

    Accepts any LLMBackend (Ollama or vLLM), maintains conversation history,
    handles tool invocations through MCP, and returns final text responses.
    """

    def __init__(
        self,
        backend: LLMBackend,
        initial_model: str,
        config: Config,
        mcp_manager: MCPManager,
    ) -> None:
        self._cfg = config
        self._mcp = mcp_manager
        self._backend = backend
        self._active_model = initial_model

    @property
    def active_model(self) -> str:
        return self._active_model

    def set_model(self, model: str) -> None:
        self._active_model = model

    async def list_models(self) -> list[str]:
        return await self._backend.list_models()

    async def run(
        self,
        chat_id: int,
        user_message: str,
        images: list[str] | None = None,
        context: str | None = None,
    ) -> str:
        """
        Run the agent for a single user message.

        Fetches history, calls the LLM backend (handling tool loops), and
        returns the final text response.
        """
        history = await get_history(
            self._cfg.history.db_path, chat_id, self._cfg.history.max_messages
        )
        tools = self._mcp.get_tool_definitions()

        prefix: list[dict] = []
        if context:
            prefix = [{"role": "system", "content": context}]

        user_msg: dict = {"role": "user", "content": user_message}
        if images:
            user_msg["images"] = images

        history_user_content = f"[image] {user_message}" if images else user_message
        messages = prefix + history + [user_msg]

        for _ in range(MAX_ITERATIONS):
            try:
                response: ChatResponse = await self._backend.chat(
                    model=self._active_model,
                    messages=messages,
                    tools=tools or None,
                )
            except Exception as exc:
                logger.exception("Backend chat error for chat_id=%s: %s", chat_id, exc)
                return f"⚠️ Backend error: {exc}"

            if response.tool_calls:
                messages.append(response.raw_assistant_message)
                for tc in response.tool_calls:
                    result = await self._mcp.call_tool(tc.name, tc.arguments)
                    messages.append(self._backend.format_tool_result(tc, result))
            else:
                await save_messages(
                    self._cfg.history.db_path,
                    chat_id,
                    [
                        {"role": "user", "content": history_user_content},
                        {"role": "assistant", "content": response.content},
                    ],
                    self._cfg.history.max_messages,
                )
                return response.content

        warning = "⚠️ Reached maximum tool call iterations. Please try again."
        await save_messages(
            self._cfg.history.db_path,
            chat_id,
            [
                {"role": "user", "content": history_user_content},
                {"role": "assistant", "content": warning},
            ],
            self._cfg.history.max_messages,
        )
        return warning
```

- [ ] **Step 4: Run agent tests — expect pass**

```bash
python -m pytest tests/test_agent.py tests/test_agent_rag.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py tests/test_agent_rag.py
git commit -m "refactor: Agent now accepts LLMBackend dependency injection

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Refactor rag.py to use LLMBackend for embeddings

**Files:**
- Modify: `rag.py`
- Modify: `tests/test_rag.py`

- [ ] **Step 1: Update test_rag.py constructor calls**

In `tests/test_rag.py`, every `RagManager(rag_cfg)` call must become `RagManager(rag_cfg, mock_embed_backend)`. Add a fixture at the top (after the existing imports):

```python
from unittest.mock import AsyncMock, MagicMock, patch
from config import RagConfig
from rag import RagManager


@pytest.fixture
def mock_embed_backend():
    backend = MagicMock()
    backend.embed = AsyncMock(return_value=[0.1] * 768)
    return backend
```

Update every test that creates `RagManager(rag_cfg)` to `RagManager(rag_cfg, mock_embed_backend)` by adding `mock_embed_backend` as a parameter. The existing `patch.object(manager, "_embed", ...)` patches still work because `_embed` remains a method on `RagManager`.

Full updated `tests/test_rag.py`:

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
def mock_embed_backend():
    backend = MagicMock()
    backend.embed = AsyncMock(return_value=[0.1] * 768)
    return backend


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
async def test_ingest_text(rag_cfg, mock_chroma, mock_embed_backend):
    client, col = mock_chroma
    manager = RagManager(rag_cfg, mock_embed_backend)

    with patch.object(manager, "_embed", new=AsyncMock(return_value=[0.1] * 768)):
        count = await manager.ingest("mycol", "test.txt", "Hello world " * 100)

    assert count > 0
    assert col.add.called


@pytest.mark.asyncio
async def test_ingest_skips_duplicate(rag_cfg, mock_chroma, mock_embed_backend):
    client, col = mock_chroma
    col.get.return_value = {"ids": ["existing-id"]}
    manager = RagManager(rag_cfg, mock_embed_backend)

    with patch.object(manager, "_embed", new=AsyncMock(return_value=[0.1] * 768)):
        count = await manager.ingest("mycol", "test.txt", "Hello world " * 100)

    assert count == 0
    col.add.assert_not_called()


@pytest.mark.asyncio
async def test_search_returns_chunks(rag_cfg, mock_chroma, mock_embed_backend):
    client, col = mock_chroma
    manager = RagManager(rag_cfg, mock_embed_backend)

    with patch.object(manager, "_embed", new=AsyncMock(return_value=[0.1] * 768)):
        chunks = await manager.search("who am I?")

    assert len(chunks) == 1
    assert "chunk text" in chunks[0]
    assert "test.txt" in chunks[0]


@pytest.mark.asyncio
async def test_search_empty_when_disabled(tmp_path, mock_embed_backend):
    cfg = RagConfig(enabled=False, db_path=str(tmp_path / "chroma"))
    manager = RagManager(cfg, mock_embed_backend)
    chunks = await manager.search("anything")
    assert chunks == []


def test_chunk_text_splits_correctly():
    from rag import chunk_text
    text = "a" * 1200
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) == 3
    assert chunks[0] == "a" * 500
    assert chunks[1] == "a" * 500
    assert len(chunks[2]) > 0


def test_chunk_text_short_input():
    from rag import chunk_text
    chunks = chunk_text("hello", size=500, overlap=50)
    assert chunks == ["hello"]


def test_chunk_text_empty():
    from rag import chunk_text
    chunks = chunk_text("", size=500, overlap=50)
    assert chunks == []


def test_list_collections(rag_cfg, mock_chroma, mock_embed_backend):
    client, col = mock_chroma
    manager = RagManager(rag_cfg, mock_embed_backend)
    cols = manager.list_collections()
    assert isinstance(cols, list)
```

- [ ] **Step 2: Run updated rag tests — expect failures**

```bash
python -m pytest tests/test_rag.py -v 2>&1 | head -20
```

Expected: `TypeError: RagManager.__init__() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Update rag.py**

Replace `rag.py` entirely:

```python
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

    async def ingest(self, collection_name: str, source: str, text: str) -> int:
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
```

- [ ] **Step 4: Run rag tests — expect pass**

```bash
python -m pytest tests/test_rag.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add rag.py tests/test_rag.py
git commit -m "refactor: RagManager accepts LLMBackend for pluggable embeddings

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Wire backends in bot.py

**Files:**
- Modify: `bot.py`

- [ ] **Step 1: Run full suite to establish baseline**

```bash
python -m pytest --tb=short -q
```

Note the pass count — target is all tests passing after this task too.

- [ ] **Step 2: Update bot.py**

Make these changes to `bot.py`:

**Add import at the top** (after `from rag import RagManager`):

```python
from llm_backend import create_backend, create_embed_backend
```

**Replace the `_post_init` function:**

```python
async def _post_init(application: Application) -> None:
    """Run async startup tasks after the bot is initialized."""
    cfg = application.bot_data["config"]
    os.makedirs(os.path.dirname(cfg.history.db_path) or ".", exist_ok=True)
    await init_db(cfg.history.db_path)
    mcp: MCPManager = application.bot_data["mcp"]
    await mcp.start()
    embed_backend = create_embed_backend(cfg)
    application.bot_data["rag"] = RagManager(cfg.rag, embed_backend)
```

**Replace the `main()` function's agent construction** — find these two lines:

```python
    mcp = MCPManager(cfg.mcp_servers)
    agent = Agent(cfg, mcp)
```

Replace with:

```python
    mcp = MCPManager(cfg.mcp_servers)
    chat_backend = create_backend(cfg)
    initial_model = cfg.vllm.default_model if cfg.backend == "vllm" else cfg.ollama.default_model
    agent = Agent(chat_backend, initial_model, cfg, mcp)
```

Also update the log line at the bottom of `main()`:

```python
    logger.info("Bot starting with model '%s' via %s backend", agent.active_model, cfg.backend)
```

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest --tb=short -q
```

Expected: all tests pass. (`test_bot_rag.py` mocks `bot_data` directly so the `_post_init` / `main()` changes don't affect it.)

- [ ] **Step 4: Commit**

```bash
git add bot.py
git commit -m "feat: wire LLMBackend factory into bot startup

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Update README and push

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add vLLM section to README**

In `README.md`, add a new section **after** the "## RAG (Retrieval-Augmented Generation)" section and **before** "## MCP Server Types":

```markdown
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
```

- [ ] **Step 2: Update Configuration Reference table**

In the Configuration Reference table, add these rows after the RAG rows:

```markdown
| `backend` | `ollama` | Chat backend: `ollama` or `vllm` |
| `vllm.base_url` | `http://localhost:8000` | vLLM API endpoint |
| `vllm.default_model` | — | Model name served by vLLM |
| `vllm.timeout` | `120` | Request timeout in seconds |
| `rag.embed_backend` | `ollama` | Embedding backend: `ollama` or `vllm` |
```

- [ ] **Step 3: Run full suite one final time**

```bash
python -m pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit and push**

```bash
git add README.md
git commit -m "docs: document vLLM backend support

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push
```
