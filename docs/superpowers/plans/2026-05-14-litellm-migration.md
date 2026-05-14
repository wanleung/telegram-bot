# LiteLLM Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `ollama` and `openai` SDK calls inside `OllamaBackend` and `VLLMBackend` with `litellm.acompletion` / `litellm.aembedding`, keeping `config.py`, `bot.py`, and `agent.py` unchanged.

**Architecture:** Both backends keep the same public interface (`LLMBackend` Protocol). Internally, `__init__` stores raw config fields instead of building an SDK client. All LLM calls go through `litellm.acompletion` (using `"ollama/<model>"` or `"hosted_vllm/<model>"` model strings) and `litellm.aembedding`. `list_models()` in both backends uses direct `httpx.AsyncClient` GET calls since LiteLLM has no model-listing API.

**Tech Stack:** `litellm`, `httpx` (already present), `pytest`, `unittest.mock`

---

## File Map

| File | Change |
|------|--------|
| `requirements.txt` | Remove `ollama`, `openai>=1.0`; add `litellm` |
| `llm_backend.py` | Remove `import ollama`, `import openai`; add `import litellm`; rewrite both backends |
| `tests/test_llm_backend.py` | Update patch targets and mock structures for both backends |

---

### Task 1: Update `requirements.txt` and install `litellm`

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Edit `requirements.txt`**

Replace the `ollama` and `openai>=1.0` lines with `litellm`:

```
python-telegram-bot==21.*
litellm
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

- [ ] **Step 2: Install the new dependency**

```bash
/home/wanleung/Projects/telegram-bot/venv/bin/pip install litellm
```

Expected: litellm and its dependencies install successfully.

- [ ] **Step 3: Verify litellm imports**

```bash
/home/wanleung/Projects/telegram-bot/venv/bin/python -c "import litellm; print(litellm.__version__)"
```

Expected: version string printed with no errors.

- [ ] **Step 4: Commit**

```bash
cd /home/wanleung/Projects/telegram-bot
git add requirements.txt
git commit -m "chore: replace ollama+openai deps with litellm

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Update `OllamaBackend` tests

**Files:**
- Modify: `tests/test_llm_backend.py` (Ollama section only: lines 58–147)

The patch target changes from `backend._client.<method>` to `"litellm.acompletion"` or `"litellm.aembedding"`. The mock response structure changes from Ollama-SDK objects to OpenAI-compatible objects (`.choices[0].message`). `format_tool_result` now uses `tool_call_id` (not `tool_name`). `list_models` now mocks `httpx.AsyncClient`.

- [ ] **Step 1: Replace the Ollama tests block in `tests/test_llm_backend.py`**

Find the block `# --- OllamaBackend ---` through `def test_ollama_format_tool_result():` and replace it with:

```python
# --- OllamaBackend ---

@pytest.mark.asyncio
async def test_ollama_backend_chat_plain_text():
    backend = OllamaBackend(_ollama_cfg())
    msg = MagicMock()
    msg.content = "Hello!"
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        result = await backend.chat("llama3.2", [{"role": "user", "content": "hi"}], None)
    assert isinstance(result, ChatResponse)
    assert result.content == "Hello!"
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_ollama_backend_chat_structured_tool_calls():
    backend = OllamaBackend(_ollama_cfg())
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "search"
    tc.function.arguments = json.dumps({"query": "python"})
    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tc]
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
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
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        result = await backend.chat("llama3.2", [], None)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_status"
    assert result.content == ""  # stripped


@pytest.mark.asyncio
async def test_ollama_backend_list_models():
    backend = OllamaBackend(_ollama_cfg())
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "models": [{"model": "llava:latest"}, {"model": "llama3.2:latest"}]
    }
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await backend.list_models()
    assert result == ["llama3.2:latest", "llava:latest"]


@pytest.mark.asyncio
async def test_ollama_backend_list_models_error():
    backend = OllamaBackend(_ollama_cfg())
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(side_effect=Exception("conn refused"))
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await backend.list_models()
    assert result == []


@pytest.mark.asyncio
async def test_ollama_backend_embed():
    backend = OllamaBackend(_ollama_cfg())
    emb = MagicMock()
    emb.embedding = [0.1, 0.2, 0.3]
    resp = MagicMock()
    resp.data = [emb]
    with patch("litellm.aembedding", new=AsyncMock(return_value=resp)):
        result = await backend.embed("nomic-embed-text", "hello")
    assert result == [0.1, 0.2, 0.3]


def test_ollama_format_tool_result():
    backend = OllamaBackend(_ollama_cfg())
    tc = ToolCall(name="search", arguments={}, id="call_abc")
    msg = backend.format_tool_result(tc, "result text")
    assert msg == {"role": "tool", "tool_call_id": "call_abc", "content": "result text"}


def test_ollama_format_tool_result_no_id():
    backend = OllamaBackend(_ollama_cfg())
    tc = ToolCall(name="search", arguments={}, id=None)
    msg = backend.format_tool_result(tc, "result text")
    assert msg == {"role": "tool", "tool_call_id": "", "content": "result text"}
```

- [ ] **Step 2: Run the Ollama tests — expect failures**

```bash
cd /home/wanleung/Projects/telegram-bot
/home/wanleung/Projects/telegram-bot/venv/bin/python -m pytest tests/test_llm_backend.py -k "ollama" -v --tb=short 2>&1 | head -60
```

Expected: Most Ollama tests FAIL because `OllamaBackend` still uses the old `ollama` SDK.

---

### Task 3: Rewrite `OllamaBackend` with LiteLLM

**Files:**
- Modify: `llm_backend.py`

- [ ] **Step 1: Replace the imports and `OllamaBackend` class in `llm_backend.py`**

Replace the top of the file (lines 1–16) with:

```python
"""LLM backend abstraction: Ollama and vLLM via LiteLLM."""

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx
import litellm

from config import Config, OllamaConfig, VLLMConfig

logger = logging.getLogger(__name__)
litellm.suppress_debug_info = True
```

Then replace the entire `OllamaBackend` class (lines 86–166) with:

```python
class OllamaBackend:
    """LLM backend backed by Ollama via LiteLLM."""

    def __init__(self, cfg: OllamaConfig) -> None:
        self._api_base = cfg.base_url
        self._timeout = cfg.timeout

    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None
    ) -> ChatResponse:
        kwargs: dict = {
            "model": f"ollama/{model}",
            "api_base": self._api_base,
            "messages": messages,
            "timeout": self._timeout,
        }
        if tools:
            kwargs["tools"] = tools
        response = await litellm.acompletion(**kwargs)
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

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        think: bool = False,
    ) -> AsyncIterator[ChatResponse]:
        kwargs: dict = {
            "model": f"ollama/{model}",
            "api_base": self._api_base,
            "messages": messages,
            "stream": True,
            "timeout": self._timeout,
        }
        if think:
            kwargs["think"] = True
        async for chunk in await litellm.acompletion(**kwargs):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content or ""
            thinking = getattr(delta, "thinking", None) or None
            if content or thinking:
                yield ChatResponse(content=content, thinking=thinking)

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._api_base}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return sorted(m["model"] for m in data.get("models", []))
        except Exception as exc:
            logger.exception("Failed to list Ollama models: %s", exc)
            return []

    async def embed(self, model: str, text: str) -> list[float]:
        response = await litellm.aembedding(
            model=f"ollama/{model}",
            api_base=self._api_base,
            input=text,
            timeout=self._timeout,
        )
        return response.data[0].embedding

    def format_tool_result(self, tool_call: ToolCall, result: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call.id or "", "content": result}
```

- [ ] **Step 2: Run the Ollama tests — expect passes**

```bash
cd /home/wanleung/Projects/telegram-bot
/home/wanleung/Projects/telegram-bot/venv/bin/python -m pytest tests/test_llm_backend.py -k "ollama" -v --tb=short 2>&1 | tail -20
```

Expected: All Ollama backend tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/wanleung/Projects/telegram-bot
git add llm_backend.py tests/test_llm_backend.py
git commit -m "feat: migrate OllamaBackend to LiteLLM

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Update `VLLMBackend` tests

**Files:**
- Modify: `tests/test_llm_backend.py` (vLLM section: lines 149–233 and streaming lines 256–445)

The patch target changes from `backend._client.chat.completions.create` to `"litellm.acompletion"`, and from `backend._client.embeddings.create` to `"litellm.aembedding"`. `list_models` mocks `httpx.AsyncClient`. Streaming tests patch `litellm.acompletion` directly. The `_has_chat_stream` tests still check for the method on the backend class.

- [ ] **Step 1: Replace the VLLMBackend test block in `tests/test_llm_backend.py`**

Find the block `# --- VLLMBackend ---` through `def test_vllm_format_tool_result_no_id():` and replace it with:

```python
# --- VLLMBackend ---

@pytest.mark.asyncio
async def test_vllm_backend_chat_plain_text():
    backend = VLLMBackend(_vllm_cfg())
    msg = MagicMock()
    msg.content = "Hello from vLLM!"
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
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
    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tc]
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        result = await backend.chat("llama3", [], [])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "python"}
    assert result.tool_calls[0].id == "call_abc"


@pytest.mark.asyncio
async def test_vllm_backend_list_models():
    backend = VLLMBackend(_vllm_cfg())
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"id": "mistral"}, {"id": "llama3"}]
    }
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await backend.list_models()
    assert result == ["llama3", "mistral"]


@pytest.mark.asyncio
async def test_vllm_backend_list_models_error():
    backend = VLLMBackend(_vllm_cfg())
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(side_effect=Exception("conn refused"))
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await backend.list_models()
    assert result == []


@pytest.mark.asyncio
async def test_vllm_backend_embed():
    backend = VLLMBackend(_vllm_cfg())
    emb = MagicMock()
    emb.embedding = [0.4, 0.5, 0.6]
    resp = MagicMock()
    resp.data = [emb]
    with patch("litellm.aembedding", new=AsyncMock(return_value=resp)):
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
```

- [ ] **Step 2: Replace the streaming tests block in `tests/test_llm_backend.py`**

Find the block `# --- ChatResponse.thinking ---` through the end of the file and replace it with:

```python
# --- ChatResponse.thinking ---

def test_chat_response_thinking_default_none():
    r = ChatResponse(content="hello")
    assert r.thinking is None


def test_chat_response_thinking_set():
    r = ChatResponse(content="answer", thinking="I reasoned about it")
    assert r.thinking == "I reasoned about it"


def test_ollama_backend_has_chat_stream():
    backend = OllamaBackend(_ollama_cfg())
    assert hasattr(backend, "chat_stream")
    assert callable(backend.chat_stream)


def test_vllm_backend_has_chat_stream():
    backend = VLLMBackend(_vllm_cfg())
    assert hasattr(backend, "chat_stream")
    assert callable(backend.chat_stream)


@pytest.mark.asyncio
async def test_vllm_backend_chat_stream_content_chunks():
    """chat_stream yields ChatResponse chunks with content from delta."""
    backend = VLLMBackend(_vllm_cfg())

    def _make_chunk(text):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text
        chunk.choices[0].delta.reasoning_content = None
        return chunk

    async def _fake_stream(*args, **kwargs):
        for text in ["Hello", " world"]:
            yield _make_chunk(text)

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        chunks = []
        async for cr in backend.chat_stream("llama3", [{"role": "user", "content": "hi"}], None):
            chunks.append(cr)

    assert len(chunks) == 2
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"


@pytest.mark.asyncio
async def test_vllm_backend_chat_stream_thinking_chunks():
    """chat_stream yields thinking from reasoning_content when think=True."""
    backend = VLLMBackend(_vllm_cfg())

    def _make_chunk(text=None, reasoning=None):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text or ""
        chunk.choices[0].delta.reasoning_content = reasoning
        return chunk

    async def _fake_stream(*args, **kwargs):
        yield _make_chunk(reasoning="thinking...")
        yield _make_chunk(text="Answer")

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        chunks = []
        async for cr in backend.chat_stream("llama3", [], None, think=True):
            chunks.append(cr)

    assert any(c.thinking == "thinking..." for c in chunks)
    assert any(c.content == "Answer" for c in chunks)


@pytest.mark.asyncio
async def test_vllm_backend_chat_stream_passes_enable_thinking_in_extra_body():
    """When think=True, extra_body with enable_thinking is passed to acompletion."""
    backend = VLLMBackend(_vllm_cfg())
    captured_kwargs = {}

    async def _fake_stream(*args, **kwargs):
        captured_kwargs.update(kwargs)
        chunk = MagicMock()
        chunk.choices[0].delta.content = "ok"
        chunk.choices[0].delta.reasoning_content = None
        yield chunk

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        async for _ in backend.chat_stream("llama3", [], None, think=True):
            pass

    assert captured_kwargs.get("extra_body") == {"enable_thinking": True}


# --- OllamaBackend.chat_stream ---

@pytest.mark.asyncio
async def test_ollama_backend_chat_stream_content_chunks():
    """chat_stream yields ChatResponse chunks with content."""
    backend = OllamaBackend(_ollama_cfg())

    def _make_chunk(text):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text
        chunk.choices[0].delta.thinking = None
        return chunk

    async def _fake_stream(*args, **kwargs):
        for text in ["Hello", " world"]:
            yield _make_chunk(text)

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        chunks = []
        async for cr in backend.chat_stream("llama3.2", [{"role": "user", "content": "hi"}], None):
            chunks.append(cr)

    assert len(chunks) == 2
    assert chunks[0].content == "Hello"
    assert chunks[0].thinking is None
    assert chunks[1].content == " world"


@pytest.mark.asyncio
async def test_ollama_backend_chat_stream_thinking_chunks():
    """chat_stream yields thinking fragments when think=True."""
    backend = OllamaBackend(_ollama_cfg())

    def _make_chunk(text=None, thinking=None):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text or ""
        chunk.choices[0].delta.thinking = thinking
        return chunk

    async def _fake_stream(*args, **kwargs):
        yield _make_chunk(thinking="I should reason...")
        yield _make_chunk(text="Final answer")

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        chunks = []
        async for cr in backend.chat_stream("llama3.2", [], None, think=True):
            chunks.append(cr)

    assert any(c.thinking == "I should reason..." for c in chunks)
    assert any(c.content == "Final answer" for c in chunks)


@pytest.mark.asyncio
async def test_ollama_backend_chat_stream_passes_think_flag():
    """chat_stream passes think=True kwarg to acompletion when requested."""
    backend = OllamaBackend(_ollama_cfg())
    captured_kwargs = {}

    async def _fake_stream(*args, **kwargs):
        captured_kwargs.update(kwargs)
        chunk = MagicMock()
        chunk.choices[0].delta.content = "ok"
        chunk.choices[0].delta.thinking = None
        yield chunk

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        async for _ in backend.chat_stream("llama3.2", [], None, think=True):
            pass

    assert captured_kwargs.get("think") is True


@pytest.mark.asyncio
async def test_vllm_backend_chat_stream_skips_empty_choices_chunk():
    """Chunks with choices=[] must not raise IndexError."""
    backend = VLLMBackend(_vllm_cfg())

    def _make_chunk(text=None, empty=False):
        chunk = MagicMock()
        chunk.choices = [] if empty else [MagicMock()]
        if not empty:
            chunk.choices[0].delta.content = text or ""
            chunk.choices[0].delta.reasoning_content = None
        return chunk

    async def _fake_stream(*args, **kwargs):
        yield _make_chunk(text="Hello")
        yield _make_chunk(empty=True)

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        chunks = [cr async for cr in backend.chat_stream("llama3", [], None)]

    assert len(chunks) == 1
    assert chunks[0].content == "Hello"


@pytest.mark.asyncio
async def test_vllm_backend_chat_stream_no_extra_body_when_think_false():
    """When think=False, extra_body must not be included."""
    backend = VLLMBackend(_vllm_cfg())
    captured_kwargs = {}

    async def _fake_stream(*args, **kwargs):
        captured_kwargs.update(kwargs)
        chunk = MagicMock()
        chunk.choices[0].delta.content = "ok"
        chunk.choices[0].delta.reasoning_content = None
        yield chunk

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        async for _ in backend.chat_stream("llama3", [], None, think=False):
            pass

    assert "extra_body" not in captured_kwargs


@pytest.mark.asyncio
async def test_ollama_backend_chat_stream_no_think_kwarg_when_false():
    """When think=False, think kwarg must not be sent."""
    backend = OllamaBackend(_ollama_cfg())
    captured_kwargs = {}

    async def _fake_stream(*args, **kwargs):
        captured_kwargs.update(kwargs)
        chunk = MagicMock()
        chunk.choices[0].delta.content = "ok"
        chunk.choices[0].delta.thinking = None
        yield chunk

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        async for _ in backend.chat_stream("llama3.2", [], None, think=False):
            pass

    assert "think" not in captured_kwargs


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

- [ ] **Step 3: Run the vLLM tests — expect failures**

```bash
cd /home/wanleung/Projects/telegram-bot
/home/wanleung/Projects/telegram-bot/venv/bin/python -m pytest tests/test_llm_backend.py -k "vllm" -v --tb=short 2>&1 | head -40
```

Expected: Most vLLM tests FAIL because `VLLMBackend` still uses the old `openai` SDK.

---

### Task 5: Rewrite `VLLMBackend` with LiteLLM

**Files:**
- Modify: `llm_backend.py`

- [ ] **Step 1: Replace the `VLLMBackend` class in `llm_backend.py`**

Replace the entire `VLLMBackend` class (the block starting with `class VLLMBackend:` through the `format_tool_result` method) with:

```python
class VLLMBackend:
    """LLM backend backed by vLLM via LiteLLM."""

    def __init__(self, cfg: VLLMConfig) -> None:
        self._api_base = cfg.base_url
        self._timeout = cfg.timeout

    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None
    ) -> ChatResponse:
        kwargs: dict = {
            "model": f"hosted_vllm/{model}",
            "api_base": self._api_base,
            "messages": messages,
            "timeout": self._timeout,
        }
        if tools:
            kwargs["tools"] = tools
        response = await litellm.acompletion(**kwargs)
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

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        think: bool = False,
    ) -> AsyncIterator[ChatResponse]:
        kwargs: dict = {
            "model": f"hosted_vllm/{model}",
            "api_base": self._api_base,
            "messages": messages,
            "stream": True,
            "timeout": self._timeout,
        }
        if tools:
            kwargs["tools"] = tools
        if think:
            kwargs["extra_body"] = {"enable_thinking": True}
        async for chunk in await litellm.acompletion(**kwargs):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content or ""
            thinking = getattr(delta, "reasoning_content", None) or None
            if content or thinking:
                yield ChatResponse(content=content, thinking=thinking)

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._api_base}/v1/models")
                resp.raise_for_status()
                data = resp.json()
                return sorted(m["id"] for m in data.get("data", []))
        except Exception as exc:
            logger.exception("Failed to list vLLM models: %s", exc)
            return []

    async def embed(self, model: str, text: str) -> list[float]:
        response = await litellm.aembedding(
            model=f"hosted_vllm/{model}",
            api_base=self._api_base,
            input=text,
            timeout=self._timeout,
        )
        return response.data[0].embedding

    def format_tool_result(self, tool_call: ToolCall, result: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call.id or "", "content": result}
```

- [ ] **Step 2: Run all backend tests — expect all pass**

```bash
cd /home/wanleung/Projects/telegram-bot
/home/wanleung/Projects/telegram-bot/venv/bin/python -m pytest tests/test_llm_backend.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/wanleung/Projects/telegram-bot
git add llm_backend.py tests/test_llm_backend.py
git commit -m "feat: migrate VLLMBackend to LiteLLM

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Full test suite validation and README update

**Files:**
- Run: `tests/` (all)
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/wanleung/Projects/telegram-bot
/home/wanleung/Projects/telegram-bot/venv/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: All 141 tests PASS (same count as before migration).

- [ ] **Step 2: Update README.md — Backend section**

Find the section describing the `backend:` config or LLM backends and update to mention LiteLLM. Replace or add to the existing backend description paragraph:

```markdown
## LLM Backend

The bot routes all LLM calls through **[LiteLLM](https://github.com/BerriAI/litellm)**, a unified async interface supporting 100+ providers. Two backends are configured via `config.yaml`:

- **`ollama`** (default) — local Ollama instance; model string sent as `ollama/<model>`
- **`vllm`** — local vLLM OpenAI-compatible server; model string sent as `hosted_vllm/<model>`

Config keys (`ollama:` and `vllm:`) are unchanged from previous versions.
```

- [ ] **Step 3: Commit final**

```bash
cd /home/wanleung/Projects/telegram-bot
git add README.md
git commit -m "docs: update README for LiteLLM migration

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

- [ ] **Step 4: Push**

```bash
cd /home/wanleung/Projects/telegram-bot
git push
```
