# Streaming & Thinking Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add token-streaming to Telegram replies and a per-chat `/think` toggle that surfaces model reasoning as a Telegram spoiler.

**Architecture:** Tool-call loop iterations stay non-streaming (unchanged); only the final response (no tool calls) is streamed via a new `chat_stream()` method on `LLMBackend`. `bot.py` edits a placeholder message every ≥0.5 s as tokens arrive. Per-chat thinking state lives in `bot_data["think_state"]` (in-memory dict); the global default is read from config.

**Tech Stack:** `ollama` Python SDK (streaming + `think` param), `openai` Python SDK (streaming + `extra_body`), `python-telegram-bot` (`edit_message_text`), `pytest-asyncio`.

---

## File Map

| File | Change |
|---|---|
| `config.py` | Add `think: bool = False` to `OllamaConfig` and `VLLMConfig` |
| `llm_backend.py` | Add `thinking` field to `ChatResponse`; add `chat_stream()` to `LLMBackend` protocol; implement in `OllamaBackend` and `VLLMBackend` |
| `agent.py` | Add `run_stream()` async generator |
| `bot.py` | Add `_think_state` dict, `/think` command, rewrite `handle_message()` to stream |
| `tests/test_config.py` | Test `think` defaults |
| `tests/test_config_vllm.py` | Test `think` defaults for vLLM |
| `tests/test_llm_backend.py` | Test `chat_stream()` for both backends |
| `tests/test_agent.py` | Test `run_stream()` |

---

## Task 1: Config — add `think` field

**Files:**
- Modify: `config.py`
- Test: `tests/test_config.py`, `tests/test_config_vllm.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:
```python
from config import OllamaConfig

def test_ollama_config_think_default():
    cfg = OllamaConfig(default_model="llama3.2")
    assert cfg.think is False

def test_ollama_config_think_enabled():
    cfg = OllamaConfig(default_model="llama3.2", think=True)
    assert cfg.think is True
```

Add to `tests/test_config_vllm.py` (find the existing test file and append):
```python
def test_vllm_config_think_default():
    cfg = VLLMConfig(base_url="http://localhost:8000", default_model="llama3")
    assert cfg.think is False

def test_vllm_config_think_enabled():
    cfg = VLLMConfig(base_url="http://localhost:8000", default_model="llama3", think=True)
    assert cfg.think is True
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/wanleung/Projects/telegram-bot
python -m pytest tests/test_config.py::test_ollama_config_think_default tests/test_config_vllm.py::test_vllm_config_think_default -v
```
Expected: FAIL — `OllamaConfig has no field 'think'`

- [ ] **Step 3: Add `think` field to both config classes**

In `config.py`, update `OllamaConfig`:
```python
class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    default_model: str
    timeout: int = 300
    think: bool = False
```

Update `VLLMConfig`:
```python
class VLLMConfig(BaseModel):
    base_url: str = "http://localhost:8000"
    default_model: str
    timeout: int = 300
    think: bool = False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_config.py tests/test_config_vllm.py -v
```
Expected: all tests PASS (including the 4 new ones)

- [ ] **Step 5: Run full suite to check no regressions**

```bash
python -m pytest --tb=short -q
```
Expected: all 109 tests PASS

- [ ] **Step 6: Commit**

```bash
git add config.py tests/test_config.py tests/test_config_vllm.py
git commit -m "feat: add think flag to OllamaConfig and VLLMConfig

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: `ChatResponse.thinking` + `LLMBackend.chat_stream()` protocol

**Files:**
- Modify: `llm_backend.py`
- Test: `tests/test_llm_backend.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm_backend.py`:
```python
def test_chat_response_thinking_default_none():
    r = ChatResponse(content="hello")
    assert r.thinking is None

def test_chat_response_thinking_set():
    r = ChatResponse(content="answer", thinking="I reasoned about it")
    assert r.thinking == "I reasoned about it"

def test_ollama_backend_has_chat_stream():
    from collections.abc import AsyncIterator
    import inspect
    backend = OllamaBackend(_ollama_cfg())
    assert hasattr(backend, "chat_stream")
    assert callable(backend.chat_stream)

def test_vllm_backend_has_chat_stream():
    backend = VLLMBackend(_vllm_cfg())
    assert hasattr(backend, "chat_stream")
    assert callable(backend.chat_stream)
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_llm_backend.py::test_chat_response_thinking_default_none tests/test_llm_backend.py::test_ollama_backend_has_chat_stream -v
```
Expected: first two FAIL

- [ ] **Step 3: Add `thinking` to `ChatResponse` and `chat_stream` to protocol**

In `llm_backend.py`, update the imports at the top:
```python
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx
import ollama
import openai

from config import Config, OllamaConfig, VLLMConfig
```

Update `ChatResponse`:
```python
@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_assistant_message: dict = field(default_factory=dict)
    thinking: str | None = None
```

Update `LLMBackend` protocol — add `chat_stream` after `chat`:
```python
@runtime_checkable
class LLMBackend(Protocol):
    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None
    ) -> ChatResponse: ...

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        think: bool = False,
    ) -> AsyncIterator[ChatResponse]: ...

    async def list_models(self) -> list[str]: ...

    async def embed(self, model: str, text: str) -> list[float]: ...

    def format_tool_result(self, tool_call: ToolCall, result: str) -> dict: ...
```

**Note:** `chat_stream` is declared as `def` (not `async def`) in the protocol because the concrete implementations are async generator functions — calling them returns an `AsyncIterator` directly without `await`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_llm_backend.py::test_chat_response_thinking_default_none tests/test_llm_backend.py::test_chat_response_thinking_set tests/test_llm_backend.py::test_ollama_backend_has_chat_stream tests/test_llm_backend.py::test_vllm_backend_has_chat_stream -v
```
Expected: all 4 PASS

- [ ] **Step 5: Run full suite**

```bash
python -m pytest --tb=short -q
```
Expected: all existing tests PASS (new field has default, backward-compatible)

- [ ] **Step 6: Commit**

```bash
git add llm_backend.py tests/test_llm_backend.py
git commit -m "feat: add ChatResponse.thinking field and chat_stream protocol method

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: `OllamaBackend.chat_stream()`

**Files:**
- Modify: `llm_backend.py`
- Test: `tests/test_llm_backend.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm_backend.py`:
```python
@pytest.mark.asyncio
async def test_ollama_backend_chat_stream_content_chunks():
    """chat_stream yields ChatResponse chunks with content."""
    backend = OllamaBackend(_ollama_cfg())

    chunk1 = MagicMock()
    chunk1.message.content = "Hello"
    chunk1.message.thinking = None

    chunk2 = MagicMock()
    chunk2.message.content = " world"
    chunk2.message.thinking = None

    async def _fake_stream(*args, **kwargs):
        for c in [chunk1, chunk2]:
            yield c

    with patch.object(backend._client, "chat", return_value=_fake_stream()):
        chunks = []
        async for cr in backend.chat_stream("llama3.2", [{"role": "user", "content": "hi"}], None):
            chunks.append(cr)

    assert len(chunks) == 2
    assert chunks[0].content == "Hello"
    assert chunks[0].thinking is None
    assert chunks[1].content == " world"


@pytest.mark.asyncio
async def test_ollama_backend_chat_stream_thinking_chunks():
    """chat_stream yields thinking fragments when think=True and model returns thinking."""
    backend = OllamaBackend(_ollama_cfg())

    think_chunk = MagicMock()
    think_chunk.message.content = ""
    think_chunk.message.thinking = "I should reason..."

    content_chunk = MagicMock()
    content_chunk.message.content = "Final answer"
    content_chunk.message.thinking = None

    async def _fake_stream(*args, **kwargs):
        for c in [think_chunk, content_chunk]:
            yield c

    with patch.object(backend._client, "chat", return_value=_fake_stream()):
        chunks = []
        async for cr in backend.chat_stream("llama3.2", [], None, think=True):
            chunks.append(cr)

    thinking_chunks = [c for c in chunks if c.thinking]
    content_chunks = [c for c in chunks if c.content]
    assert any(c.thinking == "I should reason..." for c in thinking_chunks)
    assert any(c.content == "Final answer" for c in content_chunks)


@pytest.mark.asyncio
async def test_ollama_backend_chat_stream_passes_think_flag():
    """chat_stream passes think=True to the underlying client when requested."""
    backend = OllamaBackend(_ollama_cfg())

    async def _fake_stream(*args, **kwargs):
        assert kwargs.get("think") is True
        chunk = MagicMock()
        chunk.message.content = "ok"
        chunk.message.thinking = None
        yield chunk

    with patch.object(backend._client, "chat", return_value=_fake_stream()):
        async for _ in backend.chat_stream("llama3.2", [], None, think=True):
            pass
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_llm_backend.py::test_ollama_backend_chat_stream_content_chunks -v
```
Expected: FAIL — `OllamaBackend has no attribute 'chat_stream'` (or similar)

- [ ] **Step 3: Implement `OllamaBackend.chat_stream()`**

Add this method to `OllamaBackend` (after `chat`, before `list_models`):
```python
async def chat_stream(
    self,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    think: bool = False,
) -> AsyncIterator[ChatResponse]:
    async for chunk in await self._client.chat(
        model=model,
        messages=messages,
        stream=True,
        think=think,
    ):
        content = chunk.message.content or ""
        raw_thinking = getattr(chunk.message, "thinking", None)
        thinking = raw_thinking if raw_thinking else None
        if content or thinking:
            yield ChatResponse(content=content, thinking=thinking)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_llm_backend.py::test_ollama_backend_chat_stream_content_chunks tests/test_llm_backend.py::test_ollama_backend_chat_stream_thinking_chunks tests/test_llm_backend.py::test_ollama_backend_chat_stream_passes_think_flag -v
```
Expected: all 3 PASS

- [ ] **Step 5: Run full suite**

```bash
python -m pytest --tb=short -q
```
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add llm_backend.py tests/test_llm_backend.py
git commit -m "feat: implement OllamaBackend.chat_stream with think support

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: `VLLMBackend.chat_stream()`

**Files:**
- Modify: `llm_backend.py`
- Test: `tests/test_llm_backend.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm_backend.py`:
```python
@pytest.mark.asyncio
async def test_vllm_backend_chat_stream_content_chunks():
    """chat_stream yields ChatResponse chunks with content from OpenAI delta."""
    backend = VLLMBackend(_vllm_cfg())

    def _make_chunk(text):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text
        chunk.choices[0].delta.reasoning_content = None
        return chunk

    async def _fake_stream(*args, **kwargs):
        for text in ["Hello", " world"]:
            yield _make_chunk(text)

    with patch.object(backend._client.chat.completions, "create", return_value=_fake_stream()):
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

    with patch.object(backend._client.chat.completions, "create", return_value=_fake_stream()):
        chunks = []
        async for cr in backend.chat_stream("llama3", [], None, think=True):
            chunks.append(cr)

    assert any(c.thinking == "thinking..." for c in chunks)
    assert any(c.content == "Answer" for c in chunks)


@pytest.mark.asyncio
async def test_vllm_backend_chat_stream_passes_enable_thinking_in_extra_body():
    """When think=True, extra_body with enable_thinking is passed to create()."""
    backend = VLLMBackend(_vllm_cfg())
    captured_kwargs = {}

    async def _fake_stream(*args, **kwargs):
        captured_kwargs.update(kwargs)
        chunk = MagicMock()
        chunk.choices[0].delta.content = "ok"
        chunk.choices[0].delta.reasoning_content = None
        yield chunk

    with patch.object(backend._client.chat.completions, "create", return_value=_fake_stream()):
        async for _ in backend.chat_stream("llama3", [], None, think=True):
            pass

    assert captured_kwargs.get("extra_body") == {"enable_thinking": True}
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_llm_backend.py::test_vllm_backend_chat_stream_content_chunks -v
```
Expected: FAIL — `VLLMBackend has no attribute 'chat_stream'`

- [ ] **Step 3: Implement `VLLMBackend.chat_stream()`**

Add this method to `VLLMBackend` (after `chat`, before `list_models`):
```python
async def chat_stream(
    self,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    think: bool = False,
) -> AsyncIterator[ChatResponse]:
    kwargs: dict = {"model": model, "messages": messages, "stream": True}
    if tools:
        kwargs["tools"] = tools
    if think:
        kwargs["extra_body"] = {"enable_thinking": True}

    async for chunk in await self._client.chat.completions.create(**kwargs):
        delta = chunk.choices[0].delta
        content = delta.content or ""
        thinking = getattr(delta, "reasoning_content", None) or None
        if content or thinking:
            yield ChatResponse(content=content, thinking=thinking)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_llm_backend.py::test_vllm_backend_chat_stream_content_chunks tests/test_llm_backend.py::test_vllm_backend_chat_stream_thinking_chunks tests/test_llm_backend.py::test_vllm_backend_chat_stream_passes_enable_thinking_in_extra_body -v
```
Expected: all 3 PASS

- [ ] **Step 5: Run full suite**

```bash
python -m pytest --tb=short -q
```
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add llm_backend.py tests/test_llm_backend.py
git commit -m "feat: implement VLLMBackend.chat_stream with think support

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: `agent.run_stream()`

**Files:**
- Modify: `agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py`:
```python
def _make_streaming_backend(chunks: list[tuple[str, str | None]]):
    """
    Return a mock LLMBackend whose chat_stream yields ChatResponse chunks.
    chunks is a list of (content, thinking) pairs.
    """
    backend = MagicMock()
    # non-streaming chat returns no tool calls (so run_stream goes straight to streaming)
    backend.chat = AsyncMock(return_value=ChatResponse(content="", tool_calls=[]))

    async def _stream(*args, **kwargs):
        for content, thinking in chunks:
            yield ChatResponse(content=content, thinking=thinking)

    backend.chat_stream = _stream
    backend.format_tool_result = MagicMock(
        return_value={"role": "tool", "content": "result", "tool_name": "mock"}
    )
    return backend


@pytest.mark.asyncio
async def test_run_stream_yields_content_chunks(tmp_path):
    """run_stream yields (content, None) tuples for plain text responses."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []

    backend = _make_streaming_backend([("Hello", None), (" world", None)])
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        results = []
        async for content_chunk, thinking_chunk in agent.run_stream(
            chat_id=1, user_message="Hi"
        ):
            results.append((content_chunk, thinking_chunk))

    assert results == [("Hello", None), (" world", None)]


@pytest.mark.asyncio
async def test_run_stream_yields_thinking_chunks(tmp_path):
    """run_stream yields ("", thinking) tuples for thinking fragments."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []

    backend = _make_streaming_backend([("", "I think..."), ("Answer", None)])
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        results = []
        async for content_chunk, thinking_chunk in agent.run_stream(
            chat_id=1, user_message="Hi", think=True
        ):
            results.append((content_chunk, thinking_chunk))

    assert ("", "I think...") in results
    assert ("Answer", None) in results


@pytest.mark.asyncio
async def test_run_stream_saves_history(tmp_path):
    """run_stream saves accumulated content to history after streaming completes."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []

    backend = _make_streaming_backend([("Hello", None), (" there", None)])
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages") as mock_save:
        async for _ in agent.run_stream(chat_id=1, user_message="Hi"):
            pass

    mock_save.assert_called_once()
    saved = mock_save.call_args[0][2]
    assert saved[0] == {"role": "user", "content": "Hi"}
    assert saved[1] == {"role": "assistant", "content": "Hello there"}


@pytest.mark.asyncio
async def test_run_stream_tool_then_stream(tmp_path):
    """run_stream uses non-streaming chat for tool calls, then streams final response."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = [{"type": "function", "function": {"name": "lookup"}}]
    mcp.call_tool = AsyncMock(return_value="tool result")

    tc = ToolCall(name="lookup", arguments={"q": "x"})
    tool_response = ChatResponse(
        content="",
        tool_calls=[tc],
        raw_assistant_message={"role": "assistant", "content": ""},
    )
    # After tool call, non-streaming chat returns no tool calls → triggers streaming
    no_tool_response = ChatResponse(content="", tool_calls=[])

    async def _stream(*args, **kwargs):
        yield ChatResponse(content="Final answer", thinking=None)

    backend = MagicMock()
    backend.chat = AsyncMock(side_effect=[tool_response, no_tool_response])
    backend.chat_stream = _stream
    backend.format_tool_result = MagicMock(
        return_value={"role": "tool", "content": "tool result", "tool_name": "lookup"}
    )
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        results = []
        async for chunk in agent.run_stream(chat_id=1, user_message="look up x"):
            results.append(chunk)

    assert any(content == "Final answer" for content, _ in results)
    mcp.call_tool.assert_called_once_with("lookup", {"q": "x"})


@pytest.mark.asyncio
async def test_run_stream_backend_error(tmp_path):
    """run_stream yields an error tuple when the streaming backend raises."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []

    backend = MagicMock()
    backend.chat = AsyncMock(return_value=ChatResponse(content="", tool_calls=[]))

    async def _stream(*args, **kwargs):
        raise RuntimeError("connection dropped")
        yield  # make it an async generator

    backend.chat_stream = _stream
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        results = []
        async for chunk in agent.run_stream(chat_id=1, user_message="Hi"):
            results.append(chunk)

    combined = "".join(c for c, _ in results)
    assert "error" in combined.lower() or "interrupted" in combined.lower()
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_agent.py::test_run_stream_yields_content_chunks -v
```
Expected: FAIL — `Agent has no attribute 'run_stream'`

- [ ] **Step 3: Implement `run_stream()` in `agent.py`**

Add this import at the top of `agent.py`:
```python
from collections.abc import AsyncIterator
```

Add this method to `Agent` (after `run`):
```python
async def run_stream(
    self,
    chat_id: int,
    user_message: str,
    images: list[str] | None = None,
    context: str | None = None,
    think: bool = False,
) -> AsyncIterator[tuple[str, str | None]]:
    """
    Stream the final LLM response as (content_chunk, thinking_chunk) tuples.

    When no MCP tools are configured, calls chat_stream() directly (single LLM
    call). When tools are present, uses non-streaming chat() for tool-call
    iterations, then chat_stream() for the final response.
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

    async def _stream_final(msgs: list[dict]) -> AsyncIterator[tuple[str, str | None]]:
        """Stream the final response and save to history."""
        content_buf = ""
        try:
            async for chunk in self._backend.chat_stream(
                model=self._active_model,
                messages=msgs,
                tools=None,
                think=think,
            ):
                if chunk.thinking:
                    yield ("", chunk.thinking)
                if chunk.content:
                    content_buf += chunk.content
                    yield (chunk.content, None)
        except Exception as exc:
            logger.exception("Stream error for chat_id=%s: %s", chat_id, exc)
            yield (f"\n⚠️ Stream interrupted: {exc}", None)
            content_buf += f"\n⚠️ Stream interrupted: {exc}"

        await save_messages(
            self._cfg.history.db_path,
            chat_id,
            [
                {"role": "user", "content": history_user_content},
                {"role": "assistant", "content": content_buf},
            ],
            self._cfg.history.max_messages,
        )

    # No tools: single streaming call, no double LLM invocation
    if not tools:
        async for item in _stream_final(messages):
            yield item
        return

    # Has tools: non-streaming loop for tool detection, stream final response
    for _ in range(MAX_ITERATIONS):
        try:
            response: ChatResponse = await self._backend.chat(
                model=self._active_model,
                messages=messages,
                tools=tools,
            )
        except Exception as exc:
            logger.exception("Backend chat error for chat_id=%s: %s", chat_id, exc)
            yield (f"⚠️ Backend error: {exc}", None)
            return

        if response.tool_calls:
            messages.append(response.raw_assistant_message)
            for tc in response.tool_calls:
                result = await self._mcp.call_tool(tc.name, tc.arguments)
                messages.append(self._backend.format_tool_result(tc, result))
        else:
            async for item in _stream_final(messages):
                yield item
            return

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
    yield (warning, None)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_agent.py::test_run_stream_yields_content_chunks tests/test_agent.py::test_run_stream_yields_thinking_chunks tests/test_agent.py::test_run_stream_saves_history tests/test_agent.py::test_run_stream_tool_then_stream tests/test_agent.py::test_run_stream_backend_error -v
```
Expected: all 5 PASS

- [ ] **Step 5: Run full suite**

```bash
python -m pytest --tb=short -q
```
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add Agent.run_stream() async generator

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 6: `bot.py` — `/think` command and streaming `handle_message()`

**Files:**
- Modify: `bot.py`

No unit tests for bot handlers (Telegram API is integration-only). Verify by running the full pytest suite to confirm no regressions.

- [ ] **Step 1: Add `import html` and `import time` to `bot.py`**

Update the imports block at the top of `bot.py`:
```python
import base64
import html
import logging
import os
import time

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config import load_config
from history import init_db, clear_history
from mcp_manager import MCPManager
from agent import Agent
from utils import md_to_telegram_html
from rag import RagManager
from ingest import ingest_source
from llm_backend import create_backend, create_embed_backend
```

- [ ] **Step 2: Add the `/think` command handler**

Add this function after `cmd_collections` (before `handle_message`):
```python
async def cmd_think(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /think command.

    Toggles thinking mode for the current chat. When on, the model's
    reasoning is shown as a spoiler before the answer.
    """
    cfg = context.bot_data["config"]
    think_state: dict[int, bool] = context.bot_data.setdefault("think_state", {})
    chat_id = update.effective_chat.id

    default_think = cfg.ollama.think if cfg.backend == "ollama" else cfg.vllm.think
    current = think_state.get(chat_id, default_think)
    think_state[chat_id] = not current
    status = "ON 🧠" if not current else "OFF"
    await update.message.reply_text(f"Thinking mode {status}")
```

- [ ] **Step 3: Add `_build_reply()` helper**

Add this function just before `handle_message`:
```python
def _build_reply(content_buf: str, thinking_buf: str, final: bool) -> str:
    """
    Build the Telegram HTML reply string.

    During streaming (final=False): show raw escaped text.
    On final edit: apply markdown→HTML conversion and prepend thinking spoiler.
    """
    if final:
        formatted = md_to_telegram_html(content_buf) if content_buf else "<i>(no response)</i>"
        if thinking_buf:
            escaped_thinking = html.escape(thinking_buf)
            return f"<tg-spoiler>🤔 Thinking:\n{escaped_thinking}</tg-spoiler>\n\n{formatted}"
        return formatted
    else:
        display = html.escape(content_buf) if content_buf else "⏳"
        if thinking_buf:
            return f"<tg-spoiler>🤔 Thinking…</tg-spoiler>\n\n{display}"
        return display
```

- [ ] **Step 4: Rewrite `handle_message()` to use streaming**

Replace the existing `handle_message` function with:
```python
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle user text messages and photo uploads.

    Searches the RAG knowledge base and injects relevant context into the
    agent prompt. Streams the final LLM response, editing a placeholder
    message every ≥0.5 s as tokens arrive.
    """
    agent: Agent = context.bot_data["agent"]
    rag: RagManager = context.bot_data["rag"]
    cfg = context.bot_data["config"]
    think_state: dict[int, bool] = context.bot_data.setdefault("think_state", {})

    chat_id = update.effective_chat.id
    default_think = cfg.ollama.think if cfg.backend == "ollama" else cfg.vllm.think
    think = think_state.get(chat_id, default_think)

    images: list[str] | None = None
    user_text = update.message.text or update.message.caption or ""

    if update.message.photo:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        image_bytes = await file.download_as_bytearray()
        images = [base64.b64encode(image_bytes).decode()]

    # RAG: retrieve relevant chunks and format as context block
    rag_context: str | None = None
    if user_text.strip():
        try:
            chunks = await rag.search(user_text)
            if chunks:
                rag_context = "### Context\n" + "\n\n".join(chunks)
        except Exception as exc:
            logger.warning("RAG search failed, continuing without context: %s", exc)

    await update.message.chat.send_action("typing")
    placeholder = await update.message.reply_text("⏳")

    content_buf = ""
    thinking_buf = ""
    last_edit: float = 0.0

    try:
        async for content_chunk, thinking_chunk in agent.run_stream(
            chat_id=chat_id,
            user_message=user_text,
            images=images,
            context=rag_context,
            think=think,
        ):
            if thinking_chunk:
                thinking_buf += thinking_chunk
            if content_chunk:
                content_buf += content_chunk

            now = time.monotonic()
            if now - last_edit >= 0.5:
                text = _build_reply(content_buf, thinking_buf if think else "", final=False)
                try:
                    await placeholder.edit_text(text, parse_mode="HTML")
                    last_edit = now
                except Exception:
                    pass  # ignore edit errors during streaming (e.g. message not modified)
    except Exception as exc:
        logger.exception("Streaming error for chat_id=%s: %s", chat_id, exc)
        content_buf += f"\n⚠️ Stream error: {exc}"

    final_text = _build_reply(content_buf, thinking_buf if think else "", final=True)
    await placeholder.edit_text(final_text, parse_mode="HTML")
```

- [ ] **Step 5: Register the `/think` handler in `main()`**

In `main()`, add the `/think` handler after the existing command handlers (before the message handlers):
```python
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("tools", cmd_tools))
    app.add_handler(CommandHandler("ingest", cmd_ingest))
    app.add_handler(CommandHandler("collections", cmd_collections))
    app.add_handler(CommandHandler("think", cmd_think))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_message))
```

- [ ] **Step 6: Update `/start` help text to include `/think`**

Update `cmd_start` to mention the new command:
```python
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    agent: Agent = context.bot_data["agent"]
    await update.message.reply_text(
        f"👋 Hello! I'm your Ollama-powered assistant.\n"
        f"Current model: <code>{html.escape(agent.active_model)}</code>\n\n"
        f"Commands:\n"
        f"  /models — list available models\n"
        f"  /model &lt;name&gt; — switch model\n"
        f"  /clear — clear conversation history\n"
        f"  /tools — list available MCP tools\n"
        f"  /think — toggle thinking mode (shows model reasoning)\n"
        f"  /ingest &lt;collection&gt; &lt;url-or-path&gt; — add document to RAG\n"
        f"  /collections — list RAG knowledge base collections",
        parse_mode="HTML",
    )
```

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest --tb=short -q
```
Expected: all tests PASS

- [ ] **Step 8: Commit**

```bash
git add bot.py
git commit -m "feat: add /think command and streaming handle_message

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Final verification

- [ ] **Run full suite one last time**

```bash
python -m pytest -v
```
Expected: all tests PASS (≥115 tests including the ~6 new ones from Tasks 1-5)

- [ ] **Push to origin**

```bash
git push origin master
```
