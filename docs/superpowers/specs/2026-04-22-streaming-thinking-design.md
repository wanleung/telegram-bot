# Streaming & Thinking Mode Design

**Date:** 2026-04-22  
**Status:** Approved

## Problem

The bot currently waits for the full LLM response before replying (non-streaming). For slow or large models this means long silences. Additionally, reasoning models (e.g. deepseek-r1) can produce a "thinking" chain before answering — currently this is silently discarded.

## Goals

1. Stream the final LLM response to Telegram as tokens arrive (live message edits).
2. Let users toggle "thinking mode" per-chat (`/think`), with a global default in config.
3. Render thinking content as a Telegram spoiler before the answer.

## Out of Scope

- Streaming during tool-call iterations (tool resolution stays non-streaming).
- Per-model thinking config (one flag per backend).
- Persisting thinking toggle across bot restarts.

---

## Architecture Overview

Two independent features added on top of the existing `LLMBackend` abstraction:

- **Streaming** — new `chat_stream()` method on `LLMBackend`; `agent.py` exposes `run_stream()`; `bot.py` uses it for all user messages.
- **Thinking** — `ChatResponse.thinking` field; `think: bool` config flag; per-chat toggle dict in `bot.py`; `/think` command.

No changes to RAG, MCP, history, or connection setup.

---

## Component Changes

### `llm_backend.py`

**`ChatResponse`** — add field:
```python
thinking: str | None = None
```

**`LLMBackend` protocol** — add method:
```python
async def chat_stream(
    self, model: str, messages: list[dict], tools: list[dict] | None
) -> AsyncIterator[ChatResponse]: ...
```
Yields incremental `ChatResponse` objects where `content` is a token fragment and `thinking` is a thinking fragment (or `None`). Since `chat_stream()` is only called for the final (no-tool-call) response, `tool_calls` is always empty in streamed chunks.

**`OllamaBackend.chat_stream()`**:
- Calls `client.chat(..., stream=True, think=<from config>)`
- Yields one `ChatResponse` per chunk with `content` and/or `thinking` fragments
- Passes `think` flag from `OllamaConfig.think`

**`VLLMBackend.chat_stream()`**:
- Calls `client.chat.completions.create(..., stream=True)`
- Yields delta content chunks
- Thinking via `extra_body={"enable_thinking": True}` when `VLLMConfig.think=True` (model-dependent; silently ignored if unsupported)

### `config.py`

```python
class OllamaConfig(BaseModel):
    think: bool = False          # enable thinking/reasoning mode

class VLLMConfig(BaseModel):
    think: bool = False          # enable thinking/reasoning mode (model-dependent)
```

### `agent.py`

**`run_stream(chat_id, user_message, images, context, think)`** — async generator:
1. Build history + messages (same as `run()`).
2. Tool-call loop using non-streaming `chat()` — unchanged from `run()`.
3. When loop exits with no tool calls: call `chat_stream()` and yield `(content_chunk: str, thinking_chunk: str | None)` tuples.
4. Save final accumulated content to history on completion.

**`run()` is kept unchanged** for any future non-streaming callers.

### `bot.py`

**State:**
```python
_think_state: dict[int, bool] = {}  # per-chat thinking toggle
```

**`/think` command:**
- Toggles `_think_state[chat_id]`; falls back to `cfg.ollama.think` (or `cfg.vllm.think`) when not set
- Replies: `🧠 Thinking mode ON` / `🧠 Thinking mode OFF`
- Default lookup: `cfg.ollama.think` when backend is `ollama`; `cfg.vllm.think` when backend is `vllm`

**`handle_message()` streaming flow:**
1. Send `typing` action
2. Send placeholder message `…`
3. Call `agent.run_stream(...)` — yields `(content_chunk, thinking_chunk)` tuples
4. Accumulate `content_buf` and `thinking_buf`
5. Throttle: edit message at most every 0.5s
6. On completion: final edit with full formatted response (HTML parse mode)

**Thinking rendering** (HTML, only when thinking content present and think enabled):
```
<tg-spoiler>🤔 Thinking:\n{thinking_buf}</tg-spoiler>\n\n{content_buf}
```
If thinking is off or model produces no thinking content, render `content_buf` only.

---

## Data Flow

```
user message
  → send "…" placeholder
  → run_stream()
      tool loops: non-streaming chat() [silent, no edits]
      final response: chat_stream() yields chunks
          → accumulate content_buf + thinking_buf
          → edit message every 0.5s
  → final edit: full formatted response
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Stream error mid-response | Catch exception; append `⚠️ Stream interrupted` to partial message; save partial to history |
| Telegram edit rate limit | Throttle to max 1 edit per 0.5s; skip intermediate edits if chunks arrive faster |
| `think=True` but model returns no thinking | Silently ignored — render answer only |
| vLLM model doesn't support thinking | `extra_body` field is ignored by vLLM; no error |
| Tool calls on final iteration | Cannot happen by design; guarded by `MAX_ITERATIONS` |

---

## Testing

- **`test_llm_backend.py`** — streaming tests: `AsyncMock` yields chunks; assert accumulated `content` and `thinking` correct for both backends
- **`test_agent.py`** — `test_run_stream_*`: mock streaming backend; assert correct chunk forwarding; assert history saved on completion
- **Bot-level** — manual / integration test only (Telegram API)

---

## Config Reference (additions)

| Key | Default | Description |
|---|---|---|
| `ollama.think` | `false` | Enable thinking/reasoning mode for Ollama |
| `vllm.think` | `false` | Enable thinking/reasoning mode for vLLM (model-dependent) |
