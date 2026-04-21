# vLLM Backend Support — Design Spec

**Date:** 2026-04-21  
**Status:** Approved

---

## Problem

The bot currently uses the `ollama` Python SDK directly in `agent.py` and `rag.py`. This couples the codebase to Ollama and makes it impossible to use vLLM (which exposes an OpenAI-compatible API) as the chat or embedding backend.

---

## Goal

Allow users to point the bot at either a local Ollama instance or a local/network vLLM instance by changing one line in `config.yaml`. No code changes required to switch backends.

---

## Architecture

### Backend Abstraction (`llm_backend.py`)

A new `LLMBackend` Protocol defines the interface both backends must satisfy:

```python
class LLMBackend(Protocol):
    async def chat(self, model: str, messages: list[dict], tools: list[dict] | None) -> ChatResponse: ...
    async def list_models(self) -> list[str]: ...
    async def embed(self, model: str, text: str) -> list[float]: ...
```

`ChatResponse` is a simple dataclass:

```python
@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall]  # normalised, backend-agnostic

@dataclass
class ToolCall:
    name: str
    arguments: dict
```

Both backends normalise their responses to `ChatResponse` before returning. This means `agent.py` never needs to know which backend is active.

### `OllamaBackend`

Wraps `ollama.AsyncClient`. Converts `ollama.Message` and `ollama.ToolCall` to `ChatResponse`. Retains the existing `<|python_start|>` text tool-call fallback parsing (moved from `agent.py` into the backend).

### `VLLMBackend`

Wraps `openai.AsyncOpenAI(base_url=cfg.vllm.base_url, api_key="none")`. Converts `openai.ChatCompletion` to `ChatResponse`. Tool calls use the OpenAI structured format — no text-embedded fallback needed.

`list_models()` calls `GET /v1/models` and returns model IDs.  
`embed()` calls `POST /v1/embeddings` and returns the embedding vector.

### Factory function

```python
def create_backend(config: Config) -> LLMBackend:
    if config.backend == "vllm":
        return VLLMBackend(config.vllm)
    return OllamaBackend(config.ollama)

def create_embed_backend(config: Config) -> LLMBackend:
    if config.rag.embed_backend == "vllm":
        return VLLMBackend(config.vllm)
    return OllamaBackend(config.ollama)
```

---

## Config Changes

### `config.py`

New model:

```python
class VLLMConfig(BaseModel):
    base_url: str = "http://localhost:8000"
    default_model: str
    timeout: int = 120
```

Updated `Config`:

```python
class Config(BaseModel):
    ...
    backend: Literal["ollama", "vllm"] = "ollama"
    vllm: VLLMConfig | None = None
    ...
```

Validators:
- If `backend == "vllm"` and `vllm` is `None`, raise `ValueError("backend is 'vllm' but no vllm: block found in config")`.
- If `rag.embed_backend == "vllm"` and `vllm` is `None`, raise `ValueError("rag.embed_backend is 'vllm' but no vllm: block found in config")`. This validator lives on `Config` (cross-field).

Updated `RagConfig`:

```python
class RagConfig(BaseModel):
    ...
    embed_backend: Literal["ollama", "vllm"] = "ollama"
```

### `config.example.yaml`

```yaml
backend: ollama   # or vllm

vllm:
  base_url: http://localhost:8000          # or http://192.168.1.50:8000
  default_model: meta-llama/Llama-3.2-3B-Instruct
  timeout: 120

rag:
  embed_backend: ollama   # or vllm
```

---

## Changes to Existing Files

### `agent.py`

- Remove `import ollama` and `ollama.AsyncClient` instantiation
- Constructor accepts `LLMBackend` and `initial_model: str` instead of creating its own client. `bot.py` passes `config.ollama.default_model` or `config.vllm.default_model` depending on active backend.
- `run()` calls `backend.chat()` and consumes `ChatResponse` (no Ollama-specific field access)
- `list_models()` delegates to `backend.list_models()`
- Text tool-call fallback parsing moves into `OllamaBackend.chat()`

### `rag.py`

- Constructor accepts a `LLMBackend` for embeddings
- `_embed()` calls `embed_backend.embed(model, text)` instead of `ollama.AsyncClient().embed()`

### `bot.py`

- In `_post_init`: call `create_backend(config)` and `create_embed_backend(config)`
- Pass chat backend to `Agent`, embed backend to `RagManager`

---

## Data Flow

```
config.yaml
    └── backend: vllm
            │
            ▼
    create_backend(config) → VLLMBackend
            │
            ▼
    Agent.run(chat_id, message)
            │
            ▼
    VLLMBackend.chat(model, messages, tools)
            │   openai SDK → vLLM /v1/chat/completions
            ▼
    ChatResponse(content, tool_calls)
            │
            ▼
    MCPManager.call_tool(name, args)  [if tool_calls present]
```

> **Note:** `VLLMBackend.embed()` requires vLLM to be serving an embedding-capable model. Not all vLLM deployments include one. If `embed_backend: vllm` is set but vLLM returns an error for the embed request, `RagManager` will log the error and return empty results (graceful degradation, same as today).



- `VLLMBackend` catches `openai.APIError` and re-raises as a plain `RuntimeError` with a human-readable message. `Agent` catches all exceptions and returns `"⚠️ Backend error: ..."` to the user — same as today.
- If `backend: vllm` but `vllm:` block is missing from config, `Config` validation raises at startup with a clear message.
- `list_models()` returns `[]` on error (same as current Ollama behaviour).

---

## Testing

- `tests/test_llm_backend.py` — unit tests for both backends using mocks:
  - `OllamaBackend.chat()` returns `ChatResponse` with correct tool_calls
  - `OllamaBackend.chat()` falls back to text tool-call parsing
  - `VLLMBackend.chat()` normalises OpenAI response to `ChatResponse`
  - `VLLMBackend.embed()` returns correct vector
  - `VLLMBackend.list_models()` returns model IDs
- `tests/test_config_vllm.py` — validates new config fields and the missing-vllm-block validator
- Existing `tests/test_agent.py` updated to inject a mock `LLMBackend` instead of mocking the ollama client directly

---

## Dependencies

- `openai>=1.0` added to `requirements.txt` (already likely installed transitively, but made explicit)

---

## Out of Scope

- Runtime backend switching via bot command
- Multi-backend fan-out (query both simultaneously)
- vLLM streaming responses
- API key / auth support (local-only use case)
