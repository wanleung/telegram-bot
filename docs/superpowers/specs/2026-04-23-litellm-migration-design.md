# LiteLLM Migration Design

**Date:** 2026-04-23  
**Status:** Approved

## Problem

The bot currently uses the `ollama` Python SDK and `openai` Python SDK directly inside `OllamaBackend` and `VLLMBackend`. This creates two separate code paths with duplicated logic. LiteLLM provides a unified async interface (`acompletion`, `aembedding`) that routes to Ollama, vLLM, OpenAI, Anthropic, and 100+ other providers through the same API.

## Approach

**Thin wrapper — swap SDK calls, keep config unchanged.**

No changes to `config.py`, `bot.py`, or `agent.py`. Users' existing `config.yaml` files continue to work without modification.

## Component Changes

### `requirements.txt`
- Remove: `ollama`, `openai>=1.0`
- Add: `litellm`

### `llm_backend.py`

Replace the internal SDK calls in both backends with `litellm` equivalents. The `LLMBackend` Protocol, `ChatResponse` dataclass, `format_tool_result()`, and factory functions (`create_backend`, `create_embed_backend`) are **unchanged**.

#### Model string convention
- Ollama: `"ollama/<model>"` with `api_base="http://..."`
- vLLM: `"hosted_vllm/<model>"` with `api_base="http://..."`

#### `OllamaBackend`

| Old | New |
|-----|-----|
| `ollama.AsyncClient(host=...).chat(...)` | `litellm.acompletion(model="ollama/<m>", api_base=..., messages=..., tools=...)` |
| Streaming: `client.chat(..., stream=True)` | `litellm.acompletion(..., stream=True)` → `async for chunk` |
| Thinking extraction: `chunk.message.thinking` | `chunk.choices[0].delta.get("thinking", None)` |
| `client.embed(...)` | `litellm.aembedding(model="ollama/<m>", api_base=..., input=...)` |

#### `VLLMBackend`

| Old | New |
|-----|-----|
| `openai.AsyncOpenAI(base_url=...).chat.completions.create(...)` | `litellm.acompletion(model="hosted_vllm/<m>", api_base=..., messages=..., tools=...)` |
| Streaming: `.create(..., stream=True)` | `litellm.acompletion(..., stream=True)` → `async for chunk` |
| Thinking: `extra_body={"enable_thinking": True}` | Same via `extra_body` in LiteLLM |
| Empty-choices guard: `if not chunk.choices: continue` | Same guard kept |
| `.embeddings.create(...)` | `litellm.aembedding(model="hosted_vllm/<m>", api_base=..., input=...)` |

### Tests

Mocks updated from `ollama.AsyncClient` / `openai.AsyncOpenAI` → `litellm.acompletion` / `litellm.aembedding`. All existing test cases remain valid; only the patch targets change.

## Non-Goals

- No changes to `config.py` — `OllamaConfig` and `VLLMConfig` structures stay the same
- No new config blocks or provider strings exposed to the user
- No changes to `bot.py`, `agent.py`, or the `LLMBackend` Protocol
- No new providers beyond Ollama and vLLM (user can extend later)

## Success Criteria

- All existing tests pass after migration
- Streaming and thinking mode continue to work end-to-end
- RAG embed calls work through LiteLLM
- `ollama` and `openai` packages are no longer in `requirements.txt`
