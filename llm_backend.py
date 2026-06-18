"""LLM backend abstraction: Ollama and vLLM via LiteLLM."""

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx
import litellm

from config import Config, LiteLLMProxyConfig, MimoConfig, OllamaConfig, VLLMConfig

litellm.suppress_debug_info = True

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


def _inject_last_user_suffix(
    msgs: list[dict], suffix: str, guard: tuple[str, ...] = ()
) -> None:
    """Append suffix to the last user message unless any guard string is already present."""
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            content = msgs[i].get("content") or ""
            if isinstance(content, str) and not any(s in content for s in guard):
                msgs[i] = {**msgs[i], "content": f"{content} {suffix}".strip()}
            break


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
    thinking: str | None = None


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


class OllamaBackend:
    """LLM backend backed by Ollama via LiteLLM."""

    def __init__(self, cfg: OllamaConfig) -> None:
        self._api_base = cfg.base_url
        self._timeout = cfg.timeout

    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None
    ) -> ChatResponse:
        kwargs: dict = {
            "model": f"ollama_chat/{model}",
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
                    arguments=json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
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
                            "arguments": tc.function.arguments
                            if isinstance(tc.function.arguments, str)
                            else json.dumps(tc.function.arguments),
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
        """Stream chat responses from Ollama with optional thinking support.

        Uses /api/chat (ollama_chat provider) for full tool-calling compatibility.

        Args:
            model: Model name to use.
            messages: Chat messages.
            tools: Ignored for streaming calls; tool detection is handled in agent.py.
            think: Whether to enable thinking mode for models that support it.

        Yields:
            ChatResponse chunks with content and optionally thinking text.
        """
        kwargs: dict = {
            "model": f"ollama_chat/{model}",
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
        """Stream chat responses from vLLM with optional thinking support.

        Args:
            model: Model name to use.
            messages: Chat messages.
            tools: Forwarded to the completions endpoint when provided.
            think: When True, passes extra_body={"enable_thinking": True}.

        Yields:
            ChatResponse chunks with content and optionally thinking text.
        """
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


class LiteLLMProxyBackend:
    """LLM backend that calls a LiteLLM proxy directly via httpx.

    Bypasses LiteLLM SDK routing to call the proxy's /v1/chat/completions
    and /v1/embeddings endpoints directly with full debug visibility.
    """

    def __init__(self, cfg: LiteLLMProxyConfig) -> None:
        self._api_base = cfg.base_url.rstrip("/")
        self._api_key = cfg.api_key
        self._timeout = cfg.timeout

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None
    ) -> ChatResponse:
        msgs = list(messages)
        if tools and "qwen" in model.lower():
            # Qwen3 thinking mode suppresses tool calling — disable it for tool turns.
            _inject_last_user_suffix(msgs, "/no_think", guard=("/no_think", "/think"))

            # Qwen3 benefits from explicit tool guidance in the system prompt.
            tool_names = ", ".join(
                t["function"]["name"] for t in tools if t.get("function", {}).get("name")
            )
            system_prompt = (
                f"You have access to tools: {tool_names}. "
                "When the user's request requires real-time or external data, "
                "call the appropriate tool instead of guessing."
            )
            if not msgs or msgs[0].get("role") != "system":
                msgs = [{"role": "system", "content": system_prompt}] + msgs
            else:
                msgs[0] = {
                    **msgs[0],
                    "content": f"{system_prompt}\n\n{msgs[0]['content']}",
                }

        payload: dict = {"model": model, "messages": msgs}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            logger.debug(
                "Sending %d tool(s) to proxy: %s",
                len(tools),
                [t.get("function", {}).get("name") for t in tools],
            )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._api_base}/v1/chat/completions",
                json=payload,
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices")
        if not choices:
            raise ValueError(f"Proxy returned no choices: {json.dumps(data)[:200]}")
        finish_reason = choices[0].get("finish_reason")
        logger.debug("Proxy response: finish_reason=%s", finish_reason)

        msg = choices[0]["message"]
        content = msg.get("content") or ""
        tool_calls_raw = msg.get("tool_calls")

        if tool_calls_raw:
            tool_calls = [
                ToolCall(
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"])
                    if isinstance(tc["function"]["arguments"], str)
                    else tc["function"]["arguments"],
                    id=tc.get("id"),
                )
                for tc in tool_calls_raw
            ]
            logger.debug("Model called tools: %s", [tc.name for tc in tool_calls])
            return ChatResponse(
                content=content, tool_calls=tool_calls, raw_assistant_message=msg
            )

        return ChatResponse(content=content, raw_assistant_message=msg)

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        think: bool = False,
    ) -> AsyncIterator[ChatResponse]:
        """Stream chat responses from a LiteLLM proxy.

        Args:
            model: Model name as configured in the proxy.
            messages: Chat messages.
            tools: Forwarded to the completions endpoint when provided.
            think: When True, injects /think into the last user message for
                   Qwen3-style thinking mode.

        Yields:
            ChatResponse chunks with content and optionally thinking text.
        """
        msgs = list(messages)
        if think:
            _inject_last_user_suffix(msgs, "/think", guard=("/think",))

        payload: dict = {"model": model, "messages": msgs, "stream": True}
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._api_base}/v1/chat/completions",
                json=payload,
                headers=self._headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                        delta = chunk["choices"][0]["delta"]
                        content = delta.get("content") or ""
                        thinking = (
                            delta.get("thinking")
                            or delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or None
                        )
                        if content or thinking:
                            yield ChatResponse(content=content, thinking=thinking)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._api_base}/v1/models",
                    headers=self._headers,
                )
                resp.raise_for_status()
                data = resp.json()
                return sorted(m["id"] for m in data.get("data", []))
        except Exception as exc:
            logger.exception("Failed to list LiteLLM proxy models: %s", exc)
            return []

    async def embed(self, model: str, text: str) -> list[float]:
        response = await litellm.aembedding(
            model=f"openai/{model}",
            api_base=self._api_base,
            api_key=self._api_key,
            input=text,
            timeout=self._timeout,
        )
        return response.data[0].embedding

    def format_tool_result(self, tool_call: ToolCall, result: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call.id or "", "content": result}


class MimoBackend(LiteLLMProxyBackend):
    """LLM backend for the Mimo cloud API (OpenAI-compatible)."""

    def __init__(self, cfg: MimoConfig) -> None:
        base = cfg.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        self._api_base = base
        self._api_key = cfg.api_key
        self._timeout = cfg.timeout


def create_backend(config: Config) -> LLMBackend:
    """Return the chat LLMBackend configured in config.backend."""
    if config.backend == "vllm":
        return VLLMBackend(config.vllm)
    if config.backend == "litellm_proxy":
        return LiteLLMProxyBackend(config.litellm_proxy)
    if config.backend == "mimo":
        return MimoBackend(config.mimo)
    return OllamaBackend(config.ollama)


def create_embed_backend(config: Config) -> LLMBackend:
    """Return the embedding LLMBackend configured in config.rag.embed_backend."""
    if config.rag.embed_backend == "vllm":
        return VLLMBackend(config.vllm)
    if config.rag.embed_backend == "litellm_proxy":
        return LiteLLMProxyBackend(config.litellm_proxy)
    if config.rag.embed_backend == "mimo":
        return MimoBackend(config.mimo)
    return OllamaBackend(config.ollama)
