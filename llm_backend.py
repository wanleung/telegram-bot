"""LLM backend abstraction: Ollama and vLLM (OpenAI-compat) implementations."""

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
    """LLM backend backed by the Ollama native API."""

    def __init__(self, cfg: OllamaConfig) -> None:
        timeout = httpx.Timeout(connect=10.0, read=cfg.timeout, write=cfg.timeout, pool=10.0)
        self._client = ollama.AsyncClient(host=cfg.base_url, timeout=timeout)

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

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        think: bool = False,
    ) -> AsyncIterator[ChatResponse]:
        """Stream chat responses from Ollama."""
        response = await self._client.chat(
            model=model, messages=messages, tools=tools, stream=True
        )
        async for chunk in response:
            msg = chunk.message
            content = msg.content or ""
            yield ChatResponse(content=content)

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

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        think: bool = False,
    ) -> AsyncIterator[ChatResponse]:
        """Stream chat responses from vLLM."""
        kwargs: dict = {"model": model, "messages": messages, "stream": True}
        if tools:
            kwargs["tools"] = tools
        response = await self._client.chat.completions.create(**kwargs)
        async for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                yield ChatResponse(content=content)

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
