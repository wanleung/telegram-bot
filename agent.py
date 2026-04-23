"""Agent module for orchestrating LLM interactions with tool support."""

import logging
from collections.abc import AsyncIterator

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
