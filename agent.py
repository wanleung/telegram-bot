"""Agent module for orchestrating Ollama LLM interactions with tool support."""

import logging
import ollama

from config import Config
from history import get_history, save_messages
from mcp_manager import MCPManager

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10


class Agent:
    """
    Agent that orchestrates LLM interactions with tool-use capabilities.
    
    The agent maintains conversation history, calls Ollama for responses,
    handles tool invocations through MCP, and returns final text responses.
    """

    def __init__(self, config: Config, mcp_manager: MCPManager) -> None:
        """
        Initialize the Agent with configuration and MCP manager.

        Args:
            config: Configuration object with Ollama and history settings.
            mcp_manager: MCP manager for accessing available tools.
        """
        self._cfg = config
        self._mcp = mcp_manager
        self._client = ollama.AsyncClient(
            host=config.ollama.base_url,
            timeout=config.ollama.timeout,
        )
        self._active_model: str = config.ollama.default_model

    @property
    def active_model(self) -> str:
        """Get the currently active Ollama model name."""
        return self._active_model

    def set_model(self, model: str) -> None:
        """
        Switch to a different Ollama model.

        Args:
            model: The name of the model to switch to.
        """
        self._active_model = model

    async def run(self, chat_id: int, user_message: str) -> str:
        """
        Run the agent for a single user message.

        Fetches conversation history, prepares messages with tool definitions,
        enters a loop to handle tool calls, and returns the final text response.

        Args:
            chat_id: Unique identifier for the conversation.
            user_message: The user's input message.

        Returns:
            The final text response from the LLM, or an error message if
            an exception occurs or max iterations is exceeded.
        """
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
                logger.exception("Ollama chat error for chat_id=%s: %s", chat_id, exc)
                return f"⚠️ Ollama error: {exc}"

            msg = response.message

            if msg.tool_calls:
                # Add assistant response with tool calls to message history
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                })
                
                # Execute each tool call and add results to messages
                for tc in msg.tool_calls:
                    result = await self._mcp.call_tool(
                        tc.function.name, dict(tc.function.arguments)
                    )
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_name": tc.function.name,
                    })
            else:
                # No tool calls, save conversation and return response
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

        warning = "⚠️ Reached maximum tool call iterations. Please try again."
        await save_messages(
            self._cfg.history.db_path,
            chat_id,
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": warning},
            ],
            self._cfg.history.max_messages,
        )
        return warning
