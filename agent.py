"""Agent module for orchestrating Ollama LLM interactions with tool support."""

import json
import logging
import re
import ollama

from config import Config
from history import get_history, save_messages
from mcp_manager import MCPManager

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10

# Some models emit tool calls as inline text rather than structured API fields.
# Patterns observed: <|python_start|>...<|python_end|>  and  ```json\n...\n```
_TEXT_TOOL_CALL_RE = re.compile(
    r"<\|python_start\|>(.*?)<\|python_end\|>"
    r"|```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)


def _parse_text_tool_calls(content: str) -> list[dict] | None:
    """
    Extract tool call dicts from a model's plain-text tool-call format.

    Returns a list of ``{"name": str, "arguments": dict}`` dicts, or None if
    no tool-call markers were found in *content*.
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

    async def list_models(self) -> list[str]:
        """
        Return names of all locally available Ollama models.

        Returns:
            Sorted list of model name strings, or empty list on error.
        """
        try:
            response = await self._client.list()
            return sorted(m.model for m in response.models)
        except Exception as exc:
            logger.exception("Failed to list Ollama models: %s", exc)
            return []

    async def run(
        self,
        chat_id: int,
        user_message: str,
        images: list[str] | None = None,
    ) -> str:
        """
        Run the agent for a single user message.

        Fetches conversation history, prepares messages with tool definitions,
        enters a loop to handle tool calls, and returns the final text response.

        Args:
            chat_id: Unique identifier for the conversation.
            user_message: The user's input message.
            images: Optional list of base64-encoded images to include with the
                message. Requires a vision-capable model (e.g. llava,
                llama3.2-vision). Stored in history as a text placeholder.

        Returns:
            The final text response from the LLM, or an error message if
            an exception occurs or max iterations is exceeded.
        """
        history = await get_history(
            self._cfg.history.db_path, chat_id, self._cfg.history.max_messages
        )
        tools = self._mcp.get_tool_definitions()

        # Build the outgoing user message; attach images for vision models.
        user_msg: dict = {"role": "user", "content": user_message}
        if images:
            user_msg["images"] = images

        # History placeholder — store caption text only, never raw base64.
        history_user_content = (
            f"[image] {user_message}" if images else user_message
        )

        messages = history + [user_msg]

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
            content = msg.content or ""

            # Prefer structured tool_calls; fall back to text-embedded format.
            if msg.tool_calls:
                tool_calls_to_run = [
                    {"name": tc.function.name, "arguments": dict(tc.function.arguments)}
                    for tc in msg.tool_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                })
            else:
                tool_calls_to_run = _parse_text_tool_calls(content)
                if tool_calls_to_run:
                    # Strip the raw tool-call markup before forwarding to model.
                    clean_content = _TEXT_TOOL_CALL_RE.sub("", content).strip()
                    messages.append({"role": "assistant", "content": clean_content})

            if tool_calls_to_run:
                # Execute each tool call and add results to messages
                for tc in tool_calls_to_run:
                    result = await self._mcp.call_tool(tc["name"], tc["arguments"])
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_name": tc["name"],
                    })
            else:
                # No tool calls — save conversation and return response
                await save_messages(
                    self._cfg.history.db_path,
                    chat_id,
                    [
                        {"role": "user", "content": history_user_content},
                        {"role": "assistant", "content": content},
                    ],
                    self._cfg.history.max_messages,
                )
                return content

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
