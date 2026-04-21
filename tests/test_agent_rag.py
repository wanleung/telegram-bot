# tests/test_agent_rag.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from config import Config, TelegramConfig, OllamaConfig, HistoryConfig, RagConfig
from agent import Agent
from llm_backend import ChatResponse
from mcp_manager import MCPManager


def make_config():
    return Config(
        telegram=TelegramConfig(token="tok"),
        ollama=OllamaConfig(default_model="llama3.2"),
        history=HistoryConfig(db_path=":memory:"),
        rag=RagConfig(enabled=False),
    )


@pytest.mark.asyncio
async def test_agent_run_injects_context(tmp_path):
    cfg = make_config()
    cfg.history.db_path = str(tmp_path / "h.db")
    mcp = MagicMock(spec=MCPManager)
    mcp.get_tool_definitions.return_value = []

    backend = MagicMock()
    backend.chat = AsyncMock(return_value=ChatResponse(content="42"))
    backend.format_tool_result = MagicMock()
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        result = await agent.run(
            chat_id=1,
            user_message="What is the answer?",
            context="### Context\n[source: test.txt, chunk 0]\nThe answer is 42.",
        )

    assert result == "42"
    messages = backend.chat.call_args[1]["messages"]
    system_msgs = [m for m in messages if m.get("role") == "system"]
    assert any("### Context" in m.get("content", "") for m in system_msgs)


@pytest.mark.asyncio
async def test_agent_run_no_context_no_system_message(tmp_path):
    cfg = make_config()
    cfg.history.db_path = str(tmp_path / "h.db")
    mcp = MagicMock(spec=MCPManager)
    mcp.get_tool_definitions.return_value = []

    backend = MagicMock()
    backend.chat = AsyncMock(return_value=ChatResponse(content="hello"))
    backend.format_tool_result = MagicMock()
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        result = await agent.run(chat_id=1, user_message="hi", context=None)

    assert result == "hello"
    messages = backend.chat.call_args[1]["messages"]
    system_msgs = [m for m in messages if m.get("role") == "system"]
    assert not any("### Context" in m.get("content", "") for m in system_msgs)
