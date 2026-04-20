import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from config import Config, TelegramConfig, OllamaConfig, HistoryConfig, RagConfig
from agent import Agent
from mcp_manager import MCPManager


def make_config():
    return Config(
        telegram=TelegramConfig(token="tok"),
        ollama=OllamaConfig(default_model="llama3.2"),
        history=HistoryConfig(db_path=":memory:"),
        rag=RagConfig(enabled=False),
    )


def _make_response(content: str | None = None, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    resp = MagicMock()
    resp.message = msg
    return resp


@pytest.mark.asyncio
async def test_agent_run_injects_context(tmp_path):
    cfg = make_config()
    cfg.history.db_path = str(tmp_path / "h.db")
    mcp = MagicMock(spec=MCPManager)
    mcp.get_tool_definitions.return_value = []

    agent = Agent(cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"), \
         patch.object(agent._client, "chat", return_value=_make_response(content="42")) as mock_chat:

        result = await agent.run(
            chat_id=1,
            user_message="What is the answer?",
            context="### Context\n[source: test.txt, chunk 0]\nThe answer is 42.",
        )

    assert result == "42"
    call_args = mock_chat.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    # Find the system message
    system_msgs = [m for m in messages if m.get("role") == "system"]
    assert any("### Context" in m.get("content", "") for m in system_msgs)


@pytest.mark.asyncio
async def test_agent_run_no_context_no_system_message(tmp_path):
    cfg = make_config()
    cfg.history.db_path = str(tmp_path / "h.db")
    mcp = MagicMock(spec=MCPManager)
    mcp.get_tool_definitions.return_value = []

    agent = Agent(cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"), \
         patch.object(agent._client, "chat", return_value=_make_response(content="hello")) as mock_chat:

        result = await agent.run(chat_id=1, user_message="hi", context=None)

    assert result == "hello"
    call_args = mock_chat.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    system_msgs = [m for m in messages if m.get("role") == "system"]
    assert not any("### Context" in m.get("content", "") for m in system_msgs)
