import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent import Agent, MAX_ITERATIONS
from config import Config, TelegramConfig, OllamaConfig, HistoryConfig


def _make_config(tmp_path) -> Config:
    return Config(
        telegram=TelegramConfig(token="tok"),
        ollama=OllamaConfig(default_model="llama3.2"),
        history=HistoryConfig(max_messages=50, db_path=str(tmp_path / "h.db")),
    )


def _make_response(content: str | None = None, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    resp = MagicMock()
    resp.message = msg
    return resp


def _make_tool_call(name: str, arguments: dict):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    tc.model_dump.return_value = {"function": {"name": name, "arguments": arguments}}
    return tc


@pytest.mark.asyncio
async def test_simple_text_response(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    agent = Agent(cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages") as mock_save, \
         patch.object(agent._client, "chat", return_value=_make_response(content="Hello!")):
        result = await agent.run(chat_id=1, user_message="Hi")

    assert result == "Hello!"
    mock_save.assert_called_once()
    saved_messages = mock_save.call_args[0][2]
    assert saved_messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


@pytest.mark.asyncio
async def test_tool_call_then_text_response(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = AsyncMock()
    mcp.get_tool_definitions.return_value = [
        {"type": "function", "function": {"name": "search_web"}}
    ]
    mcp.call_tool.return_value = "Search result: Python docs"
    agent = Agent(cfg, mcp)

    tc = _make_tool_call("search_web", {"query": "python"})
    tool_response = _make_response(content="", tool_calls=[tc])
    final_response = _make_response(content="Here is what I found.")

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"), \
         patch.object(agent._client, "chat", side_effect=[tool_response, final_response]):
        result = await agent.run(chat_id=1, user_message="Search python")

    assert result == "Here is what I found."
    mcp.call_tool.assert_called_once_with("search_web", {"query": "python"})


@pytest.mark.asyncio
async def test_max_iterations_guard(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = AsyncMock()
    mcp.get_tool_definitions.return_value = []
    mcp.call_tool.return_value = "result"
    agent = Agent(cfg, mcp)

    tc = _make_tool_call("loop_tool", {})
    always_tool = _make_response(content="", tool_calls=[tc])

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"), \
         patch.object(agent._client, "chat", return_value=always_tool):
        result = await agent.run(chat_id=1, user_message="loop")

    assert "maximum" in result.lower()


@pytest.mark.asyncio
async def test_ollama_error_returns_error_message(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    agent = Agent(cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"), \
         patch.object(agent._client, "chat", side_effect=Exception("connection refused")):
        result = await agent.run(chat_id=1, user_message="Hi")

    assert "Ollama error" in result


@pytest.mark.asyncio
async def test_history_passed_to_ollama(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    agent = Agent(cfg, mcp)

    existing_history = [
        {"role": "user", "content": "prev question"},
        {"role": "assistant", "content": "prev answer"},
    ]

    with patch("agent.get_history", return_value=existing_history), \
         patch("agent.save_messages"), \
         patch.object(agent._client, "chat", return_value=_make_response(content="ok")) as mock_chat:
        await agent.run(chat_id=1, user_message="new question")

    sent_messages = mock_chat.call_args[1]["messages"]
    assert sent_messages[0] == {"role": "user", "content": "prev question"}
    assert sent_messages[-1] == {"role": "user", "content": "new question"}


def test_set_model_changes_active_model(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    agent = Agent(cfg, mcp)
    assert agent.active_model == "llama3.2"
    agent.set_model("mistral")
    assert agent.active_model == "mistral"
