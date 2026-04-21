import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent import Agent, MAX_ITERATIONS
from config import Config, TelegramConfig, OllamaConfig, HistoryConfig
from llm_backend import ChatResponse, ToolCall


def _make_config(tmp_path) -> Config:
    return Config(
        telegram=TelegramConfig(token="tok"),
        ollama=OllamaConfig(default_model="llama3.2"),
        history=HistoryConfig(max_messages=50, db_path=str(tmp_path / "h.db")),
    )


def _make_backend(content: str = "Hello!", tool_calls: list | None = None):
    """Return a mock LLMBackend with chat returning the given ChatResponse."""
    backend = MagicMock()
    response = ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        raw_assistant_message={"role": "assistant", "content": content},
    )
    backend.chat = AsyncMock(return_value=response)
    backend.list_models = AsyncMock(return_value=[])
    backend.format_tool_result = MagicMock(
        return_value={"role": "tool", "content": "tool result", "tool_name": "mock_tool"}
    )
    return backend


@pytest.mark.asyncio
async def test_simple_text_response(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    backend = _make_backend(content="Hello!")
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages") as mock_save:
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
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = [
        {"type": "function", "function": {"name": "search_web"}}
    ]
    mcp.call_tool = AsyncMock(return_value="Search result: Python docs")

    tc = ToolCall(name="search_web", arguments={"query": "python"})
    tool_response = ChatResponse(
        content="",
        tool_calls=[tc],
        raw_assistant_message={"role": "assistant", "content": "", "tool_calls": []},
    )
    final_response = ChatResponse(content="Here is what I found.")

    backend = MagicMock()
    backend.chat = AsyncMock(side_effect=[tool_response, final_response])
    backend.format_tool_result = MagicMock(
        return_value={"role": "tool", "content": "Search result: Python docs", "tool_name": "search_web"}
    )
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"), \
         patch("agent.save_messages"):
        result = await agent.run(chat_id=1, user_message="Search python")

    assert result == "Here is what I found."
    mcp.call_tool.assert_called_once_with("search_web", {"query": "python"})

    second_call_messages = backend.chat.call_args_list[1][1]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) >= 1
    assert tool_messages[0]["tool_name"] == "search_web"


@pytest.mark.asyncio
async def test_max_iterations_guard(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    mcp.call_tool = AsyncMock(return_value="result")

    tc = ToolCall(name="loop_tool", arguments={})
    always_tool = ChatResponse(
        content="",
        tool_calls=[tc],
        raw_assistant_message={"role": "assistant", "content": ""},
    )
    backend = MagicMock()
    backend.chat = AsyncMock(return_value=always_tool)
    backend.format_tool_result = MagicMock(
        return_value={"role": "tool", "content": "result", "tool_name": "loop_tool"}
    )
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages") as mock_save:
        result = await agent.run(chat_id=1, user_message="loop")

    assert "maximum" in result.lower()
    mock_save.assert_called_once()
    saved_messages = mock_save.call_args[0][2]
    assert saved_messages[0] == {"role": "user", "content": "loop"}


@pytest.mark.asyncio
async def test_backend_error_returns_error_message(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []

    backend = MagicMock()
    backend.chat = AsyncMock(side_effect=Exception("connection refused"))
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        result = await agent.run(chat_id=1, user_message="Hi")

    assert "error" in result.lower()


@pytest.mark.asyncio
async def test_history_passed_to_backend(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    backend = _make_backend(content="ok")
    agent = Agent(backend, "llama3.2", cfg, mcp)

    existing_history = [
        {"role": "user", "content": "prev question"},
        {"role": "assistant", "content": "prev answer"},
    ]

    with patch("agent.get_history", return_value=existing_history), \
         patch("agent.save_messages"):
        await agent.run(chat_id=1, user_message="new question")

    sent_messages = backend.chat.call_args[1]["messages"]
    assert sent_messages[0] == {"role": "user", "content": "prev question"}
    assert sent_messages[-1] == {"role": "user", "content": "new question"}


def test_set_model_changes_active_model(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    backend = _make_backend()
    agent = Agent(backend, "llama3.2", cfg, mcp)
    assert agent.active_model == "llama3.2"
    agent.set_model("mistral")
    assert agent.active_model == "mistral"


@pytest.mark.asyncio
async def test_list_models_delegates_to_backend(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    backend = MagicMock()
    backend.list_models = AsyncMock(return_value=["llama3.2", "llava"])
    agent = Agent(backend, "llama3.2", cfg, mcp)
    result = await agent.list_models()
    assert result == ["llama3.2", "llava"]


@pytest.mark.asyncio
async def test_image_passed_to_backend(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    backend = _make_backend(content="Nice image!")
    agent = Agent(backend, "llama3.2", cfg, mcp)

    fake_b64 = "aGVsbG8="

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        result = await agent.run(chat_id=1, user_message="What is this?", images=[fake_b64])

    assert result == "Nice image!"
    sent = backend.chat.call_args[1]["messages"]
    user_msg = next(m for m in sent if m["role"] == "user")
    assert user_msg["images"] == [fake_b64]
    assert user_msg["content"] == "What is this?"


@pytest.mark.asyncio
async def test_image_stored_as_placeholder_in_history(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    backend = _make_backend(content="ok")
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages") as mock_save:
        await agent.run(chat_id=1, user_message="describe it", images=["aGVsbG8="])

    saved = mock_save.call_args[0][2]
    assert saved[0]["role"] == "user"
    assert saved[0]["content"] == "[image] describe it"
    assert "aGVsbG8=" not in saved[0]["content"]


@pytest.mark.asyncio
async def test_empty_content_response(tmp_path):
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    backend = _make_backend(content="")
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        result = await agent.run(chat_id=1, user_message="Hi")

    assert result == ""


@pytest.mark.asyncio
async def test_text_tool_call_fallback_executed(tmp_path):
    """Agent executes tool calls present in ChatResponse regardless of their source."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []
    mcp.call_tool = AsyncMock(return_value="All lines good")

    tc = ToolCall(name="get_status", arguments={})
    tool_response = ChatResponse(
        content="",
        tool_calls=[tc],
        raw_assistant_message={"role": "assistant", "content": ""},
    )
    final_response = ChatResponse(content="All lines are running normally.")

    backend = MagicMock()
    backend.chat = AsyncMock(side_effect=[tool_response, final_response])
    backend.format_tool_result = MagicMock(
        return_value={"role": "tool", "content": "All lines good", "tool_name": "get_status"}
    )
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        result = await agent.run(chat_id=1, user_message="list line status")

    assert result == "All lines are running normally."
    mcp.call_tool.assert_called_once_with("get_status", {})
