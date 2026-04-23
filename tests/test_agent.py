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


def _make_streaming_backend(chunks: list[tuple[str, str | None]]):
    """
    Return a mock LLMBackend whose chat_stream yields ChatResponse chunks.
    chunks is a list of (content, thinking) pairs.
    """
    backend = MagicMock()
    # non-streaming chat returns no tool calls (so run_stream goes straight to streaming)
    backend.chat = AsyncMock(return_value=ChatResponse(content="", tool_calls=[]))

    async def _stream(*args, **kwargs):
        for content, thinking in chunks:
            yield ChatResponse(content=content, thinking=thinking)

    backend.chat_stream = _stream
    backend.format_tool_result = MagicMock(
        return_value={"role": "tool", "content": "result", "tool_name": "mock"}
    )
    return backend


@pytest.mark.asyncio
async def test_run_stream_yields_content_chunks(tmp_path):
    """run_stream yields (content, None) tuples for plain text responses."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []

    backend = _make_streaming_backend([("Hello", None), (" world", None)])
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        results = []
        async for content_chunk, thinking_chunk in agent.run_stream(
            chat_id=1, user_message="Hi"
        ):
            results.append((content_chunk, thinking_chunk))

    assert results == [("Hello", None), (" world", None)]


@pytest.mark.asyncio
async def test_run_stream_yields_thinking_chunks(tmp_path):
    """run_stream yields ("", thinking) tuples for thinking fragments."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []

    backend = _make_streaming_backend([("", "I think..."), ("Answer", None)])
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        results = []
        async for content_chunk, thinking_chunk in agent.run_stream(
            chat_id=1, user_message="Hi", think=True
        ):
            results.append((content_chunk, thinking_chunk))

    assert ("", "I think...") in results
    assert ("Answer", None) in results


@pytest.mark.asyncio
async def test_run_stream_saves_history(tmp_path):
    """run_stream saves accumulated content to history after streaming completes."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []

    backend = _make_streaming_backend([("Hello", None), (" there", None)])
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages") as mock_save:
        async for _ in agent.run_stream(chat_id=1, user_message="Hi"):
            pass

    mock_save.assert_called_once()
    saved = mock_save.call_args[0][2]
    assert saved[0] == {"role": "user", "content": "Hi"}
    assert saved[1] == {"role": "assistant", "content": "Hello there"}


@pytest.mark.asyncio
async def test_run_stream_tool_then_stream(tmp_path):
    """run_stream uses non-streaming chat for tool calls, then streams final response."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = [{"type": "function", "function": {"name": "lookup"}}]
    mcp.call_tool = AsyncMock(return_value="tool result")

    tc = ToolCall(name="lookup", arguments={"q": "x"})
    tool_response = ChatResponse(
        content="",
        tool_calls=[tc],
        raw_assistant_message={"role": "assistant", "content": ""},
    )
    # After tool call, non-streaming chat returns no tool calls → triggers streaming
    no_tool_response = ChatResponse(content="", tool_calls=[])

    captured = {}

    async def _stream(*args, **kwargs):
        captured["messages"] = kwargs.get("messages", args[1] if len(args) > 1 else [])
        yield ChatResponse(content="Final answer", thinking=None)

    backend = MagicMock()
    backend.chat = AsyncMock(side_effect=[tool_response, no_tool_response])
    backend.chat_stream = _stream
    backend.format_tool_result = MagicMock(
        return_value={"role": "tool", "content": "tool result", "tool_name": "lookup"}
    )
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        results = []
        async for chunk in agent.run_stream(chat_id=1, user_message="look up x"):
            results.append(chunk)

    assert any(content == "Final answer" for content, _ in results)
    mcp.call_tool.assert_called_once_with("lookup", {"q": "x"})
    tool_msgs = [m for m in captured.get("messages", []) if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].get("tool_name") == "lookup"


@pytest.mark.asyncio
async def test_run_stream_backend_error(tmp_path):
    """run_stream yields an error tuple when the streaming backend raises."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = []

    backend = MagicMock()
    backend.chat = AsyncMock(return_value=ChatResponse(content="", tool_calls=[]))

    async def _stream(*args, **kwargs):
        raise RuntimeError("connection dropped")
        yield  # make it an async generator

    backend.chat_stream = _stream
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), \
         patch("agent.save_messages"):
        results = []
        async for chunk in agent.run_stream(chat_id=1, user_message="Hi"):
            results.append(chunk)

    combined = "".join(c for c, _ in results)
    assert "error" in combined.lower() or "interrupted" in combined.lower()


@pytest.mark.asyncio
async def test_run_stream_non_streaming_error_in_tool_loop(tmp_path):
    """run_stream yields an error tuple when chat() raises during the tool-call loop."""
    cfg = _make_config(tmp_path)
    mcp = MagicMock()
    mcp.get_tool_definitions.return_value = [{"type": "function", "function": {"name": "t"}}]

    backend = MagicMock()
    backend.chat = AsyncMock(side_effect=RuntimeError("backend unavailable"))
    agent = Agent(backend, "llama3.2", cfg, mcp)

    with patch("agent.get_history", return_value=[]), patch("agent.save_messages"):
        results = []
        async for chunk in agent.run_stream(chat_id=1, user_message="Hi"):
            results.append(chunk)

    combined = "".join(c for c, _ in results)
    assert "error" in combined.lower()
