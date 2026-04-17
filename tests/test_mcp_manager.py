import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_manager import MCPManager
from config import MCPServerConfig


def _make_tool(name: str, description: str, schema: dict):
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = schema
    return t


def _make_content(text: str):
    c = MagicMock()
    c.text = text
    return c


async def test_tool_registry_populated():
    servers = {
        "search": MCPServerConfig(type="stdio", command=["echo"], enabled=True)
    }
    mgr = MCPManager(servers)

    mock_session = AsyncMock()
    mock_tool = _make_tool("search_web", "Search the web", {"type": "object", "properties": {}})
    mock_session.list_tools.return_value = MagicMock(tools=[mock_tool])

    with patch.object(mgr, "_connect", return_value=mock_session):
        await mgr.start()

    defs = mgr.get_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "search_web"
    assert defs[0]["function"]["description"] == "Search the web"


async def test_call_tool_dispatches_correctly():
    servers = {
        "search": MCPServerConfig(type="stdio", command=["echo"], enabled=True)
    }
    mgr = MCPManager(servers)

    mock_session = AsyncMock()
    mock_tool = _make_tool("search_web", "Search", {"type": "object"})
    mock_session.list_tools.return_value = MagicMock(tools=[mock_tool])
    mock_session.call_tool.return_value = MagicMock(content=[_make_content("result text")], isError=False)

    with patch.object(mgr, "_connect", return_value=mock_session):
        await mgr.start()

    result = await mgr.call_tool("search_web", {"query": "python"})
    assert result == "result text"
    mock_session.call_tool.assert_called_once_with("search_web", {"query": "python"})


async def test_unknown_tool_returns_error():
    mgr = MCPManager({})
    result = await mgr.call_tool("nonexistent", {})
    assert "unknown tool" in result.lower()


async def test_disabled_server_not_connected():
    servers = {
        "search": MCPServerConfig(type="stdio", command=["echo"], enabled=False)
    }
    mgr = MCPManager(servers)
    with patch.object(mgr, "_connect") as mock_connect:
        await mgr.start()
        mock_connect.assert_not_called()
    assert mgr.get_tool_definitions() == []


async def test_failed_server_is_skipped_gracefully():
    servers = {
        "broken": MCPServerConfig(type="stdio", command=["bad-cmd"], enabled=True)
    }
    mgr = MCPManager(servers)
    with patch.object(mgr, "_connect", side_effect=RuntimeError("connection failed")):
        await mgr.start()  # must not raise

    assert mgr.get_tool_definitions() == []


async def test_list_tools_summary_empty():
    mgr = MCPManager({})
    assert "No MCP tools" in mgr.list_tools_summary()


async def test_list_tools_summary_shows_tools():
    servers = {
        "fs": MCPServerConfig(type="stdio", command=["npx", "fs"], enabled=True)
    }
    mgr = MCPManager(servers)

    mock_session = AsyncMock()
    mock_tool = _make_tool("read_file", "Read a file", {"type": "object"})
    mock_session.list_tools.return_value = MagicMock(tools=[mock_tool])

    with patch.object(mgr, "_connect", return_value=mock_session):
        await mgr.start()

    summary = mgr.list_tools_summary()
    assert "read_file" in summary
    assert "Read a file" in summary


async def test_call_tool_returns_error_on_is_error():
    servers = {
        "search": MCPServerConfig(type="stdio", command=["echo"], enabled=True)
    }
    mgr = MCPManager(servers)

    mock_session = AsyncMock()
    mock_tool = _make_tool("search_web", "Search", {"type": "object"})
    mock_session.list_tools.return_value = MagicMock(tools=[mock_tool])
    mock_session.call_tool.return_value = MagicMock(
        content=[_make_content("boom")], isError=True
    )

    with patch.object(mgr, "_connect", return_value=mock_session):
        await mgr.start()

    result = await mgr.call_tool("search_web", {})
    assert "Error" in result
    assert "boom" in result


async def test_tool_name_collision_logs_warning(caplog):
    import logging
    servers = {
        "server_a": MCPServerConfig(type="stdio", command=["echo"], enabled=True),
        "server_b": MCPServerConfig(type="stdio", command=["echo"], enabled=True),
    }
    mgr = MCPManager(servers)

    mock_session_a = AsyncMock()
    mock_session_b = AsyncMock()
    duplicate_tool = _make_tool("search_web", "Search", {"type": "object"})
    mock_session_a.list_tools.return_value = MagicMock(tools=[duplicate_tool])
    mock_session_b.list_tools.return_value = MagicMock(tools=[duplicate_tool])

    sessions = iter([mock_session_a, mock_session_b])
    with patch.object(mgr, "_connect", side_effect=lambda *a, **k: next(sessions)):
        with caplog.at_level(logging.WARNING, logger="mcp_manager"):
            await mgr.start()

    assert any("overrides" in r.message for r in caplog.records)
