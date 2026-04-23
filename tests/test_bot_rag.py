import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, Message, Chat, User
from telegram.ext import ContextTypes
from config import Config, TelegramConfig, OllamaConfig, VLLMConfig, HistoryConfig, RagConfig
from rag import RagManager
from bot import cmd_ingest, cmd_collections, handle_message


def make_update(text: str, args: list[str] | None = None):
    user = MagicMock(spec=User)
    chat = MagicMock(spec=Chat)
    chat.id = 42
    chat.send_action = AsyncMock()
    msg = MagicMock(spec=Message)
    msg.text = text
    msg.caption = None
    msg.photo = []
    msg.reply_text = AsyncMock()
    msg.chat = chat
    update = MagicMock(spec=Update)
    update.message = msg
    update.effective_chat = chat
    return update


def make_context(args: list[str] | None = None):
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    ctx.args = args or []
    rag = MagicMock(spec=RagManager)
    rag.list_collections.return_value = [{"name": "rfcs", "count": 42}]
    rag.ingest = AsyncMock(return_value=5)
    ctx.bot_data = {"rag": rag}
    return ctx


def make_handle_context(rag_chunks=None, rag_error=None):
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    ctx.args = []
    
    rag = MagicMock(spec=RagManager)
    if rag_error:
        rag.search = AsyncMock(side_effect=rag_error)
    else:
        rag.search = AsyncMock(return_value=rag_chunks or [])
    
    agent = MagicMock()
    # Mock run_stream as an async generator
    async def mock_run_stream(*args, **kwargs):
        yield ("Test response", None)
    agent.run_stream = mock_run_stream
    
    # Create a mock config with ollama/vllm backends
    cfg = MagicMock(spec=Config)
    cfg.backend = "ollama"
    cfg.ollama = MagicMock(spec=OllamaConfig)
    cfg.ollama.think = False
    cfg.vllm = MagicMock(spec=VLLMConfig)
    cfg.vllm.think = False
    
    ctx.bot_data = {"rag": rag, "agent": agent, "config": cfg}
    return ctx, agent, rag


@pytest.mark.asyncio
async def test_cmd_collections_lists(monkeypatch):
    update = make_update("/collections")
    ctx = make_context()
    await cmd_collections(update, ctx)
    update.message.reply_text.assert_called_once()
    text = update.message.reply_text.call_args.args[0]
    assert "rfcs" in text
    assert "42" in text


@pytest.mark.asyncio
async def test_cmd_collections_empty(monkeypatch):
    update = make_update("/collections")
    ctx = make_context()
    ctx.bot_data["rag"].list_collections.return_value = []
    await cmd_collections(update, ctx)
    text = update.message.reply_text.call_args.args[0]
    assert "No collections" in text


@pytest.mark.asyncio
async def test_cmd_ingest_requires_args(monkeypatch):
    update = make_update("/ingest")
    ctx = make_context(args=[])
    await cmd_ingest(update, ctx)
    text = update.message.reply_text.call_args.args[0]
    assert "Usage" in text


@pytest.mark.asyncio
async def test_cmd_ingest_calls_manager(monkeypatch):
    update = make_update("/ingest rfcs https://example.com/rfc.txt")
    ctx = make_context(args=["rfcs", "https://example.com/rfc.txt"])

    with patch("bot.ingest_source", new=AsyncMock(return_value=7)):
        await cmd_ingest(update, ctx)

    thinking = update.message.reply_text.return_value
    thinking.edit_text.assert_called_once()
    text = thinking.edit_text.call_args.args[0]
    assert "7" in text


@pytest.mark.asyncio
async def test_handle_message_uses_rag_context():
    update = make_update("What is HTTP?")
    update.message.photo = []
    ctx, agent, rag = make_handle_context(rag_chunks=["[source: rfc9110] HTTP is a protocol"])

    await handle_message(update, ctx)

    # Check that RAG search was called with the user message
    rag.search.assert_called_once_with("What is HTTP?")


@pytest.mark.asyncio
async def test_handle_message_degrades_when_rag_fails():
    update = make_update("What is HTTP?")
    update.message.photo = []
    ctx, agent, rag = make_handle_context(rag_error=Exception("ChromaDB unavailable"))

    await handle_message(update, ctx)

    # Should continue despite RAG error
    rag.search.assert_called_once()


@pytest.mark.asyncio
async def test_handle_message_no_context_when_rag_empty():
    update = make_update("What is HTTP?")
    update.message.photo = []
    ctx, agent, rag = make_handle_context(rag_chunks=[])

    await handle_message(update, ctx)

    # RAG search should still be called even if it returns empty
    rag.search.assert_called_once()


@pytest.mark.asyncio
async def test_cmd_ingest_already_ingested():
    update = make_update("/ingest rfcs https://example.com/rfc.txt")
    ctx = make_context(args=["rfcs", "https://example.com/rfc.txt"])

    with patch("bot.ingest_source", new=AsyncMock(return_value=0)):
        await cmd_ingest(update, ctx)

    thinking = update.message.reply_text.return_value
    text = thinking.edit_text.call_args.args[0]
    assert "Already ingested" in text or "empty" in text


@pytest.mark.asyncio
async def test_cmd_ingest_handles_exception():
    update = make_update("/ingest rfcs https://example.com/rfc.txt")
    ctx = make_context(args=["rfcs", "https://example.com/rfc.txt"])

    with patch("bot.ingest_source", new=AsyncMock(side_effect=Exception("Network error"))):
        await cmd_ingest(update, ctx)

    thinking = update.message.reply_text.return_value
    text = thinking.edit_text.call_args.args[0]
    assert "failed" in text.lower() or "⚠️" in text


@pytest.mark.asyncio
async def test_cmd_ingest_single_arg():
    update = make_update("/ingest rfcs")
    ctx = make_context(args=["rfcs"])

    await cmd_ingest(update, ctx)

    text = update.message.reply_text.call_args.args[0]
    assert "Usage" in text
