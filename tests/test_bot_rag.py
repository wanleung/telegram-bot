import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, Message, Chat, User
from telegram.ext import ContextTypes
from config import Config, TelegramConfig, OllamaConfig, HistoryConfig, RagConfig
from rag import RagManager
from bot import cmd_ingest, cmd_collections


def make_update(text: str, args: list[str] | None = None):
    user = MagicMock(spec=User)
    chat = MagicMock(spec=Chat)
    chat.id = 42
    msg = MagicMock(spec=Message)
    msg.text = text
    msg.reply_text = AsyncMock()
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
