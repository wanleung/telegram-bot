"""Telegram bot entry point orchestrating agent interactions with MCP tools."""

import base64
import logging
import os

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config import load_config
from history import init_db, clear_history
from mcp_manager import MCPManager
from agent import Agent
from utils import md_to_telegram_html
from rag import RagManager
from ingest import ingest_source

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /start command.
    
    Displays welcome message with current model and available commands.
    """
    agent: Agent = context.bot_data["agent"]
    await update.message.reply_text(
        f"👋 Hello! I'm your Ollama-powered assistant.\n"
        f"Current model: `{agent.active_model}`\n\n"
        f"Commands:\n"
        f"  /models — list available models\n"
        f"  /model <name> — switch model\n"
        f"  /clear — clear conversation history\n"
        f"  /tools — list available MCP tools\n"
        f"  /ingest <collection> <url-or-path> — add document to RAG\n"
        f"  /collections — list RAG knowledge base collections",
        parse_mode="Markdown",
    )


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /model command.
    
    Display current model or switch to a new one if model name is provided.
    """
    agent: Agent = context.bot_data["agent"]
    if not context.args:
        await update.message.reply_text(
            f"Current model: `{agent.active_model}`\nUsage: /model <name>",
            parse_mode="Markdown",
        )
        return
    new_model = context.args[0]
    agent.set_model(new_model)
    await update.message.reply_text(f"✅ Switched to model `{new_model}`", parse_mode="Markdown")


async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /models command.

    List all locally available Ollama models by querying the Ollama API.
    """
    agent: Agent = context.bot_data["agent"]
    models = await agent.list_models()
    if not models:
        await update.message.reply_text("⚠️ No models found (is Ollama running?)")
        return
    lines = "\n".join(
        f"{'▶️' if m == agent.active_model else '  •'} `{m}`" for m in models
    )
    await update.message.reply_text(f"*Available models:*\n{lines}", parse_mode="Markdown")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /clear command.

    Clears the conversation history for the current chat.
    """
    cfg = context.bot_data["config"]
    await clear_history(cfg.history.db_path, update.effective_chat.id)
    await update.message.reply_text("🗑️ Conversation history cleared.")


async def cmd_tools(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /tools command.
    
    Display a formatted list of all available MCP tools.
    """
    mcp: MCPManager = context.bot_data["mcp"]
    await update.message.reply_text(mcp.list_tools_summary(), parse_mode="Markdown")


async def cmd_ingest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ingest <collection> <url-or-path> — ingest a document into RAG."""
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /ingest <collection> <url-or-path>\n"
            "Example: /ingest rfcs https://rfc-editor.org/rfc/rfc9110.txt"
        )
        return
    collection = context.args[0]
    source = context.args[1]
    rag: RagManager = context.bot_data["rag"]
    thinking = await update.message.reply_text(f"⏳ Ingesting {source}…")
    try:
        count = await ingest_source(rag, collection, source)
        if count == 0:
            await thinking.edit_text(f"ℹ️ Already ingested or empty: {source}")
        else:
            await thinking.edit_text(f"✅ Ingested {count} chunks into '{collection}'.")
    except Exception as exc:
        await thinking.edit_text(f"⚠️ Ingest failed: {exc}")


async def cmd_collections(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /collections — list all RAG collections with chunk counts."""
    rag: RagManager = context.bot_data["rag"]
    cols = rag.list_collections()
    if not cols:
        await update.message.reply_text("No collections found. Use /ingest to add documents.")
        return
    lines = "\n".join(f"  • <b>{c['name']}</b> — {c['count']} chunks" for c in cols)
    await update.message.reply_text(f"<b>RAG Collections:</b>\n{lines}", parse_mode="HTML")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle user text messages and photo uploads.

    Searches the RAG knowledge base and injects relevant context into the
    agent prompt before calling Ollama.
    """
    agent: Agent = context.bot_data["agent"]
    rag: RagManager = context.bot_data["rag"]
    thinking = await update.message.reply_text("⏳ Thinking…")

    images: list[str] | None = None
    user_text = update.message.text or update.message.caption or ""

    if update.message.photo:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        image_bytes = await file.download_as_bytearray()
        images = [base64.b64encode(image_bytes).decode()]

    # RAG: retrieve relevant chunks and format as context block
    rag_context: str | None = None
    if user_text.strip():
        try:
            chunks = await rag.search(user_text)
            if chunks:
                rag_context = "### Context\n" + "\n\n".join(chunks)
        except Exception as exc:
            logger.warning("RAG search failed, continuing without context: %s", exc)

    reply = await agent.run(
        chat_id=update.effective_chat.id,
        user_message=user_text,
        images=images,
        context=rag_context,
    )
    formatted = md_to_telegram_html(reply) if reply else "_(no response)_"
    parse_mode = "HTML" if reply else "Markdown"
    await thinking.edit_text(formatted, parse_mode=parse_mode)


async def _post_init(application: Application) -> None:
    """Run async startup tasks after the bot is initialized."""
    cfg = application.bot_data["config"]
    os.makedirs(os.path.dirname(cfg.history.db_path) or ".", exist_ok=True)
    await init_db(cfg.history.db_path)
    mcp: MCPManager = application.bot_data["mcp"]
    await mcp.start()
    application.bot_data["rag"] = RagManager(cfg.rag)


async def _post_shutdown(application: Application) -> None:
    """Run async cleanup tasks after polling stops."""
    await application.bot_data["mcp"].stop()


def main() -> None:
    """
    Initialize and run the Telegram bot.

    Loads configuration, registers PTB lifecycle hooks for async startup/cleanup,
    registers command and message handlers, and starts the polling loop.
    """
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    cfg = load_config(config_path)

    mcp = MCPManager(cfg.mcp_servers)
    agent = Agent(cfg, mcp)

    app = (
        Application.builder()
        .token(cfg.telegram.token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    app.bot_data["config"] = cfg
    app.bot_data["agent"] = agent
    app.bot_data["mcp"] = mcp

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("tools", cmd_tools))
    app.add_handler(CommandHandler("ingest", cmd_ingest))
    app.add_handler(CommandHandler("collections", cmd_collections))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_message))

    logger.info("Bot starting with model '%s'", agent.active_model)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
