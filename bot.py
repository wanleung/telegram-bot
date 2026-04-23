"""Telegram bot entry point orchestrating agent interactions with MCP tools."""

import base64
import html
import logging
import os
import time

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config import load_config
from history import init_db, clear_history
from mcp_manager import MCPManager
from agent import Agent
from utils import md_to_telegram_html
from rag import RagManager
from ingest import ingest_source
from llm_backend import create_backend, create_embed_backend

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
        f"Current model: <code>{html.escape(agent.active_model)}</code>\n\n"
        f"Commands:\n"
        f"  /models — list available models\n"
        f"  /model &lt;name&gt; — switch model\n"
        f"  /clear — clear conversation history\n"
        f"  /tools — list available MCP tools\n"
        f"  /think — toggle thinking mode (shows model reasoning)\n"
        f"  /ingest &lt;collection&gt; &lt;url-or-path&gt; — add document to RAG\n"
        f"  /collections — list RAG knowledge base collections",
        parse_mode="HTML",
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


async def cmd_think(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /think command.

    Toggles thinking mode for the current chat. When on, the model's
    reasoning is shown as a spoiler before the answer.
    """
    cfg = context.bot_data["config"]
    think_state: dict[int, bool] = context.bot_data.setdefault("think_state", {})
    chat_id = update.effective_chat.id

    default_think = cfg.ollama.think if cfg.backend == "ollama" else cfg.vllm.think
    current = think_state.get(chat_id, default_think)
    think_state[chat_id] = not current
    status = "ON 🧠" if not current else "OFF"
    await update.message.reply_text(f"Thinking mode {status}")


TELEGRAM_LIMIT = 4096


def _build_reply(content_buf: str, thinking_buf: str, final: bool) -> str:
    """
    Build the Telegram HTML reply string.

    During streaming (final=False): show raw escaped text.
    On final edit: apply markdown→HTML conversion and prepend thinking spoiler.
    Message is guaranteed to fit within Telegram's 4096-character limit.
    """
    if final:
        formatted = md_to_telegram_html(content_buf) if content_buf else "<i>(no response)</i>"
        if thinking_buf:
            # Reserve space for spoiler wrapper overhead (~60 chars) and the content
            max_thinking = TELEGRAM_LIMIT - len(formatted) - 60
            if max_thinking > 0:
                truncated = thinking_buf[:max_thinking]
                if len(thinking_buf) > max_thinking:
                    truncated += "… (truncated)"
                escaped_thinking = html.escape(truncated)
                candidate = f"<tg-spoiler>🤔 Thinking:\n{escaped_thinking}</tg-spoiler>\n\n{formatted}"
                if len(candidate) <= TELEGRAM_LIMIT:
                    return candidate
        # Thinking didn't fit or no thinking — return formatted content, truncated if needed
        return formatted[:TELEGRAM_LIMIT]
    else:
        display = html.escape(content_buf) if content_buf else "⏳"
        if thinking_buf:
            return f"<tg-spoiler>🤔 Thinking…</tg-spoiler>\n\n{display}"
        return display


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle user text messages and photo uploads.

    Searches the RAG knowledge base and injects relevant context into the
    agent prompt. Streams the final LLM response, editing a placeholder
    message every ≥0.5 s as tokens arrive.
    """
    agent: Agent = context.bot_data["agent"]
    rag: RagManager = context.bot_data["rag"]
    cfg = context.bot_data["config"]
    think_state: dict[int, bool] = context.bot_data.setdefault("think_state", {})

    chat_id = update.effective_chat.id
    default_think = cfg.ollama.think if cfg.backend == "ollama" else cfg.vllm.think
    think = think_state.get(chat_id, default_think)

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

    await update.message.chat.send_action("typing")
    placeholder = await update.message.reply_text("⏳")

    content_buf = ""
    thinking_buf = ""
    last_edit: float = 0.0

    try:
        async for content_chunk, thinking_chunk in agent.run_stream(
            chat_id=chat_id,
            user_message=user_text,
            images=images,
            context=rag_context,
            think=think,
        ):
            if thinking_chunk:
                thinking_buf += thinking_chunk
            if content_chunk:
                content_buf += content_chunk

            now = time.monotonic()
            if now - last_edit >= 0.5:
                text = _build_reply(content_buf, thinking_buf if think else "", final=False)
                last_edit = now  # always advance, regardless of edit success
                try:
                    await placeholder.edit_text(text, parse_mode="HTML")
                except Exception:
                    pass  # ignore edit errors during streaming (e.g. message not modified)
    except Exception as exc:
        logger.exception("Streaming error for chat_id=%s: %s", chat_id, exc)
        content_buf += f"\n⚠️ Stream error: {exc}"

    final_text = _build_reply(content_buf, thinking_buf if think else "", final=True)
    try:
        await placeholder.edit_text(final_text, parse_mode="HTML")
    except Exception as exc:
        logger.error("Failed to send final reply for chat_id=%s: %s", chat_id, exc)
        try:
            await placeholder.edit_text(content_buf or "⚠️ (response rendering failed)")
        except Exception:
            pass


async def _post_init(application: Application) -> None:
    """Run async startup tasks after the bot is initialized."""
    cfg = application.bot_data["config"]
    os.makedirs(os.path.dirname(cfg.history.db_path) or ".", exist_ok=True)
    await init_db(cfg.history.db_path)
    mcp: MCPManager = application.bot_data["mcp"]
    await mcp.start()
    embed_backend = create_embed_backend(cfg)
    application.bot_data["rag"] = RagManager(cfg.rag, embed_backend)


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
    chat_backend = create_backend(cfg)
    initial_model = cfg.vllm.default_model if cfg.backend == "vllm" else cfg.ollama.default_model
    agent = Agent(chat_backend, initial_model, cfg, mcp)

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
    app.add_handler(CommandHandler("think", cmd_think))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_message))

    logger.info("Bot starting with model '%s' via %s backend", agent.active_model, cfg.backend)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
