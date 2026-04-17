"""Telegram bot entry point orchestrating agent interactions with MCP tools."""

import asyncio
import logging
import os

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config import load_config
from history import init_db, clear_history
from mcp_manager import MCPManager
from agent import Agent

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
        f"  /model <name> — switch model\n"
        f"  /clear — clear conversation history\n"
        f"  /tools — list available MCP tools",
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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle user messages.
    
    Runs the agent and streams the response. Shows "thinking" placeholder
    while waiting for the LLM to respond.
    """
    agent: Agent = context.bot_data["agent"]
    thinking = await update.message.reply_text("⏳ Thinking…")
    reply = await agent.run(
        chat_id=update.effective_chat.id,
        user_message=update.message.text,
    )
    await thinking.edit_text(reply or "_(no response)_")


async def main() -> None:
    """
    Initialize and run the Telegram bot.
    
    Loads configuration, initializes database, connects to MCP servers,
    creates the agent, and starts the polling loop.
    """
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    cfg = load_config(config_path)

    os.makedirs(os.path.dirname(cfg.history.db_path) or ".", exist_ok=True)
    await init_db(cfg.history.db_path)

    mcp = MCPManager(cfg.mcp_servers)
    await mcp.start()

    agent = Agent(cfg, mcp)

    app = Application.builder().token(cfg.telegram.token).build()
    app.bot_data["config"] = cfg
    app.bot_data["agent"] = agent
    app.bot_data["mcp"] = mcp

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("tools", cmd_tools))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting with model '%s'", agent.active_model)
    try:
        await app.run_polling(drop_pending_updates=True)
    finally:
        await mcp.stop()


if __name__ == "__main__":
    asyncio.run(main())
