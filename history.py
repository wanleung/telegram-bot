"""History module for storing and retrieving conversation messages with SQLite persistence."""

import aiosqlite


async def init_db(db_path: str) -> None:
    """
    Initialize the SQLite database with the messages table and index.
    
    Args:
        db_path: Path to the SQLite database file.
    """
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id   INTEGER NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_chat_id ON messages(chat_id)")
        await db.commit()


async def get_history(db_path: str, chat_id: int, max_messages: int) -> list[dict]:
    """
    Retrieve conversation history for a specific chat.
    
    Returns messages in oldest-first order (suitable for LLM context).
    
    Args:
        db_path: Path to the SQLite database file.
        chat_id: Unique identifier for the conversation.
        max_messages: Maximum number of messages to retrieve.
    
    Returns:
        List of message dictionaries with 'role' and 'content' keys,
        ordered from oldest to newest.
    """
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, max_messages),
        ) as cursor:
            rows = await cursor.fetchall()
    # Reverse so oldest-first order is preserved for the LLM
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]


async def save_messages(
    db_path: str, chat_id: int, messages: list[dict], max_messages: int
) -> None:
    """
    Save messages to the database and trim history to max_messages.
    
    Inserts new messages and removes oldest messages if the total exceeds max_messages.
    
    Args:
        db_path: Path to the SQLite database file.
        chat_id: Unique identifier for the conversation.
        messages: List of message dictionaries with 'role' and 'content' keys.
        max_messages: Maximum number of messages to retain for this chat.
    """
    async with aiosqlite.connect(db_path) as db:
        for msg in messages:
            await db.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, msg["role"], msg["content"]),
            )
        # Trim: keep only the newest max_messages rows for this chat
        await db.execute(
            """
            DELETE FROM messages
            WHERE chat_id = ?
              AND id NOT IN (
                  SELECT id FROM messages
                  WHERE chat_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (chat_id, chat_id, max_messages),
        )
        await db.commit()


async def clear_history(db_path: str, chat_id: int) -> None:
    """
    Clear all messages for a specific chat.
    
    Args:
        db_path: Path to the SQLite database file.
        chat_id: Unique identifier for the conversation to clear.
    """
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        await db.commit()
