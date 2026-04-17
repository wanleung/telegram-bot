import pytest
from history import init_db, get_history, save_messages, clear_history


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


async def test_empty_history(db_path):
    await init_db(db_path)
    result = await get_history(db_path, chat_id=1, max_messages=50)
    assert result == []


async def test_save_and_retrieve(db_path):
    await init_db(db_path)
    await save_messages(db_path, chat_id=1, messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ], max_messages=50)
    history = await get_history(db_path, chat_id=1, max_messages=50)
    assert history == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


async def test_trim_at_max_messages(db_path):
    await init_db(db_path)
    # Save 3 batches of 2 messages each (6 total), max is 4
    for i in range(3):
        await save_messages(db_path, chat_id=1, messages=[
            {"role": "user", "content": f"user {i}"},
            {"role": "assistant", "content": f"reply {i}"},
        ], max_messages=4)
    history = await get_history(db_path, chat_id=1, max_messages=4)
    assert len(history) == 4
    # Oldest messages trimmed, newest retained
    assert history[-1] == {"role": "assistant", "content": "reply 2"}


async def test_clear_history(db_path):
    await init_db(db_path)
    await save_messages(db_path, chat_id=1, messages=[
        {"role": "user", "content": "Hi"},
    ], max_messages=50)
    await clear_history(db_path, chat_id=1)
    result = await get_history(db_path, chat_id=1, max_messages=50)
    assert result == []


async def test_separate_chat_ids_isolated(db_path):
    await init_db(db_path)
    await save_messages(db_path, chat_id=1, messages=[
        {"role": "user", "content": "chat one"},
    ], max_messages=50)
    await save_messages(db_path, chat_id=2, messages=[
        {"role": "user", "content": "chat two"},
    ], max_messages=50)
    h1 = await get_history(db_path, chat_id=1, max_messages=50)
    h2 = await get_history(db_path, chat_id=2, max_messages=50)
    assert h1 == [{"role": "user", "content": "chat one"}]
    assert h2 == [{"role": "user", "content": "chat two"}]


async def test_clear_does_not_affect_other_chats(db_path):
    await init_db(db_path)
    await save_messages(db_path, chat_id=1, messages=[
        {"role": "user", "content": "keep me"},
    ], max_messages=50)
    await save_messages(db_path, chat_id=2, messages=[
        {"role": "user", "content": "delete me"},
    ], max_messages=50)
    await clear_history(db_path, chat_id=2)
    h1 = await get_history(db_path, chat_id=1, max_messages=50)
    h2 = await get_history(db_path, chat_id=2, max_messages=50)
    assert h1 == [{"role": "user", "content": "keep me"}]
    assert h2 == []
