import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from config import RagConfig
from rag import RagManager


@pytest.fixture
def rag_cfg(tmp_path):
    return RagConfig(
        enabled=True,
        embed_model="nomic-embed-text",
        db_path=str(tmp_path / "chroma"),
        top_k=2,
        similarity_threshold=0.0,
    )


@pytest.fixture
def mock_embed_backend():
    backend = MagicMock()
    backend.embed = AsyncMock(return_value=[0.1] * 768)
    return backend


@pytest.fixture
def mock_chroma(monkeypatch):
    """Patch chromadb.PersistentClient to return a mock."""
    col = MagicMock()
    col.count.return_value = 0
    col.get.return_value = {"ids": []}
    col.query.return_value = {
        "ids": [["id1"]],
        "documents": [["chunk text"]],
        "metadatas": [[{"source": "test.txt", "chunk_index": 0}]],
        "distances": [[0.1]],
    }
    client = MagicMock()
    client.get_or_create_collection.return_value = col
    client.list_collections.return_value = [MagicMock(name="test")]
    monkeypatch.setattr("rag.chromadb.PersistentClient", lambda path: client)
    return client, col


@pytest.mark.asyncio
async def test_ingest_text(rag_cfg, mock_chroma, mock_embed_backend):
    client, col = mock_chroma
    manager = RagManager(rag_cfg, mock_embed_backend)

    with patch.object(manager, "_embed", new=AsyncMock(return_value=[0.1] * 768)):
        count = await manager.ingest("mycol", "test.txt", "Hello world " * 100)

    assert count > 0
    assert col.add.called


@pytest.mark.asyncio
async def test_ingest_skips_duplicate(rag_cfg, mock_chroma, mock_embed_backend):
    client, col = mock_chroma
    col.get.return_value = {"ids": ["existing-id"]}
    manager = RagManager(rag_cfg, mock_embed_backend)

    with patch.object(manager, "_embed", new=AsyncMock(return_value=[0.1] * 768)):
        count = await manager.ingest("mycol", "test.txt", "Hello world " * 100)

    assert count == 0
    col.add.assert_not_called()


@pytest.mark.asyncio
async def test_search_returns_chunks(rag_cfg, mock_chroma, mock_embed_backend):
    client, col = mock_chroma
    manager = RagManager(rag_cfg, mock_embed_backend)

    with patch.object(manager, "_embed", new=AsyncMock(return_value=[0.1] * 768)):
        chunks = await manager.search("who am I?")

    assert len(chunks) == 1
    assert "chunk text" in chunks[0]
    assert "test.txt" in chunks[0]


@pytest.mark.asyncio
async def test_search_empty_when_disabled(tmp_path, mock_embed_backend):
    cfg = RagConfig(enabled=False, db_path=str(tmp_path / "chroma"))
    manager = RagManager(cfg, mock_embed_backend)
    chunks = await manager.search("anything")
    assert chunks == []


def test_chunk_text_splits_correctly():
    from rag import chunk_text
    text = "a" * 1200
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) == 3
    assert chunks[0] == "a" * 500
    assert chunks[1] == "a" * 500
    assert len(chunks[2]) > 0


def test_chunk_text_short_input():
    from rag import chunk_text
    chunks = chunk_text("hello", size=500, overlap=50)
    assert chunks == ["hello"]


def test_chunk_text_empty():
    from rag import chunk_text
    chunks = chunk_text("", size=500, overlap=50)
    assert chunks == []


def test_list_collections(rag_cfg, mock_chroma, mock_embed_backend):
    client, col = mock_chroma
    manager = RagManager(rag_cfg, mock_embed_backend)
    cols = manager.list_collections()
    assert isinstance(cols, list)
