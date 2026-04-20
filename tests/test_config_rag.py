import pytest
from pydantic import ValidationError
from config import load_config, RagConfig


def test_rag_config_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "telegram:\n  token: tok\n"
        "ollama:\n  default_model: llama3.2\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg.rag.enabled is False
    assert cfg.rag.embed_model == "nomic-embed-text"
    assert cfg.rag.db_path == "data/chroma"
    assert cfg.rag.top_k == 4
    assert cfg.rag.similarity_threshold == 0.5


def test_rag_config_custom(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "telegram:\n  token: tok\n"
        "ollama:\n  default_model: llama3.2\n"
        "rag:\n  enabled: true\n  embed_model: mxbai-embed-large\n  top_k: 6\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg.rag.enabled is True
    assert cfg.rag.embed_model == "mxbai-embed-large"
    assert cfg.rag.top_k == 6
    assert cfg.rag.db_path == "data/chroma"
    assert cfg.rag.similarity_threshold == 0.5


def test_rag_config_rejects_invalid_top_k():
    with pytest.raises(ValidationError):
        RagConfig(top_k=0)


def test_rag_config_rejects_invalid_threshold():
    with pytest.raises(ValidationError):
        RagConfig(similarity_threshold=1.5)
