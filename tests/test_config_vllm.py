import pytest
from pydantic import ValidationError
from config import load_config, Config, TelegramConfig, OllamaConfig, VLLMConfig, RagConfig


def _write_cfg(tmp_path, content: str):
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return str(p)


def test_vllm_config_parsed(tmp_path):
    path = _write_cfg(tmp_path,
        "telegram:\n  token: tok\n"
        "backend: vllm\n"
        "vllm:\n  base_url: http://192.168.1.50:8000\n  default_model: llama3\n"
    )
    cfg = load_config(path)
    assert cfg.backend == "vllm"
    assert cfg.vllm.base_url == "http://192.168.1.50:8000"
    assert cfg.vllm.default_model == "llama3"
    assert cfg.vllm.timeout == 300


def test_backend_ollama_requires_ollama_block(tmp_path):
    path = _write_cfg(tmp_path,
        "telegram:\n  token: tok\n"
        "backend: ollama\n"
    )
    with pytest.raises((ValidationError, ValueError)):
        load_config(path)


def test_backend_vllm_requires_vllm_block(tmp_path):
    path = _write_cfg(tmp_path,
        "telegram:\n  token: tok\n"
        "backend: vllm\n"
        "ollama:\n  default_model: llama3.2\n"
    )
    with pytest.raises((ValidationError, ValueError)):
        load_config(path)


def test_embed_backend_vllm_requires_vllm_block(tmp_path):
    path = _write_cfg(tmp_path,
        "telegram:\n  token: tok\n"
        "backend: ollama\n"
        "ollama:\n  default_model: llama3.2\n"
        "rag:\n  embed_backend: vllm\n"
    )
    with pytest.raises((ValidationError, ValueError)):
        load_config(path)


def test_full_vllm_config(tmp_path):
    path = _write_cfg(tmp_path,
        "telegram:\n  token: tok\n"
        "backend: vllm\n"
        "vllm:\n  base_url: http://localhost:8000\n  default_model: llama3\n"
        "rag:\n  embed_backend: vllm\n"
    )
    cfg = load_config(path)
    assert cfg.backend == "vllm"
    assert cfg.rag.embed_backend == "vllm"


def test_embed_backend_defaults_to_ollama(tmp_path):
    path = _write_cfg(tmp_path,
        "telegram:\n  token: tok\n"
        "ollama:\n  default_model: llama3.2\n"
    )
    cfg = load_config(path)
    assert cfg.rag.embed_backend == "ollama"


def test_vllm_config_default_timeout():
    cfg = VLLMConfig(base_url="http://localhost:8000", default_model="llama3")
    assert cfg.timeout == 300


def test_vllm_config_think_default():
    cfg = VLLMConfig(base_url="http://localhost:8000", default_model="llama3")
    assert cfg.think is False


def test_vllm_config_think_enabled():
    cfg = VLLMConfig(base_url="http://localhost:8000", default_model="llama3", think=True)
    assert cfg.think is True
