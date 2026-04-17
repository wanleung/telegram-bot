import os
import pytest
from pydantic import ValidationError
from config import load_config


def write_cfg(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return str(p)


def test_load_valid_config(tmp_path):
    path = write_cfg(tmp_path, """
telegram:
  token: "test-token"
ollama:
  base_url: "http://localhost:11434"
  default_model: "llama3.2"
  timeout: 60
history:
  max_messages: 20
  db_path: "data/test.db"
mcp_servers: {}
""")
    cfg = load_config(path)
    assert cfg.telegram.token == "test-token"
    assert cfg.ollama.default_model == "llama3.2"
    assert cfg.ollama.timeout == 60
    assert cfg.history.max_messages == 20
    assert cfg.mcp_servers == {}


def test_missing_telegram_token_raises(tmp_path):
    path = write_cfg(tmp_path, """
telegram: {}
ollama:
  default_model: "llama3.2"
""")
    with pytest.raises(ValidationError):
        load_config(path)


def test_missing_ollama_model_raises(tmp_path):
    path = write_cfg(tmp_path, """
telegram:
  token: "tok"
ollama:
  base_url: "http://localhost:11434"
""")
    with pytest.raises(ValidationError):
        load_config(path)


def test_env_var_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "resolved-token")
    path = write_cfg(tmp_path, """
telegram:
  token: "${MY_TOKEN}"
ollama:
  default_model: "llama3.2"
""")
    cfg = load_config(path)
    assert cfg.telegram.token == "resolved-token"


def test_unset_env_var_left_as_is(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    path = write_cfg(tmp_path, """
telegram:
  token: "${MISSING_VAR}"
ollama:
  default_model: "llama3.2"
""")
    cfg = load_config(path)
    assert cfg.telegram.token == "${MISSING_VAR}"


def test_mcp_server_stdio_config(tmp_path):
    path = write_cfg(tmp_path, """
telegram:
  token: "tok"
ollama:
  default_model: "llama3.2"
mcp_servers:
  fs:
    type: stdio
    command: ["npx", "server-fs"]
    enabled: true
""")
    cfg = load_config(path)
    assert cfg.mcp_servers["fs"].type == "stdio"
    assert cfg.mcp_servers["fs"].command == ["npx", "server-fs"]
    assert cfg.mcp_servers["fs"].enabled is True


def test_mcp_server_disabled_by_default_is_false(tmp_path):
    path = write_cfg(tmp_path, """
telegram:
  token: "tok"
ollama:
  default_model: "llama3.2"
mcp_servers:
  remote:
    type: sse
    url: "http://localhost:8080/sse"
    enabled: false
""")
    cfg = load_config(path)
    assert cfg.mcp_servers["remote"].enabled is False


def test_mcp_server_invalid_type_raises(tmp_path):
    path = write_cfg(tmp_path, """
telegram:
  token: "tok"
ollama:
  default_model: "llama3.2"
mcp_servers:
  bad:
    type: grpc
""")
    with pytest.raises(ValidationError):
        load_config(path)


def test_mcp_stdio_without_command_raises(tmp_path):
    path = write_cfg(tmp_path, """
telegram:
  token: "tok"
ollama:
  default_model: "llama3.2"
mcp_servers:
  fs:
    type: stdio
""")
    with pytest.raises(ValidationError):
        load_config(path)


def test_mcp_sse_without_url_raises(tmp_path):
    path = write_cfg(tmp_path, """
telegram:
  token: "tok"
ollama:
  default_model: "llama3.2"
mcp_servers:
  remote:
    type: sse
""")
    with pytest.raises(ValidationError):
        load_config(path)
