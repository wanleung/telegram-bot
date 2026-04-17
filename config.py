import os
import re
import yaml
from pydantic import BaseModel


class TelegramConfig(BaseModel):
    token: str


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    default_model: str
    timeout: int = 120


class HistoryConfig(BaseModel):
    max_messages: int = 50
    db_path: str = "data/history.db"


class MCPServerConfig(BaseModel):
    type: str  # "stdio", "sse", "http"
    command: list[str] | None = None
    url: str | None = None
    enabled: bool = True


class Config(BaseModel):
    telegram: TelegramConfig
    ollama: OllamaConfig
    history: HistoryConfig = HistoryConfig()
    mcp_servers: dict[str, MCPServerConfig] = {}


def _resolve(obj: object) -> object:
    """Recursively resolve ${VAR} references in string values."""
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(i) for i in obj]
    return obj


def load_config(path: str = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(_resolve(raw))
