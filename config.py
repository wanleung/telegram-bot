import os
import re
import yaml
from typing import Literal
from pydantic import BaseModel, model_validator


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
    type: Literal["stdio", "sse", "http"]
    command: list[str] | None = None
    url: str | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def check_fields_for_type(self) -> "MCPServerConfig":
        if self.type == "stdio" and not self.command:
            raise ValueError("MCP server type 'stdio' requires 'command'")
        if self.type in ("sse", "http") and not self.url:
            raise ValueError(f"MCP server type '{self.type}' requires 'url'")
        return self


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
