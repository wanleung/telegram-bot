import os
import re
import yaml
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class TelegramConfig(BaseModel):
    token: str


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    default_model: str
    timeout: int = 300


class VLLMConfig(BaseModel):
    base_url: str = "http://localhost:8000"
    default_model: str
    timeout: int = 300


class HistoryConfig(BaseModel):
    max_messages: int = 50
    db_path: str = "data/history.db"


class RagConfig(BaseModel):
    enabled: bool = False
    embed_backend: Literal["ollama", "vllm"] | None = None
    embed_model: str = Field(default="nomic-embed-text", min_length=1)
    db_path: str = Field(default="data/chroma", min_length=1)
    top_k: int = Field(default=4, gt=0)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


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
    backend: Literal["ollama", "vllm"] = "ollama"
    ollama: OllamaConfig | None = None
    vllm: VLLMConfig | None = None
    history: HistoryConfig = HistoryConfig()
    rag: RagConfig = RagConfig()
    mcp_servers: dict[str, MCPServerConfig] = {}

    @model_validator(mode="after")
    def check_backend_config(self) -> "Config":
        if self.backend == "ollama" and self.ollama is None:
            raise ValueError("backend is 'ollama' but no ollama: block found in config")
        if self.backend == "vllm" and self.vllm is None:
            raise ValueError("backend is 'vllm' but no vllm: block found in config")
        
        # Set embed_backend to main backend if not specified
        if self.rag.embed_backend is None:
            self.rag.embed_backend = self.backend
        
        # Validate embed_backend blocks when explicitly set
        if self.rag.embed_backend == "ollama" and self.ollama is None:
            raise ValueError("rag.embed_backend is 'ollama' but no ollama: block found in config")
        if self.rag.embed_backend == "vllm" and self.vllm is None:
            raise ValueError("rag.embed_backend is 'vllm' but no vllm: block found in config")
        return self


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
