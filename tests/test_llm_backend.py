import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from config import Config, TelegramConfig, OllamaConfig, VLLMConfig, RagConfig
from llm_backend import (
    ChatResponse, ToolCall, OllamaBackend, VLLMBackend,
    create_backend, create_embed_backend, _parse_text_tool_calls,
)


def _ollama_cfg():
    return OllamaConfig(base_url="http://localhost:11434", default_model="llama3.2")


def _vllm_cfg():
    return VLLMConfig(base_url="http://localhost:8000", default_model="llama3")


def _make_config(backend="ollama", embed_backend="ollama"):
    return Config(
        telegram=TelegramConfig(token="tok"),
        backend=backend,
        ollama=_ollama_cfg() if (backend == "ollama" or embed_backend == "ollama") else None,
        vllm=_vllm_cfg() if (backend == "vllm" or embed_backend == "vllm") else None,
        rag=RagConfig(embed_backend=embed_backend),
    )


# --- _parse_text_tool_calls ---

def test_parse_text_tool_calls_python_tags():
    content = '<|python_start|>{"type": "function", "name": "get_status", "parameters": {}}<|python_end|>'
    calls = _parse_text_tool_calls(content)
    assert calls == [{"name": "get_status", "arguments": {}}]


def test_parse_text_tool_calls_with_arguments():
    content = '<|python_start|>{"name": "search", "parameters": {"query": "python"}}<|python_end|>'
    calls = _parse_text_tool_calls(content)
    assert calls == [{"name": "search", "arguments": {"query": "python"}}]


def test_parse_text_tool_calls_no_match():
    assert _parse_text_tool_calls("Hello, how can I help?") is None


def test_parse_text_tool_calls_multiple():
    content = (
        '<|python_start|>{"name": "tool_a", "parameters": {}}<|python_end|>'
        '<|python_start|>{"name": "tool_b", "parameters": {"x": 1}}<|python_end|>'
    )
    calls = _parse_text_tool_calls(content)
    assert len(calls) == 2
    assert calls[0]["name"] == "tool_a"
    assert calls[1] == {"name": "tool_b", "arguments": {"x": 1}}


# --- OllamaBackend ---

@pytest.mark.asyncio
async def test_ollama_backend_chat_plain_text():
    backend = OllamaBackend(_ollama_cfg())
    msg = MagicMock()
    msg.content = "Hello!"
    msg.tool_calls = []
    resp = MagicMock()
    resp.message = msg
    with patch.object(backend._client, "chat", new=AsyncMock(return_value=resp)):
        result = await backend.chat("llama3.2", [{"role": "user", "content": "hi"}], None)
    assert isinstance(result, ChatResponse)
    assert result.content == "Hello!"
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_ollama_backend_chat_structured_tool_calls():
    backend = OllamaBackend(_ollama_cfg())
    tc = MagicMock()
    tc.function.name = "search"
    tc.function.arguments = {"query": "python"}
    tc.model_dump.return_value = {"function": {"name": "search", "arguments": {"query": "python"}}}
    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tc]
    resp = MagicMock()
    resp.message = msg
    with patch.object(backend._client, "chat", new=AsyncMock(return_value=resp)):
        result = await backend.chat("llama3.2", [], [])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "python"}
    assert result.raw_assistant_message["role"] == "assistant"
    assert "tool_calls" in result.raw_assistant_message


@pytest.mark.asyncio
async def test_ollama_backend_chat_text_embedded_fallback():
    backend = OllamaBackend(_ollama_cfg())
    msg = MagicMock()
    msg.content = '<|python_start|>{"name": "get_status", "parameters": {}}<|python_end|>'
    msg.tool_calls = []
    resp = MagicMock()
    resp.message = msg
    with patch.object(backend._client, "chat", new=AsyncMock(return_value=resp)):
        result = await backend.chat("llama3.2", [], None)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_status"
    assert result.content == ""  # stripped


@pytest.mark.asyncio
async def test_ollama_backend_list_models():
    backend = OllamaBackend(_ollama_cfg())
    m1, m2 = MagicMock(), MagicMock()
    m1.model = "llava"
    m2.model = "llama3.2"
    resp = MagicMock()
    resp.models = [m1, m2]
    with patch.object(backend._client, "list", new=AsyncMock(return_value=resp)):
        result = await backend.list_models()
    assert result == ["llama3.2", "llava"]


@pytest.mark.asyncio
async def test_ollama_backend_list_models_error():
    backend = OllamaBackend(_ollama_cfg())
    with patch.object(backend._client, "list", side_effect=Exception("conn refused")):
        result = await backend.list_models()
    assert result == []


@pytest.mark.asyncio
async def test_ollama_backend_embed():
    backend = OllamaBackend(_ollama_cfg())
    resp = MagicMock()
    resp.embeddings = [[0.1, 0.2, 0.3]]
    with patch.object(backend._client, "embed", new=AsyncMock(return_value=resp)):
        result = await backend.embed("nomic-embed-text", "hello")
    assert result == [0.1, 0.2, 0.3]


def test_ollama_format_tool_result():
    backend = OllamaBackend(_ollama_cfg())
    tc = ToolCall(name="search", arguments={})
    msg = backend.format_tool_result(tc, "result text")
    assert msg == {"role": "tool", "content": "result text", "tool_name": "search"}


# --- VLLMBackend ---

@pytest.mark.asyncio
async def test_vllm_backend_chat_plain_text():
    backend = VLLMBackend(_vllm_cfg())
    choice = MagicMock()
    choice.message.content = "Hello from vLLM!"
    choice.message.tool_calls = None
    resp = MagicMock()
    resp.choices = [choice]
    with patch.object(backend._client.chat.completions, "create", new=AsyncMock(return_value=resp)):
        result = await backend.chat("llama3", [{"role": "user", "content": "hi"}], None)
    assert result.content == "Hello from vLLM!"
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_vllm_backend_chat_tool_calls():
    backend = VLLMBackend(_vllm_cfg())
    tc = MagicMock()
    tc.id = "call_abc"
    tc.function.name = "search"
    tc.function.arguments = json.dumps({"query": "python"})
    choice = MagicMock()
    choice.message.content = ""
    choice.message.tool_calls = [tc]
    resp = MagicMock()
    resp.choices = [choice]
    with patch.object(backend._client.chat.completions, "create", new=AsyncMock(return_value=resp)):
        result = await backend.chat("llama3", [], [])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "python"}
    assert result.tool_calls[0].id == "call_abc"


@pytest.mark.asyncio
async def test_vllm_backend_list_models():
    backend = VLLMBackend(_vllm_cfg())
    m1, m2 = MagicMock(), MagicMock()
    m1.id = "llama3"
    m2.id = "mistral"
    resp = MagicMock()
    resp.data = [m2, m1]
    with patch.object(backend._client.models, "list", new=AsyncMock(return_value=resp)):
        result = await backend.list_models()
    assert result == ["llama3", "mistral"]


@pytest.mark.asyncio
async def test_vllm_backend_list_models_error():
    backend = VLLMBackend(_vllm_cfg())
    with patch.object(backend._client.models, "list", side_effect=Exception("conn refused")):
        result = await backend.list_models()
    assert result == []


@pytest.mark.asyncio
async def test_vllm_backend_embed():
    backend = VLLMBackend(_vllm_cfg())
    emb = MagicMock()
    emb.embedding = [0.4, 0.5, 0.6]
    resp = MagicMock()
    resp.data = [emb]
    with patch.object(backend._client.embeddings, "create", new=AsyncMock(return_value=resp)):
        result = await backend.embed("e5-mistral", "hello")
    assert result == [0.4, 0.5, 0.6]


def test_vllm_format_tool_result():
    backend = VLLMBackend(_vllm_cfg())
    tc = ToolCall(name="search", arguments={}, id="call_xyz")
    msg = backend.format_tool_result(tc, "result text")
    assert msg == {"role": "tool", "tool_call_id": "call_xyz", "content": "result text"}


def test_vllm_format_tool_result_no_id():
    backend = VLLMBackend(_vllm_cfg())
    tc = ToolCall(name="search", arguments={}, id=None)
    msg = backend.format_tool_result(tc, "result")
    assert msg["tool_call_id"] == ""


# --- ChatResponse.thinking ---

def test_chat_response_thinking_default_none():
    r = ChatResponse(content="hello")
    assert r.thinking is None


def test_chat_response_thinking_set():
    r = ChatResponse(content="answer", thinking="I reasoned about it")
    assert r.thinking == "I reasoned about it"


def test_ollama_backend_has_chat_stream():
    backend = OllamaBackend(_ollama_cfg())
    assert hasattr(backend, "chat_stream")
    assert callable(backend.chat_stream)


def test_vllm_backend_has_chat_stream():
    backend = VLLMBackend(_vllm_cfg())
    assert hasattr(backend, "chat_stream")
    assert callable(backend.chat_stream)


# --- Factories ---

def test_create_backend_ollama():
    cfg = _make_config(backend="ollama")
    backend = create_backend(cfg)
    assert isinstance(backend, OllamaBackend)


def test_create_backend_vllm():
    cfg = _make_config(backend="vllm", embed_backend="vllm")
    backend = create_backend(cfg)
    assert isinstance(backend, VLLMBackend)


def test_create_embed_backend_ollama():
    cfg = _make_config(embed_backend="ollama")
    backend = create_embed_backend(cfg)
    assert isinstance(backend, OllamaBackend)


def test_create_embed_backend_vllm():
    cfg = _make_config(backend="vllm", embed_backend="vllm")
    backend = create_embed_backend(cfg)
    assert isinstance(backend, VLLMBackend)
