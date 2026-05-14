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
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        result = await backend.chat("llama3.2", [{"role": "user", "content": "hi"}], None)
    assert isinstance(result, ChatResponse)
    assert result.content == "Hello!"
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_ollama_backend_chat_structured_tool_calls():
    backend = OllamaBackend(_ollama_cfg())
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "search"
    tc.function.arguments = json.dumps({"query": "python"})
    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tc]
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        result = await backend.chat("llama3.2", [], [])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "python"}
    assert result.tool_calls[0].id == "call_1"
    assert result.raw_assistant_message["role"] == "assistant"
    assert "tool_calls" in result.raw_assistant_message


@pytest.mark.asyncio
async def test_ollama_backend_chat_text_embedded_fallback():
    backend = OllamaBackend(_ollama_cfg())
    msg = MagicMock()
    msg.content = '<|python_start|>{"name": "get_status", "parameters": {}}<|python_end|>'
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        result = await backend.chat("llama3.2", [], None)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_status"
    assert result.content == ""  # stripped


@pytest.mark.asyncio
async def test_ollama_backend_list_models():
    backend = OllamaBackend(_ollama_cfg())
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "models": [{"model": "llava:latest"}, {"model": "llama3.2:latest"}]
    }
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await backend.list_models()
    assert result == ["llama3.2:latest", "llava:latest"]


@pytest.mark.asyncio
async def test_ollama_backend_list_models_error():
    backend = OllamaBackend(_ollama_cfg())
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(side_effect=Exception("conn refused"))
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await backend.list_models()
    assert result == []


@pytest.mark.asyncio
async def test_ollama_backend_embed():
    backend = OllamaBackend(_ollama_cfg())
    emb = MagicMock()
    emb.embedding = [0.1, 0.2, 0.3]
    resp = MagicMock()
    resp.data = [emb]
    with patch("litellm.aembedding", new=AsyncMock(return_value=resp)):
        result = await backend.embed("nomic-embed-text", "hello")
    assert result == [0.1, 0.2, 0.3]


def test_ollama_format_tool_result():
    backend = OllamaBackend(_ollama_cfg())
    tc = ToolCall(name="search", arguments={}, id="call_abc")
    msg = backend.format_tool_result(tc, "result text")
    assert msg == {"role": "tool", "tool_call_id": "call_abc", "content": "result text"}


def test_ollama_format_tool_result_no_id():
    backend = OllamaBackend(_ollama_cfg())
    tc = ToolCall(name="search", arguments={}, id=None)
    msg = backend.format_tool_result(tc, "result text")
    assert msg == {"role": "tool", "tool_call_id": "", "content": "result text"}


# --- VLLMBackend ---

@pytest.mark.asyncio
async def test_vllm_backend_chat_plain_text():
    backend = VLLMBackend(_vllm_cfg())
    msg = MagicMock()
    msg.content = "Hello from vLLM!"
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
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
    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tc]
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        result = await backend.chat("llama3", [], [])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "python"}
    assert result.tool_calls[0].id == "call_abc"


@pytest.mark.asyncio
async def test_vllm_backend_list_models():
    backend = VLLMBackend(_vllm_cfg())
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"id": "mistral"}, {"id": "llama3"}]
    }
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_resp)
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await backend.list_models()
    assert result == ["llama3", "mistral"]


@pytest.mark.asyncio
async def test_vllm_backend_list_models_error():
    backend = VLLMBackend(_vllm_cfg())
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(side_effect=Exception("conn refused"))
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await backend.list_models()
    assert result == []


@pytest.mark.asyncio
async def test_vllm_backend_embed():
    backend = VLLMBackend(_vllm_cfg())
    emb = MagicMock()
    emb.embedding = [0.4, 0.5, 0.6]
    resp = MagicMock()
    resp.data = [emb]
    with patch("litellm.aembedding", new=AsyncMock(return_value=resp)):
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


@pytest.mark.asyncio
async def test_vllm_backend_chat_stream_content_chunks():
    """chat_stream yields ChatResponse chunks with content from delta."""
    backend = VLLMBackend(_vllm_cfg())

    def _make_chunk(text):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text
        chunk.choices[0].delta.reasoning_content = None
        return chunk

    async def _fake_stream(*args, **kwargs):
        for text in ["Hello", " world"]:
            yield _make_chunk(text)

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        chunks = []
        async for cr in backend.chat_stream("llama3", [{"role": "user", "content": "hi"}], None):
            chunks.append(cr)

    assert len(chunks) == 2
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"


@pytest.mark.asyncio
async def test_vllm_backend_chat_stream_thinking_chunks():
    """chat_stream yields thinking from reasoning_content when think=True."""
    backend = VLLMBackend(_vllm_cfg())

    def _make_chunk(text=None, reasoning=None):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text or ""
        chunk.choices[0].delta.reasoning_content = reasoning
        return chunk

    async def _fake_stream(*args, **kwargs):
        yield _make_chunk(reasoning="thinking...")
        yield _make_chunk(text="Answer")

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        chunks = []
        async for cr in backend.chat_stream("llama3", [], None, think=True):
            chunks.append(cr)

    assert any(c.thinking == "thinking..." for c in chunks)
    assert any(c.content == "Answer" for c in chunks)


@pytest.mark.asyncio
async def test_vllm_backend_chat_stream_passes_enable_thinking_in_extra_body():
    """When think=True, extra_body with enable_thinking is passed to acompletion."""
    backend = VLLMBackend(_vllm_cfg())
    captured_kwargs = {}

    async def _fake_stream(*args, **kwargs):
        captured_kwargs.update(kwargs)
        chunk = MagicMock()
        chunk.choices[0].delta.content = "ok"
        chunk.choices[0].delta.reasoning_content = None
        yield chunk

    async def mock_acompletion(*args, **kwargs):
        return _fake_stream(*args, **kwargs)

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        async for _ in backend.chat_stream("llama3", [], None, think=True):
            pass

    assert captured_kwargs.get("extra_body") == {"enable_thinking": True}


# --- OllamaBackend.chat_stream ---

@pytest.mark.asyncio
async def test_ollama_backend_chat_stream_content_chunks():
    """chat_stream yields ChatResponse chunks with content."""
    backend = OllamaBackend(_ollama_cfg())

    def _make_chunk(text):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text
        chunk.choices[0].delta.thinking = None
        return chunk

    async def _fake_stream(*args, **kwargs):
        for text in ["Hello", " world"]:
            yield _make_chunk(text)

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        chunks = []
        async for cr in backend.chat_stream("llama3.2", [{"role": "user", "content": "hi"}], None):
            chunks.append(cr)

    assert len(chunks) == 2
    assert chunks[0].content == "Hello"
    assert chunks[0].thinking is None
    assert chunks[1].content == " world"


@pytest.mark.asyncio
async def test_ollama_backend_chat_stream_thinking_chunks():
    """chat_stream yields thinking fragments when think=True."""
    backend = OllamaBackend(_ollama_cfg())

    def _make_chunk(text=None, thinking=None):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text or ""
        chunk.choices[0].delta.thinking = thinking
        return chunk

    async def _fake_stream(*args, **kwargs):
        yield _make_chunk(thinking="I should reason...")
        yield _make_chunk(text="Final answer")

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        chunks = []
        async for cr in backend.chat_stream("llama3.2", [], None, think=True):
            chunks.append(cr)

    assert any(c.thinking == "I should reason..." for c in chunks)
    assert any(c.content == "Final answer" for c in chunks)


@pytest.mark.asyncio
async def test_ollama_backend_chat_stream_passes_think_flag():
    """chat_stream passes think=True kwarg to acompletion when requested."""
    backend = OllamaBackend(_ollama_cfg())
    captured_kwargs = {}

    async def _fake_stream(*args, **kwargs):
        captured_kwargs.update(kwargs)
        chunk = MagicMock()
        chunk.choices[0].delta.content = "ok"
        chunk.choices[0].delta.thinking = None
        yield chunk

    async def _fake_acompletion(*args, **kwargs):
        return _fake_stream(*args, **kwargs)

    with patch("litellm.acompletion", side_effect=_fake_acompletion):
        async for _ in backend.chat_stream("llama3.2", [], None, think=True):
            pass

    assert captured_kwargs.get("think") is True


@pytest.mark.asyncio
async def test_vllm_backend_chat_stream_skips_empty_choices_chunk():
    """Chunks with choices=[] must not raise IndexError."""
    backend = VLLMBackend(_vllm_cfg())

    def _make_chunk(text=None, empty=False):
        chunk = MagicMock()
        chunk.choices = [] if empty else [MagicMock()]
        if not empty:
            chunk.choices[0].delta.content = text or ""
            chunk.choices[0].delta.reasoning_content = None
        return chunk

    async def _fake_stream(*args, **kwargs):
        yield _make_chunk(text="Hello")
        yield _make_chunk(empty=True)

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        chunks = [cr async for cr in backend.chat_stream("llama3", [], None)]

    assert len(chunks) == 1
    assert chunks[0].content == "Hello"


@pytest.mark.asyncio
async def test_vllm_backend_chat_stream_no_extra_body_when_think_false():
    """When think=False, extra_body must not be included."""
    backend = VLLMBackend(_vllm_cfg())
    captured_kwargs = {}

    async def _fake_stream(*args, **kwargs):
        captured_kwargs.update(kwargs)
        chunk = MagicMock()
        chunk.choices[0].delta.content = "ok"
        chunk.choices[0].delta.reasoning_content = None
        yield chunk

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        async for _ in backend.chat_stream("llama3", [], None, think=False):
            pass

    assert "extra_body" not in captured_kwargs


@pytest.mark.asyncio
async def test_ollama_backend_chat_stream_no_think_kwarg_when_false():
    """When think=False, think kwarg must not be sent."""
    backend = OllamaBackend(_ollama_cfg())
    captured_kwargs = {}

    async def _fake_stream(*args, **kwargs):
        captured_kwargs.update(kwargs)
        chunk = MagicMock()
        chunk.choices[0].delta.content = "ok"
        chunk.choices[0].delta.thinking = None
        yield chunk

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream())):
        async for _ in backend.chat_stream("llama3.2", [], None, think=False):
            pass

    assert "think" not in captured_kwargs


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
