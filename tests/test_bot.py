"""Tests for bot.py helper functions."""
import html as html_lib

import pytest

from bot import _build_reply


def test_build_reply_final_plain_content():
    result = _build_reply("Hello world", "", final=True)
    assert "Hello world" in result


def test_build_reply_final_no_content():
    result = _build_reply("", "", final=True)
    assert result == "<i>(no response)</i>"


def test_build_reply_final_with_thinking():
    result = _build_reply("Answer", "I think...", final=True)
    assert "<tg-spoiler>" in result
    assert "I think..." in result
    assert "Answer" in result


def test_build_reply_final_thinking_html_escaped():
    result = _build_reply("ok", "<script>alert(1)</script>", final=True)
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_build_reply_streaming_placeholder():
    result = _build_reply("", "", final=False)
    assert result == "⏳"


def test_build_reply_streaming_with_content():
    result = _build_reply("Hello", "", final=False)
    assert "Hello" in result


def test_build_reply_streaming_with_thinking():
    result = _build_reply("Hello", "reasoning...", final=False)
    assert "<tg-spoiler>" in result
    assert "Hello" in result


def test_build_reply_truncates_thinking_to_fit_limit():
    long_thinking = "x" * 5000
    content = "Short answer"
    result = _build_reply(content, long_thinking, final=True)
    assert len(result) <= 4096
    assert "Short answer" in result


def test_build_reply_truncates_content_if_needed():
    long_content = "y" * 5000
    result = _build_reply(long_content, "", final=True)
    assert len(result) <= 4096
