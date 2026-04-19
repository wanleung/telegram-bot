from utils import md_to_telegram_html


def test_bold():
    assert "<b>hello</b>" in md_to_telegram_html("**hello**")


def test_italic():
    assert "<i>hello</i>" in md_to_telegram_html("*hello*")


def test_heading_becomes_bold():
    result = md_to_telegram_html("### Status")
    assert "<b>Status</b>" in result


def test_inline_code():
    result = md_to_telegram_html("`code`")
    assert "<code>code</code>" in result


def test_fenced_code_block():
    result = md_to_telegram_html("```\nprint('hi')\n```")
    assert "<pre>" in result


def test_table_renders_as_pre():
    md = "| Line | Status |\n|------|--------|\n| Central | Good |"
    result = md_to_telegram_html(md)
    assert "<pre>" in result
    assert "Central" in result
    assert "Good" in result
    assert "<table>" not in result


def test_table_header_separator():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    result = md_to_telegram_html(md)
    # Separator row should be present
    assert "---" in result or "-+-" in result


def test_plain_text_unchanged():
    result = md_to_telegram_html("Hello world")
    assert "Hello world" in result


def test_no_excessive_newlines():
    md = "Para one\n\nPara two\n\n\n\nPara three"
    result = md_to_telegram_html(md)
    assert "\n\n\n" not in result


def test_html_entities_escaped():
    result = md_to_telegram_html("x < y & z > 0")
    assert "<" not in result.replace("<b>", "").replace("<i>", "").replace("<pre>", "").replace("<code>", "")
    assert "&lt;" in result or "x" in result
