import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ingest import (
    detect_source_type,
    extract_text_from_pdf,
    fetch_url_text,
    collect_links,
)


def test_detect_source_type_pdf():
    assert detect_source_type("/path/to/file.pdf") == "pdf"


def test_detect_source_type_text():
    assert detect_source_type("/path/to/readme.txt") == "text"
    assert detect_source_type("/path/to/spec.md") == "text"


def test_detect_source_type_url():
    assert detect_source_type("https://example.com/page") == "url"
    assert detect_source_type("http://rfc-editor.org/rfc/rfc9110.txt") == "url"


def test_collect_links_same_domain():
    html = """<html><body>
    <a href="/page1">P1</a>
    <a href="https://example.com/page2">P2</a>
    <a href="https://other.com/page3">P3</a>
    </body></html>"""
    links = collect_links(html, base_url="https://example.com")
    assert "https://example.com/page1" in links
    assert "https://example.com/page2" in links
    assert "https://other.com/page3" not in links


def test_collect_links_deduplication():
    html = """<html><body>
    <a href="/page">A</a>
    <a href="/page">B</a>
    </body></html>"""
    links = collect_links(html, base_url="https://example.com")
    assert links.count("https://example.com/page") == 1


@pytest.mark.asyncio
async def test_fetch_url_text_plain():
    mock_response = MagicMock()
    mock_response.text = "RFC content here"
    mock_response.headers = {"content-type": "text/plain"}
    mock_response.raise_for_status = MagicMock()

    with patch("ingest.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        text = await fetch_url_text("https://example.com/rfc.txt")
    assert text == "RFC content here"


@pytest.mark.asyncio
async def test_fetch_url_text_html_strips_tags():
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Hello world</p></body></html>"
    mock_response.headers = {"content-type": "text/html"}
    mock_response.raise_for_status = MagicMock()

    with patch("ingest.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        text = await fetch_url_text("https://example.com/page.html")
    assert "Hello world" in text
    assert "<p>" not in text


def test_collect_links_excludes_fragments():
    html = """<html><body>
    <a href="#top">Top</a>
    <a href="/page#section">Section</a>
    </body></html>"""
    links = collect_links(html, base_url="https://example.com")
    # Fragment-only link resolves to base URL (no fragment)
    assert "https://example.com" in links or links == ["https://example.com/page"]
    # /page#section should appear as /page (no fragment)
    assert all("#" not in link for link in links)


@pytest.mark.asyncio
async def test_ingest_source_file_not_found():
    from ingest import ingest_source
    mock_manager = MagicMock()
    mock_manager.ingest = AsyncMock(return_value=0)
    count = await ingest_source(mock_manager, "col", "/nonexistent/path.txt")
    assert count == 0
    mock_manager.ingest.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_source_ingest_failure(tmp_path):
    from ingest import ingest_source
    test_file = tmp_path / "test.txt"
    test_file.write_text("some content")
    mock_manager = MagicMock()
    mock_manager.ingest = AsyncMock(side_effect=RuntimeError("embed failed"))
    count = await ingest_source(mock_manager, "col", str(test_file))
    assert count == 0
