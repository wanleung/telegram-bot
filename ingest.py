"""Ingestion CLI and helper functions for the RAG knowledge base."""

import argparse
import asyncio
import logging
import os
import sys
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config import load_config
from rag import RagManager

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def detect_source_type(source: str) -> str:
    """Return 'pdf', 'text', or 'url' based on the source string."""
    if source.startswith("http://") or source.startswith("https://"):
        return "url"
    if source.lower().endswith(".pdf"):
        return "pdf"
    return "text"


def extract_text_from_pdf(path: str) -> str:
    """Extract plain text from a PDF file using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


async def fetch_url_text(url: str) -> str:
    """Fetch a URL and return plain text (strips HTML if necessary)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        return response.text


def collect_links(html: str, base_url: str) -> list[str]:
    """Extract same-domain absolute links from an HTML string."""
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(base_url).netloc
    seen: set[str] = set()
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.netloc == base_domain and absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


async def ingest_source(
    manager: RagManager, collection: str, source: str
) -> int:
    """Ingest a single source (path or URL) into *collection*. Returns chunk count."""
    source_type = detect_source_type(source)
    try:
        if source_type == "pdf":
            text = extract_text_from_pdf(source)
        elif source_type == "url":
            text = await fetch_url_text(source)
        else:
            with open(source, encoding="utf-8") as f:
                text = f.read()
    except Exception as exc:
        logger.error("Failed to read %s: %s", source, exc)
        return 0

    return await manager.ingest(collection, source, text)


async def crawl(
    manager: RagManager,
    collection: str,
    start_url: str,
    max_depth: int,
) -> None:
    """Recursively crawl *start_url* up to *max_depth* and ingest all pages."""
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(start_url, 0)]
    base_domain = urlparse(start_url).netloc

    while queue:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"[{len(visited)}/~] {url}")
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                response = await client.get(url)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                html = response.text
        except Exception as exc:
            logger.warning("Skipping %s: %s", url, exc)
            continue

        if "html" in content_type:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

            if depth < max_depth:
                for link in collect_links(html, url):
                    if link not in visited and urlparse(link).netloc == base_domain:
                        queue.append((link, depth + 1))
        else:
            text = html

        await manager.ingest(collection, url, text)


async def main_async(args: argparse.Namespace) -> None:
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    cfg = load_config(config_path)
    manager = RagManager(cfg.rag)

    if not cfg.rag.enabled:
        print("RAG is disabled in config. Set rag.enabled: true to use ingest.")
        sys.exit(1)

    collection = args.collection

    if args.crawl:
        depth = args.depth if hasattr(args, "depth") and args.depth else 2
        await crawl(manager, collection, args.crawl, max_depth=depth)

    elif args.url_file:
        with open(args.url_file) as f:
            urls = [line.strip() for line in f if line.strip()]
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {url}")
            count = await ingest_source(manager, collection, url)
            print(f"  → {count} chunks ingested")

    elif args.source:
        count = await ingest_source(manager, collection, args.source)
        print(f"Ingested {count} chunks from {args.source} into '{collection}'")

    else:
        print("Specify a source, --url-file, or --crawl. See --help.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base")
    parser.add_argument("--collection", required=True, help="Collection name")
    parser.add_argument("source", nargs="?", help="URL or file path to ingest")
    parser.add_argument("--url-file", metavar="FILE", help="File with one URL per line")
    parser.add_argument("--crawl", metavar="URL", help="Start URL for recursive crawl")
    parser.add_argument("--depth", type=int, default=2, help="Max crawl depth (default: 2)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
