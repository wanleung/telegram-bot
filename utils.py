"""Utilities for formatting LLM output for Telegram."""

import html as html_lib
import re
from html.parser import HTMLParser

import markdown as md_lib


class _TelegramHTMLConverter(HTMLParser):
    """
    Converts standard HTML to Telegram's limited HTML subset.

    Supported Telegram tags: <b>, <i>, <u>, <s>, <code>, <pre>, <a>.
    Tables are rendered as monospace <pre> blocks.
    Headings are rendered as bold text.
    """

    def __init__(self) -> None:
        super().__init__()
        self._out: list[str] = []
        self._tag_stack: list[str] = []
        # Table state
        self._in_table = False
        self._table_rows: list[tuple[list[str], bool]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._cell_is_header = False
        # Code state (data inside code/pre must not be re-escaped)
        self._in_pre = False

    # ------------------------------------------------------------------
    # HTMLParser callbacks
    # ------------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self._tag_stack.append(tag)
        attrs_dict = dict(attrs)

        if tag in ("b", "strong"):
            self._emit("<b>")
        elif tag in ("i", "em"):
            self._emit("<i>")
        elif tag in ("u", "ins"):
            self._emit("<u>")
        elif tag in ("s", "del", "strike"):
            self._emit("<s>")
        elif tag == "code":
            self._emit("<code>")
        elif tag == "pre":
            self._in_pre = True
            self._emit("<pre>")
        elif tag == "a":
            href = html_lib.escape(attrs_dict.get("href", ""), quote=True)
            self._emit(f'<a href="{href}">')
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._emit("\n<b>")
        elif tag == "blockquote":
            self._emit("<blockquote>")
        elif tag == "br":
            self._emit("\n")
        elif tag == "li":
            self._emit("• ")
        elif tag == "table":
            self._in_table = True
            self._table_rows = []
        elif tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._current_cell = []
            self._cell_is_header = tag == "th"

    def handle_endtag(self, tag: str) -> None:
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()

        if tag in ("b", "strong"):
            self._emit("</b>")
        elif tag in ("i", "em"):
            self._emit("</i>")
        elif tag in ("u", "ins"):
            self._emit("</u>")
        elif tag in ("s", "del", "strike"):
            self._emit("</s>")
        elif tag == "code":
            self._emit("</code>")
        elif tag == "pre":
            self._in_pre = False
            self._emit("</pre>")
        elif tag == "a":
            self._emit("</a>")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._emit("</b>\n")
        elif tag == "blockquote":
            self._emit("</blockquote>")
        elif tag in ("p", "div"):
            self._emit("\n\n")
        elif tag in ("ul", "ol"):
            self._emit("\n")
        elif tag == "li":
            self._emit("\n")
        elif tag in ("td", "th"):
            self._current_row.append("".join(self._current_cell))
            self._current_cell = []
        elif tag == "tr":
            self._table_rows.append((self._current_row[:], self._cell_is_header))
            self._current_row = []
        elif tag == "table":
            self._in_table = False
            self._emit(self._render_table())
            self._table_rows = []

    def handle_data(self, data: str) -> None:
        escaped = html_lib.escape(data) if not self._in_pre else html_lib.escape(data)
        if self._in_table:
            if self._current_cell is not None:
                self._current_cell.append(escaped)
        else:
            self._emit(escaped)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit(self, text: str) -> None:
        self._out.append(text)

    def _render_table(self) -> str:
        """Render an HTML table as a monospace <pre> block."""
        if not self._table_rows:
            return ""
        all_rows = [row for row, _ in self._table_rows]
        cols = max((len(r) for r in all_rows), default=0)
        if cols == 0:
            return ""
        widths = [0] * cols
        for row, _ in self._table_rows:
            for i, cell in enumerate(row):
                if i < cols:
                    # Strip HTML tags from cell text for width calculation
                    plain = re.sub(r"<[^>]+>", "", cell)
                    widths[i] = max(widths[i], len(plain))

        lines: list[str] = []
        for i, (row, is_header) in enumerate(self._table_rows):
            padded = []
            for j in range(cols):
                cell = row[j] if j < len(row) else ""
                plain = re.sub(r"<[^>]+>", "", cell)
                padded.append(plain.ljust(widths[j]))
            lines.append(" | ".join(padded))
            if is_header:
                lines.append("-+-".join("-" * w for w in widths))

        return "<pre>" + html_lib.escape("\n".join(lines)) + "</pre>\n"

    def result(self) -> str:
        text = "".join(self._out)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def md_to_telegram_html(text: str) -> str:
    """
    Convert a markdown string to Telegram-compatible HTML.

    Renders tables as monospace <pre> blocks (Telegram doesn't support HTML
    tables). Headings become bold. All other standard markdown formatting is
    preserved using Telegram's supported tag set.
    """
    raw_html = md_lib.markdown(text, extensions=["tables", "fenced_code"])
    converter = _TelegramHTMLConverter()
    converter.feed(raw_html)
    return converter.result()
