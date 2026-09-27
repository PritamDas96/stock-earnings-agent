"""Render a docs markdown file to a professional PDF.

Converts the markdown to styled HTML and prints it to PDF with Chromium, so any
diagrams and screenshots are embedded at full quality.

Usage:
    python -m scripts.build_pdf                      # PROJECT_DOCUMENT.md
    python -m scripts.build_pdf TECHNICAL_GUIDE.md   # a specific file
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parent.parent / "docs"
_NAME = sys.argv[1] if len(sys.argv) > 1 else "PROJECT_DOCUMENT.md"
MD = DOCS / _NAME
HTML = DOCS / f"_{MD.stem.lower()}.html"
PDF = MD.with_suffix(".pdf")

CSS = """
  @page { size: A4; margin: 18mm 16mm; }
  * { box-sizing: border-box; }
  body { font-family: 'Segoe UI', Arial, sans-serif; color: #1f2937;
         font-size: 10.8pt; line-height: 1.55; }
  h1 { font-size: 25pt; color: #111827; margin: 0 0 2pt; }
  h1 + h3 { margin-top: 0; color: #6b7280; font-weight: 600; letter-spacing: .02em; }
  h2 { font-size: 15pt; color: #1e293b; margin: 22pt 0 8pt;
       padding-bottom: 4pt; border-bottom: 2px solid #e5e7eb; }
  h3 { font-size: 12.5pt; color: #334155; margin: 14pt 0 6pt; }
  p { margin: 6pt 0; }
  a { color: #4338ca; text-decoration: none; word-break: break-word; }
  hr { border: none; border-top: 1px solid #e5e7eb; margin: 14pt 0; }
  ul, ol { margin: 6pt 0 6pt 18pt; }
  li { margin: 3pt 0; }
  table { border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 9.8pt;
          page-break-inside: avoid; }
  th, td { border: 1px solid #d7dce3; padding: 6pt 8pt; text-align: left;
           vertical-align: top; }
  th { background: #f1f5f9; color: #334155; font-weight: 700; }
  tr:nth-child(even) td { background: #fafbfc; }
  code { font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 9pt;
         background: #eef1f5; padding: 1px 5px; border-radius: 4px; color: #334155; }
  pre { background: #f6f8fa; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 11pt 13pt; overflow-x: auto; page-break-inside: avoid; margin: 9pt 0; }
  pre code { background: none; padding: 0; font-size: 8.8pt; line-height: 1.5;
             white-space: pre-wrap; word-break: break-word; color: #1f2937; }
  img { max-width: 100%; height: auto; display: block; margin: 10pt auto;
        border: 1px solid #e5e7eb; border-radius: 8px; page-break-inside: avoid; }
  h2 { page-break-after: avoid; }
  h3 { page-break-after: avoid; }
"""


def main() -> None:
    body = markdown.markdown(
        MD.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    HTML.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(HTML.as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(PDF),
            format="A4",
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        )
        browser.close()

    HTML.unlink(missing_ok=True)
    print(f"saved {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
