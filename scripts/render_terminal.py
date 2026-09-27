"""Render captured command output as a clean terminal-style PNG.

Used to produce the verification "output" screenshot for the project document.

Usage:
    python -m scripts.render_terminal
"""

from __future__ import annotations

import html
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# (command, output_lines)
BLOCKS = [
    (
        "python -m scripts.healthcheck",
        [
            "stock-earnings-agent health check",
            "----------------------------------------",
            "  [OK]   configuration: model=qwen/qwen3.8-27b, embedding=gemini-embedding-001",
            "  [OK]   google embeddings: 3072-dimensional vector",
            "  [OK]   vector store: collection holds 1 chunk(s)",
            "  [OK]   groq llm: responded ('ok')",
            "----------------------------------------",
            "All systems operational.",
        ],
    ),
    (
        "python -m pytest",
        ["29 passed, 6 deselected in 25.01s"],
    ),
    (
        "python -m pytest tests/test_vector_store.py -m integration",
        ["6 passed in 14.44s"],
    ),
    (
        "python -m evaluation.ragas_eval",
        ["Retrieval metrics: {'samples': 3, 'hit_rate': 1.0, 'mrr': 1.0}"],
    ),
]


def _render_lines() -> str:
    parts = []
    for command, output in BLOCKS:
        parts.append(f'<div class="cmd"><span class="prompt">$</span> {html.escape(command)}</div>')
        for line in output:
            cls = "out ok" if "[OK]" in line or "operational" in line or "passed" in line else "out"
            parts.append(f'<div class="{cls}">{html.escape(line)}</div>')
        parts.append('<div class="gap"></div>')
    return "\n".join(parts)


HTML = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  body {{ margin: 0; background: #f4f5f7; padding: 28px; }}
  .term {{ width: 900px; border-radius: 10px; overflow: hidden;
           box-shadow: 0 8px 30px rgba(0,0,0,.12); font-family: 'Cascadia Code','Consolas',monospace; }}
  .bar {{ background: #23262e; padding: 12px 16px; }}
  .dot {{ height: 12px; width: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }}
  .r {{ background:#ff5f56 }} .y {{ background:#ffbd2e }} .g {{ background:#27c93f }}
  .body {{ background: #1b1e24; color: #d6dae0; padding: 20px 22px; font-size: 14px; line-height: 1.7; }}
  .cmd {{ color: #e6e9ee; }}
  .prompt {{ color: #56b6c2; font-weight: 700; margin-right: 6px; }}
  .out {{ color: #9aa4b2; white-space: pre; }}
  .out.ok {{ color: #6fcf97; }}
  .gap {{ height: 10px; }}
</style></head><body>
  <div class="term">
    <div class="bar"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span></div>
    <div class="body">{_render_lines()}</div>
  </div>
</body></html>"""


def main() -> None:
    tmp = OUT / "_terminal.html"
    tmp.write_text(HTML, encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=2)
        page.goto(tmp.as_uri(), wait_until="load")
        page.locator(".term").screenshot(path=str(OUT / "06_verification_output.png"))
        browser.close()
    tmp.unlink(missing_ok=True)
    print("saved 06_verification_output.png")


if __name__ == "__main__":
    main()
