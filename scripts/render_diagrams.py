"""Render clean architecture and data-flow diagrams as PNGs.

Produces vector-quality diagrams (via Chromium at 2x) for the project document,
avoiding any external diagram tooling or logos.

Usage:
    python -m scripts.render_diagrams
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

CSS = """
  * { box-sizing: border-box; }
  body { margin: 0; background: #ffffff; font-family: 'Segoe UI', Arial, sans-serif; }
  .canvas { width: 960px; padding: 32px; background: #ffffff; }
  .layer { border: 1.5px solid #d3d9e3; border-radius: 12px; padding: 16px 18px;
           background: #f8fafc; }
  .layer h4 { margin: 0 0 12px 0; font-size: 13px; letter-spacing: .04em;
              text-transform: uppercase; color: #64748b; font-weight: 700; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; }
  .box { flex: 1; min-width: 150px; background: #ffffff; border: 1.5px solid #cbd5e1;
         border-radius: 9px; padding: 12px 14px; text-align: center; }
  .box .t { font-weight: 700; color: #1e293b; font-size: 15px; }
  .box .s { color: #64748b; font-size: 12.5px; margin-top: 3px; }
  .accent { border-color: #6366f1; background: #eef2ff; }
  .accent .t { color: #4338ca; }
  .green { border-color: #10b981; background: #ecfdf5; }
  .green .t { color: #047857; }
  .amber { border-color: #f59e0b; background: #fffbeb; }
  .amber .t { color: #b45309; }
  .arrow { display: flex; justify-content: center; margin: 6px 0; }
  .arrow svg { display: block; }
  .split { display: flex; gap: 16px; }
  .split > .layer { flex: 1; }
  .side { display: flex; gap: 16px; align-items: stretch; }
  .side > .layer:first-child { flex: 3; }
  .side > .layer:last-child { flex: 1.1; }
  .loop { display: flex; align-items: center; justify-content: center; margin: 10px 0;
          border: 1.5px dashed #c7d2fe; background: #eef2ff; border-radius: 8px;
          padding: 8px 14px; color: #4338ca; font-size: 13px; font-weight: 600; }
"""

ARROW = (
    '<div class="arrow"><svg width="24" height="26">'
    '<line x1="12" y1="0" x2="12" y2="18" stroke="#94a3b8" stroke-width="2"/>'
    '<polygon points="6,16 18,16 12,25" fill="#94a3b8"/></svg></div>'
)


def _page_html(css: str, body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"


ARCHITECTURE = f"""
<div class="canvas" id="cap">
  <div class="side">
    <div class="layer">
      <h4>1. Data Sources</h4>
      <div class="row">
        <div class="box"><div class="t">yfinance</div><div class="s">Fundamentals, prices, ratios, analyst views</div></div>
        <div class="box"><div class="t">SEC EDGAR</div><div class="s">10-Q / 10-K / 8-K filings</div></div>
        <div class="box"><div class="t">Earnings PDFs</div><div class="s">Transcripts and reports</div></div>
      </div>
    </div>
    <div class="layer">
      <h4>Cross-cutting</h4>
      <div class="box accent" style="margin-bottom:10px"><div class="t">core</div><div class="s">Config (pydantic) + logging (loguru)</div></div>
      <div class="box amber"><div class="t">evaluation</div><div class="s">Retrieval metrics + RAGAS</div></div>
    </div>
  </div>

  {ARROW}

  <div class="layer">
    <h4>2. RAG Pipeline</h4>
    <div class="row">
      <div class="box"><div class="t">Ingestion</div><div class="s">pdfplumber</div></div>
      <div class="box"><div class="t">Chunking</div><div class="s">Recursive splitter</div></div>
      <div class="box"><div class="t">Embeddings</div><div class="s">Gemini REST, 3072-d</div></div>
      <div class="box green"><div class="t">ChromaDB</div><div class="s">Persistent vector store</div></div>
    </div>
  </div>

  {ARROW}

  <div class="layer">
    <h4>3. Shared Tool Layer &nbsp;(mcp_server/tools)</h4>
    <div class="row">
      <div class="box"><div class="t">Financials</div><div class="s">get_financials, ratios, prices, analysts</div></div>
      <div class="box"><div class="t">Filings</div><div class="s">fetch_sec_filing</div></div>
      <div class="box green"><div class="t">RAG Search</div><div class="s">search_earnings_documents</div></div>
    </div>
  </div>

  {ARROW}

  <div class="split">
    <div class="layer">
      <h4>4a. MCP Server</h4>
      <div class="box accent"><div class="t">FastMCP (stdio)</div><div class="s">6 tools over the MCP protocol &rarr; Claude and other MCP clients</div></div>
    </div>
    <div class="layer">
      <h4>4b. Reasoning Agent</h4>
      <div class="box accent"><div class="t">LangGraph ReAct</div><div class="s">Groq LLM plans and calls the same tools</div></div>
    </div>
  </div>

  {ARROW}

  <div class="layer">
    <h4>5. Presentation</h4>
    <div class="box"><div class="t">Streamlit UI</div><div class="s">Snapshot &nbsp;|&nbsp; Ask the analyst &nbsp;|&nbsp; Ingest documents</div></div>
  </div>
</div>
"""

AGENT_FLOW = f"""
<div class="canvas" id="cap">
  <div class="layer">
    <h4>Ask the analyst: request flow</h4>
    <div class="box"><div class="t">User question</div><div class="s">"How did NVDA's margins trend and what is its P/E?"</div></div>
    {ARROW}
    <div class="box accent"><div class="t">LangGraph agent (Groq LLM)</div><div class="s">Reason: decide which tool to call next</div></div>
    {ARROW}
    <div class="row">
      <div class="box"><div class="t">Financial tools</div><div class="s">yfinance</div></div>
      <div class="box"><div class="t">Filing tool</div><div class="s">SEC EDGAR</div></div>
      <div class="box green"><div class="t">RAG search</div><div class="s">ChromaDB + Gemini</div></div>
    </div>
    {ARROW}
    <div class="box"><div class="t">Observation</div><div class="s">Tool result returned to the model as context</div></div>
    <div class="loop">
      <svg width="34" height="24" style="vertical-align:middle;margin-right:8px">
        <path d="M22 20 C 4 20, 4 4, 22 4" fill="none" stroke="#6366f1" stroke-width="2" stroke-dasharray="4 3"/>
        <polygon points="18,0 26,3 18,8" fill="#6366f1"/>
      </svg>
      <span>ReAct loop: the model repeats reason and act until the question can be answered</span>
    </div>
    {ARROW}
    <div class="box accent"><div class="t">Final answer</div><div class="s">Quantitative summary with a labelled bottom line</div></div>
  </div>
</div>
"""


def _shoot(page, html_body: str, name: str) -> None:
    page.set_content(_page_html(CSS, html_body), wait_until="load")
    page.locator("#cap").screenshot(path=str(OUT / name))
    print("saved", name)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=2)
        _shoot(page, ARCHITECTURE, "diagram_architecture.png")
        _shoot(page, AGENT_FLOW, "diagram_agent_flow.png")
        browser.close()


if __name__ == "__main__":
    main()
