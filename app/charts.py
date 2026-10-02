"""Professional financial visualizations (Plotly) for the Streamlit UI.

Each ``*_figure`` builder takes the JSON-serialisable dicts returned by
``mcp_server.tools`` and returns a Plotly ``Figure`` ready for
``st.plotly_chart`` — or ``None`` when the input lacks enough data to draw a
meaningful chart, so callers can simply skip rendering.

Design follows finance-dashboard conventions: a clean white template,
consistent colour coding (green favourable / red unfavourable / blue accent /
grey context), hover interactivity and generous whitespace.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Shared theme --------------------------------------------------------

UP = "#16a34a"       # favourable / gains
DOWN = "#dc2626"     # unfavourable / losses
ACCENT = "#2563eb"   # primary accent (price, revenue)
ACCENT_2 = "#7c3aed"  # secondary accent (net income)
GREY = "#9ca3af"     # context / neutral
INK = "#111827"      # text

_FONT = dict(family="Inter, Segoe UI, system-ui, sans-serif", color=INK, size=13)


def _theme(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    """Apply the shared professional look to a figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INK)),
        template="plotly_white",
        font=_FONT,
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _num(value: Any) -> float | None:
    """Coerce to float, treating NaN / non-numeric as missing."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return None if f != f else f


# --- 1. Price: candlestick + volume --------------------------------------

def price_figure(series: dict[str, Any]) -> go.Figure | None:
    """Candlestick price chart with a volume sub-panel.

    ``series`` is the output of ``tools.get_price_series``.
    """
    if not series or "error" in series or not series.get("close"):
        return None

    dates = series["dates"]
    close = series["close"]
    period = series.get("period", "")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.76, 0.24],
    )
    fig.add_trace(
        go.Candlestick(
            x=dates, open=series["open"], high=series["high"],
            low=series["low"], close=close, name="Price",
            increasing_line_color=UP, decreasing_line_color=DOWN,
            increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Volume bars coloured by up/down day.
    vol_colors = [
        UP if close[i] >= (close[i - 1] if i else close[i]) else DOWN
        for i in range(len(close))
    ]
    fig.add_trace(
        go.Bar(x=dates, y=series.get("volume", []), marker_color=vol_colors,
               name="Volume", opacity=0.5, showlegend=False),
        row=2, col=1,
    )

    gain = close[-1] >= close[0]
    _theme(fig, f"Price &amp; volume · {period}", height=440)
    fig.update_layout(hovermode="x", xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    # Subtle performance tint via a cumulative line is unnecessary; colour cue:
    fig.update_layout(title=dict(
        text=f"Price &amp; volume · {period} "
             f"<span style='color:{UP if gain else DOWN}'>"
             f"({'+' if gain else ''}{round((close[-1]-close[0])/close[0]*100,1)}%)</span>"))
    return fig


# --- 2. Fundamentals trend: revenue & net income + margins ---------------

def fundamentals_trend_figure(hist: dict[str, Any]) -> go.Figure | None:
    """Grouped revenue & net-income bars with margin lines on a second axis.

    ``hist`` is the output of ``tools.get_financials_history``.
    """
    if not hist or "error" in hist or not hist.get("periods"):
        return None
    if not any(v is not None for v in hist.get("revenue", [])):
        return None

    periods = hist["periods"]
    scale = 1e9  # display in $B
    rev_b = [None if v is None else round(v / scale, 2) for v in hist["revenue"]]
    ni_b = [None if v is None else round(v / scale, 2) for v in hist["net_income"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=periods, y=rev_b, name="Revenue",
                         marker_color=ACCENT, opacity=0.85), secondary_y=False)
    fig.add_trace(go.Bar(x=periods, y=ni_b, name="Net income",
                         marker_color=ACCENT_2, opacity=0.85), secondary_y=False)

    for key, label, color in (
        ("gross_margin", "Gross %", UP),
        ("operating_margin", "Operating %", "#f59e0b"),
        ("net_margin", "Net %", DOWN),
    ):
        vals = hist.get(key, [])
        if any(v is not None for v in vals):
            fig.add_trace(
                go.Scatter(x=periods, y=vals, name=label, mode="lines+markers",
                           line=dict(color=color, width=2)),
                secondary_y=True,
            )

    _theme(fig, "Revenue, net income &amp; margins", height=430)
    # Give the title its own line at the very top so the legend sits cleanly
    # below it instead of overlapping (the chart has five legend entries).
    fig.update_layout(
        barmode="group",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=96, b=10),
        title=dict(y=0.97, yref="container", yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left"),
    )
    fig.update_yaxes(title_text="$ Billions", secondary_y=False)
    fig.update_yaxes(title_text="Margin %", secondary_y=True, showgrid=False)
    return fig


# --- 3. Analyst price-target range (bullet) ------------------------------

def analyst_target_figure(
    analyst: dict[str, Any], current_price: float | None
) -> go.Figure | None:
    """Horizontal range of low / mean / high analyst targets vs current price."""
    if not analyst or "error" in analyst:
        return None
    low = _num(analyst.get("target_price_low"))
    mean = _num(analyst.get("target_price_mean"))
    high = _num(analyst.get("target_price_high"))
    price = _num(current_price)
    if low is None or high is None or mean is None:
        return None

    fig = go.Figure()
    # The target range as a thick bar.
    fig.add_trace(go.Scatter(
        x=[low, high], y=[0, 0], mode="lines",
        line=dict(color=GREY, width=14), name="Target range",
        hoverinfo="skip", showlegend=False,
    ))
    # Low / mean / high markers.
    fig.add_trace(go.Scatter(
        x=[low, mean, high], y=[0, 0, 0], mode="markers+text",
        marker=dict(color=[DOWN, ACCENT, UP], size=16, line=dict(color="white", width=2)),
        text=[f"Low<br>${low:,.0f}", f"Mean<br>${mean:,.0f}", f"High<br>${high:,.0f}"],
        textposition="bottom center", showlegend=False,
        hovertemplate="$%{x:,.2f}<extra></extra>",
    ))
    # Current price marker.
    if price is not None:
        fig.add_trace(go.Scatter(
            x=[price], y=[0], mode="markers+text",
            marker=dict(color=INK, size=18, symbol="diamond",
                        line=dict(color="white", width=2)),
            text=[f"Current<br>${price:,.0f}"], textposition="top center",
            showlegend=False, hovertemplate="Current $%{x:,.2f}<extra></extra>",
        ))

    title = "Analyst price targets"
    if price and mean:
        upside = (mean - price) / price * 100
        color = UP if upside >= 0 else DOWN
        title += (f" <span style='color:{color}'>"
                  f"({'+' if upside >= 0 else ''}{upside:.1f}% to mean)</span>")

    _theme(fig, title, height=240)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(visible=False, range=[-1, 1])
    fig.update_xaxes(title_text="Price ($)", showgrid=True)
    return fig


# --- 4. Valuation multiples bar ------------------------------------------

def valuation_figure(
    financials: dict[str, Any], ratios: dict[str, Any]
) -> go.Figure | None:
    """Horizontal bar of the key valuation multiples."""
    spec = [
        ("P/E (TTM)", _num(financials.get("pe_ratio"))),
        ("Forward P/E", _num(financials.get("forward_pe"))),
        ("PEG", _num(ratios.get("peg_ratio"))),
        ("Price/Book", _num(ratios.get("price_to_book"))),
        ("Price/Sales", _num(ratios.get("price_to_sales"))),
        ("EV/EBITDA", _num(ratios.get("ev_to_ebitda"))),
    ]
    spec = [(label, v) for label, v in spec if v is not None and v > 0]
    if not spec:
        return None
    labels = [s[0] for s in spec][::-1]
    values = [s[1] for s in spec][::-1]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=ACCENT,
        text=[f"{v:.1f}×" for v in values], textposition="outside",
        hovertemplate="%{y}: %{x:.2f}×<extra></extra>",
    ))
    _theme(fig, "Valuation multiples", height=320)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="Multiple (×)")
    return fig


# --- 5. Financial-health radar -------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def health_radar_figure(
    financials: dict[str, Any], ratios: dict[str, Any]
) -> go.Figure | None:
    """Normalized 0-100 financial-health profile across five dimensions.

    Scores are heuristic (higher = healthier) so the *shape* communicates the
    company's balance of profitability, returns, growth, liquidity and low
    leverage at a glance.
    """
    pm = _num(financials.get("profit_margins"))
    roe = _num(financials.get("return_on_equity"))
    growth = _num(financials.get("revenue_growth"))
    current = _num(financials.get("current_ratio"))
    de = _num(financials.get("debt_to_equity"))

    axes: list[tuple[str, float | None]] = []
    axes.append(("Profitability", None if pm is None else _clamp(pm * 100 * 2.5)))
    axes.append(("Returns (ROE)", None if roe is None else _clamp(roe * 100 * 2.5)))
    axes.append(("Growth", None if growth is None else _clamp((growth * 100 + 20) / 60 * 100)))
    axes.append(("Liquidity", None if current is None else _clamp(current / 3 * 100)))
    if de is not None:
        de_ratio = de / 100 if de > 5 else de  # yfinance often reports as a percent
        axes.append(("Low leverage", _clamp(100 - de_ratio * 40)))
    else:
        axes.append(("Low leverage", None))

    present = [(label, score) for label, score in axes if score is not None]
    if len(present) < 3:
        return None

    labels = [p[0] for p in present]
    scores = [round(p[1], 1) for p in present]
    # Close the polygon.
    labels_closed = labels + [labels[0]]
    scores_closed = scores + [scores[0]]

    fig = go.Figure(go.Scatterpolar(
        r=scores_closed, theta=labels_closed, fill="toself",
        line=dict(color=ACCENT, width=2), fillcolor="rgba(37,99,235,0.25)",
        hovertemplate="%{theta}: %{r:.0f}/100<extra></extra>",
    ))
    _theme(fig, "Financial-health profile (0–100)", height=420)
    # Generous margins so the angular labels (e.g. "Low leverage" at the bottom)
    # are never clipped by the figure edge.
    fig.update_layout(
        margin=dict(l=90, r=90, t=64, b=76),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    )
    return fig


# --- 6. KPI gauges -------------------------------------------------------

def kpi_gauges_figure(
    financials: dict[str, Any], ratios: dict[str, Any]
) -> go.Figure | None:
    """A row of speedometer gauges for headline KPIs."""
    pe = _num(financials.get("pe_ratio"))
    margin = _num(financials.get("profit_margins"))
    roe = _num(financials.get("return_on_equity"))

    gauges = []
    if pe is not None and pe > 0:
        gauges.append(dict(
            title="P/E (TTM)", value=round(pe, 1), axis=[0, 60],
            steps=[(0, 15, "#dcfce7"), (15, 30, "#fef9c3"), (30, 60, "#fee2e2")],
            bar=ACCENT,
        ))
    if margin is not None:
        gauges.append(dict(
            title="Profit margin %", value=round(margin * 100, 1), axis=[0, 40],
            steps=[(0, 10, "#fee2e2"), (10, 20, "#fef9c3"), (20, 40, "#dcfce7")],
            bar=UP,
        ))
    if roe is not None:
        gauges.append(dict(
            title="Return on equity %", value=round(roe * 100, 1), axis=[0, 50],
            steps=[(0, 10, "#fee2e2"), (10, 20, "#fef9c3"), (20, 50, "#dcfce7")],
            bar=ACCENT_2,
        ))
    if not gauges:
        return None

    # Auto-expand the axis for outliers (e.g. ROE ~150%) so the bar never
    # saturates past the end of the scale; stretch the top band to match.
    for g in gauges:
        lo, hi = g["axis"]
        if g["value"] > hi:
            hi = round(g["value"] * 1.15)
            g["axis"][1] = hi
            first, _, color = g["steps"][-1]
            g["steps"][-1] = (first, hi, color)

    fig = make_subplots(
        rows=1, cols=len(gauges),
        specs=[[{"type": "indicator"} for _ in gauges]],
    )
    for i, g in enumerate(gauges, start=1):
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=g["value"],
            title=dict(text=g["title"], font=dict(size=13)),
            gauge=dict(
                axis=dict(range=g["axis"]),
                bar=dict(color=g["bar"]),
                steps=[dict(range=[lo, hi], color=c) for lo, hi, c in g["steps"]],
            ),
        ), row=1, col=i)

    fig.update_layout(
        template="plotly_white", font=_FONT, height=240,
        margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="white",
    )
    return fig
