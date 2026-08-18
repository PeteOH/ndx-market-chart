# ruff: noqa: E501
from __future__ import annotations

import html
import json
from collections import OrderedDict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ndx_chart.dataset import ATR_PERIODS, CHART_SLOPE_WINDOWS, CHART_SMA_WINDOWS
from ndx_chart.moving_averages import moving_average_column

_SURFACE = "#f7f8f4"
_PANEL = "#ffffff"
_INK = "#17201b"
_MUTED = "#68716b"
_GRID = "#e4e8e3"
_AXIS = "#cbd2cc"
_GREEN = "#087f5b"
_RED = "#c92a2a"
_BLUE = "#1c7ed6"
_GOLD = "#d97706"
_VIOLET = "#7048e8"
_TEAL = "#0b7285"
_PLOT_ID = "market-chart"

_SMA_COLORS = {
    3: "#f08c00",
    5: "#e8590c",
    7: "#d6336c",
    10: "#12b886",
    20: "#7048e8",
    50: "#1c7ed6",
    200: "#087f5b",
    250: "#495057",
}

_ATR_COLORS = {
    5: "#c92a2a",
    7: "#e8590c",
    10: "#1c7ed6",
    14: "#7048e8",
}

_TRACE_DESCRIPTIONS = {
    "^NDX price": (
        "Nasdaq-100 index closing level. It is the unleveraged market baseline "
        "for judging the direction of large non-financial Nasdaq companies."
    ),
    "TQQQ price": (
        "Closing price of the 3x daily leveraged Nasdaq-100 bull ETF. It magnifies "
        "daily gains and losses, while compounding makes long-term returns path-dependent."
    ),
    "SQQQ price": (
        "Closing price of the -3x daily inverse Nasdaq-100 ETF. It generally rises "
        "on Nasdaq down days, but daily resets can cause decay when held over time."
    ),
    "SMA(3)": (
        "Three-session simple average of NDX. It follows price very closely, making "
        "short-term direction changes visible but also producing more noise."
    ),
    "SMA(5)": (
        "Five-session simple average of NDX, roughly one trading week. It smooths "
        "daily noise while remaining sensitive to short-term trend changes."
    ),
    "SMA(7)": (
        "Seven-session simple average of NDX. It highlights the short-term trend and "
        "can help confirm whether a brief move is persisting."
    ),
    "SMA(10)": (
        "Ten-session simple average of NDX, roughly two trading weeks. It balances "
        "short-term responsiveness with more noise reduction than the faster averages."
    ),
    "SMA(20)": (
        "Twenty-session simple average of NDX, roughly one trading month. Price above "
        "or below it is commonly used to judge the near-term trend."
    ),
    "SMA(50)": (
        "Fifty-session simple average of NDX. It is a widely watched intermediate-trend "
        "reference and can act as a dynamic support or resistance area."
    ),
    "SMA(200)": (
        "Two-hundred-session simple average of NDX. It is a common long-term trend and "
        "market-regime reference; crossings can be important but lag price."
    ),
    "SMA(250)": (
        "Two-hundred-fifty-session simple average of NDX, about one trading year. It "
        "shows the broad annual trend while filtering most short-term movement."
    ),
    "EMA(5)": (
        "Five-session exponential average of NDX that weights recent prices more heavily. "
        "It reacts faster than SMA(5), which helps expose very short-term momentum."
    ),
    "Chandelier long exit": (
        "Trailing long-position stop: the 22-session high minus 3x ATR(22). A close below "
        "it can flag that an uptrend has weakened enough to consider exiting."
    ),
    "Chandelier short exit": (
        "Trailing short-position stop: the 22-session low plus 3x ATR(22). A close above "
        "it can flag that a downtrend has weakened enough to consider exiting."
    ),
    "RSI(2)": (
        "Very fast 0-100 momentum oscillator using two sessions. Extreme readings can "
        "highlight short-term overextension, but they do not guarantee a reversal."
    ),
    "RSI(3)": (
        "Fast 0-100 momentum oscillator using three sessions. It detects short-term "
        "overbought or oversold conditions with slightly less noise than RSI(2)."
    ),
    "RSI(14)": (
        "Standard 14-session 0-100 momentum oscillator. Readings above 70 or below 30 "
        "often mark strong or stretched momentum, not automatic sell or buy signals."
    ),
    "SMA slope(5)": (
        "One-session percentage change of SMA(5). Its sign shows short-term trend "
        "direction and its size shows how quickly that trend is changing."
    ),
    "SMA slope(7)": (
        "One-session percentage change of SMA(7). It measures the direction and pace "
        "of the approximately one-week NDX trend."
    ),
    "SMA slope(10)": (
        "One-session percentage change of SMA(10). It measures the direction and pace "
        "of the approximately two-week NDX trend."
    ),
    "SMA slope(20)": (
        "One-session percentage change of SMA(20). Positive values indicate a rising "
        "monthly trend; negative values indicate a falling one."
    ),
    "SMA slope(50)": (
        "One-session percentage change of SMA(50). It measures the direction and speed "
        "of the intermediate trend with less noise than shorter slopes."
    ),
    "SMA slope(200)": (
        "One-session percentage change of SMA(200). It shows whether the long-term trend "
        "is strengthening or weakening, but reacts slowly to new conditions."
    ),
    "SMA slope(250)": (
        "One-session percentage change of SMA(250). It measures the direction and pace "
        "of the broad annual trend and is the slowest slope shown."
    ),
    "ATR(5)": (
        "Five-session Average True Range of NDX. It reacts fastest to volatility "
        "changes, making short bursts visible but also producing the most noise."
    ),
    "ATR(7)": (
        "Seven-session Average True Range of NDX, roughly one trading week. It "
        "shows recent volatility with slightly more smoothing than ATR(5)."
    ),
    "ATR(10)": (
        "Ten-session Average True Range of NDX, roughly two trading weeks. It "
        "balances responsiveness with more stability than the shorter readings."
    ),
    "ATR(14)": (
        "Fourteen-session Average True Range of NDX. It measures typical price movement "
        "using the standard lookback; ATR measures volatility, not direction."
    ),
    "MACD histogram": (
        "Difference between the MACD line and its signal line. Movement away from zero "
        "suggests momentum is accelerating; movement toward zero suggests it is fading."
    ),
    "MACD line": (
        "NDX EMA(12) minus EMA(26). Positive values favor faster upside momentum and "
        "negative values favor downside momentum; turns can precede trend changes."
    ),
    "MACD signal": (
        "Nine-session EMA of the MACD line. Crossovers with the MACD line are commonly "
        "used as momentum-change signals, though they can whipsaw in sideways markets."
    ),
}


def build_chart_html(
    datasets: dict[str, pd.DataFrame],
    *,
    title: str = "Nasdaq-100 market & indicator explorer",
) -> str:
    """Return a self-contained interactive page with offline Plotly assets.

    In addition to Plotly's pan, box-zoom, wheel-zoom, and range slider, the
    page adds explicit start/end inputs, quick ranges, and grouped trace
    checkboxes. No CDN or data request is needed after the file is created.
    """
    fig = build_chart_figure(datasets, title=title)
    dates = pd.concat(
        [pd.to_datetime(frame["session_date"]) for frame in datasets.values()],
        ignore_index=True,
    )
    first_date = dates.min().date().isoformat()
    last_date = dates.max().date().isoformat()

    controls: OrderedDict[str, list[tuple[str, list[int], bool]]] = OrderedDict()
    for index, trace in enumerate(fig.data):
        metadata = trace.meta or {}
        category = metadata.get("control_category")
        label = metadata.get("control_label")
        if not category or not label:
            continue
        controls.setdefault(category, [])
        matching = next((item for item in controls[category] if item[0] == label), None)
        visible = trace.visible is True or trace.visible is None
        if matching:
            matching[1].append(index)
        else:
            controls[category].append((label, [index], visible))

    fieldsets = []
    for category, items in controls.items():
        checkboxes = []
        for label, indices, visible in items:
            checked = " checked" if visible else ""
            index_value = ",".join(str(value) for value in indices)
            description = html.escape(_TRACE_DESCRIPTIONS[label], quote=True)
            checkboxes.append(
                f'<label class="trace-option"><input type="checkbox" '
                f'data-traces="{index_value}"{checked}>'
                f'<span class="trace-label">{html.escape(label)}</span>'
                f'<span class="trace-help" tabindex="0" aria-label="About {html.escape(label, quote=True)}: {description}" '
                f'data-description="{description}">i</span></label>'
            )
        fieldsets.append(
            '<fieldset class="trace-group">'
            f'<legend>{html.escape(category)}</legend>'
            '<div class="group-actions"><button type="button" data-group-action="all">All</button>'
            '<button type="button" data-group-action="none">None</button></div>'
            f'<div class="trace-options">{"".join(checkboxes)}</div></fieldset>'
        )

    rule_operands = [label for items in controls.values() for label, _, _ in items]
    initial_rulesets = {
        "version": 2,
        "rulesets": [
            {
                "name": "Golden cross",
                "type": "simple",
                "enabled": True,
                "color": "#51cf66",
                "expression": {
                    "join": "AND",
                    "items": [
                        {
                            "left": "SMA(50)",
                            "operator": ">=",
                            "right": {"type": "indicator", "value": "SMA(200)"},
                        }
                    ],
                },
            }
        ],
    }
    rule_operands_json = json.dumps(rule_operands).replace("<", "\\u003c")
    initial_rulesets_json = json.dumps(initial_rulesets).replace("<", "\\u003c")

    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs="inline",
        div_id=_PLOT_ID,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: {_SURFACE}; color: {_INK}; }}
    header {{ padding: 28px 32px 18px; border-bottom: 1px solid {_GRID}; background: {_PANEL}; }}
    h1 {{ margin: 0; font-size: clamp(1.45rem, 2.5vw, 2.2rem); letter-spacing: -0.035em; }}
    header p {{ margin: 8px 0 0; color: {_MUTED}; max-width: 78ch; line-height: 1.5; }}
    .workspace {{ display: grid; grid-template-columns: minmax(340px, 410px) minmax(0, 1fr); gap: 16px; padding: 16px; }}
    aside {{ align-self: start; position: sticky; top: 16px; max-height: calc(100vh - 32px); overflow: auto; background: {_PANEL}; border: 1px solid {_GRID}; border-radius: 14px; padding: 16px; box-shadow: 0 8px 28px rgba(23,32,27,.06); }}
    .control-title {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; color: {_MUTED}; font-weight: 750; margin: 0 0 10px; }}
    .date-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .date-grid label {{ color: {_MUTED}; font-size: .76rem; font-weight: 650; }}
    input[type="date"] {{ width: 100%; margin-top: 5px; padding: 8px; border: 1px solid {_AXIS}; border-radius: 8px; color: {_INK}; background: white; }}
    button {{ border: 1px solid {_AXIS}; background: white; color: {_INK}; border-radius: 8px; padding: 6px 9px; cursor: pointer; font: inherit; font-size: .78rem; font-weight: 650; }}
    button:hover {{ border-color: {_GREEN}; color: {_GREEN}; }}
    button:disabled {{ cursor: not-allowed; opacity: .45; }}
    .apply {{ width: 100%; margin-top: 9px; background: {_INK}; border-color: {_INK}; color: white; padding: 9px; }}
    .apply:hover {{ background: {_GREEN}; border-color: {_GREEN}; color: white; }}
    .quick-ranges {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 16px; }}
    .trace-group {{ position: relative; margin: 0 0 12px; padding: 11px 10px 9px; border: 1px solid {_GRID}; border-radius: 10px; }}
    .trace-group legend {{ padding: 0 5px; font-size: .83rem; font-weight: 750; }}
    .group-actions {{ position: absolute; top: -14px; right: 6px; display: flex; gap: 3px; background: {_PANEL}; padding: 2px; }}
    .group-actions button {{ border: 0; padding: 3px 5px; color: {_MUTED}; font-size: .68rem; }}
    .trace-options {{ display: grid; gap: 4px; }}
    .trace-option {{ display: flex; align-items: center; gap: 8px; min-height: 27px; font-size: .81rem; cursor: pointer; }}
    .trace-option input {{ accent-color: {_GREEN}; }}
    .trace-label {{ flex: 1; min-width: 0; }}
    .trace-help {{ position: relative; display: grid; flex: none; place-items: center; width: 17px; height: 17px; border: 1px solid {_AXIS}; border-radius: 50%; color: {_MUTED}; font-size: .68rem; font-style: normal; font-weight: 800; line-height: 1; cursor: help; }}
    .trace-help::after {{ content: attr(data-description); display: none; position: absolute; z-index: 30; right: -2px; bottom: calc(100% + 7px); width: 240px; padding: 9px 10px; border-radius: 8px; background: {_INK}; color: white; box-shadow: 0 8px 24px rgba(23,32,27,.24); font-size: .73rem; font-weight: 500; line-height: 1.42; pointer-events: none; text-align: left; }}
    .trace-option:hover .trace-help::after, .trace-help:focus::after {{ display: block; }}
    .trace-help:focus {{ outline: 2px solid {_GREEN}; outline-offset: 2px; }}
    .rules-panel {{ margin: 18px 0 2px; padding-top: 15px; border-top: 1px solid {_GRID}; }}
    .rules-heading {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 9px; }}
    .rules-heading .control-title {{ margin: 0; }}
    .rules-actions {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }}
    .rules-actions button {{ flex: 1 1 auto; }}
    .rules-json {{ width: 100%; min-height: 82px; resize: vertical; padding: 8px; border: 1px solid {_AXIS}; border-radius: 8px; background: white; color: {_INK}; font: 500 .72rem/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; }}
    .rules-status {{ min-height: 1.2em; margin: 5px 1px 9px; color: {_MUTED}; font-size: .72rem; }}
    .rules-status.error {{ color: {_RED}; }}
    .ruleset-card {{ margin: 9px 0; padding: 10px; border: 1px solid {_GRID}; border-radius: 10px; background: {_SURFACE}; }}
    .ruleset-header {{ display: grid; grid-template-columns: auto minmax(0, 1fr) 34px auto; align-items: center; gap: 7px; }}
    .ruleset-header input[type="text"] {{ min-width: 0; width: 100%; font-weight: 750; }}
    .ruleset-enabled {{ display: grid; place-items: center; }}
    .ruleset-enabled input {{ accent-color: {_GREEN}; }}
    .ruleset-color {{ width: 34px; height: 31px; padding: 2px; border: 1px solid {_AXIS}; border-radius: 7px; background: white; cursor: pointer; }}
    .ruleset-type {{ display: flex; align-items: center; gap: 7px; margin: 9px 0 7px; color: {_MUTED}; font-size: .72rem; }}
    .ruleset-type select {{ min-height: 31px; padding: 5px 6px; border: 1px solid {_AXIS}; border-radius: 7px; background: white; color: {_INK}; font: inherit; font-size: .72rem; }}
    .remove-button {{ padding: 6px 8px; color: {_RED}; }}
    .ruleset-join {{ display: flex; align-items: center; gap: 7px; margin: 9px 0 6px; color: {_MUTED}; font-size: .72rem; }}
    .ruleset-join select {{ width: auto; }}
    .expression-group {{ margin: 7px 0; padding: 8px; border: 1px solid {_GRID}; border-left: 3px solid {_AXIS}; border-radius: 8px; background: {_PANEL}; }}
    .expression-group.nested {{ margin-left: 10px; background: {_SURFACE}; }}
    .expression-header {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 6px; margin-bottom: 6px; }}
    .expression-header .ruleset-join {{ margin: 0; }}
    .expression-actions {{ display: flex; flex-wrap: wrap; gap: 4px; }}
    .expression-actions button {{ padding: 4px 6px; font-size: .68rem; }}
    .expression-items {{ min-width: 0; }}
    .condition-row {{ display: grid; grid-template-columns: minmax(0, 1fr) 62px 31px; gap: 5px; align-items: center; margin: 7px 0; }}
    .condition-row select, .condition-row input, .ruleset-header input, .ruleset-join select {{ min-height: 31px; padding: 5px 6px; border: 1px solid {_AXIS}; border-radius: 7px; background: white; color: {_INK}; font: inherit; font-size: .72rem; }}
    .condition-row input[type="number"] {{ width: 100%; min-width: 0; }}
    .condition-row [data-field="left"] {{ grid-column: 1; grid-row: 1; }}
    .condition-row [data-field="operator"] {{ grid-column: 2; grid-row: 1; }}
    .condition-row [data-field="rightType"] {{ grid-column: 1; grid-row: 2; }}
    .condition-row [data-field="right"] {{ grid-column: 2 / 4; grid-row: 2; min-width: 0; width: 100%; }}
    .condition-row .remove-button {{ grid-column: 3; grid-row: 1; min-width: 31px; }}
    .ruleset-footer {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 8px; }}
    .signal-group {{ margin-top: 9px; padding: 8px; border: 1px solid {_GRID}; border-radius: 8px; background: {_PANEL}; }}
    .signal-group-title {{ display: flex; align-items: center; gap: 6px; font-size: .76rem; font-weight: 800; }}
    .signal-arrow {{ font-size: 1rem; line-height: 1; }}
    .signal-arrow.buy {{ color: {_GREEN}; }}
    .signal-arrow.sell {{ color: {_RED}; }}
    .rule-match-status {{ color: {_MUTED}; font-size: .69rem; }}
    .hint {{ color: {_MUTED}; font-size: .74rem; line-height: 1.45; margin: 13px 2px 0; }}
    .chart-card {{ position: relative; min-width: 0; overflow: hidden; background: {_PANEL}; border: 1px solid {_GRID}; border-radius: 14px; box-shadow: 0 8px 28px rgba(23,32,27,.06); }}
    .global-crosshair {{ display: none; position: absolute; z-index: 20; width: 1px; background: {_MUTED}; opacity: .72; pointer-events: none; }}
    .global-tooltip {{ display: none; position: absolute; z-index: 21; width: min(430px, calc(100% - 24px)); max-height: min(680px, calc(100vh - 48px)); overflow: auto; padding: 11px 12px; border: 1px solid {_AXIS}; border-radius: 10px; background: rgba(255,255,255,.96); color: {_INK}; box-shadow: 0 8px 24px rgba(23,32,27,.16); pointer-events: none; font-size: .75rem; }}
    .hover-date {{ margin-bottom: 7px; font-weight: 800; font-size: .83rem; }}
    .hover-groups {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px 14px; }}
    .hover-group-title {{ margin-bottom: 3px; color: {_MUTED}; font-size: .64rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }}
    .hover-value {{ display: flex; justify-content: space-between; gap: 8px; padding: 1px 0; }}
    .hover-value-name {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .hover-value-number {{ flex: none; font-variant-numeric: tabular-nums; font-weight: 700; }}
    .hover-rules {{ display: grid; gap: 7px; margin-top: 10px; padding-top: 9px; border-top: 1px solid {_GRID}; }}
    .hover-rule {{ padding: 7px 8px; border: 1px solid {_GRID}; border-left: 4px solid var(--rule-color); border-radius: 7px; background: {_SURFACE}; }}
    .hover-rule-name {{ font-weight: 800; }}
    .hover-rule-period {{ margin: 2px 0 6px; color: {_MUTED}; font-size: .69rem; }}
    .hover-rule-performance {{ display: grid; grid-template-columns: minmax(48px, 1fr) repeat(3, auto); gap: 2px 10px; align-items: baseline; font-variant-numeric: tabular-nums; }}
    .hover-rule-performance .heading {{ color: {_MUTED}; font-size: .63rem; font-weight: 750; text-transform: uppercase; }}
    .hover-rule-performance .symbol {{ font-weight: 750; }}
    .hover-rule-performance .gain {{ font-weight: 800; }}
    .hover-rule-performance .positive {{ color: {_GREEN}; }}
    .hover-rule-performance .negative {{ color: {_RED}; }}
    .hover-signals {{ display: grid; gap: 4px; margin-top: 9px; padding-top: 8px; border-top: 1px solid {_GRID}; }}
    .hover-signal {{ display: flex; align-items: center; gap: 6px; font-weight: 800; }}
    .hover-signal.buy {{ color: {_GREEN}; }}
    .hover-signal.sell {{ color: {_RED}; }}
    #{_PLOT_ID} {{ width: 100%; min-height: 1400px; }}
    @media (max-width: 820px) {{
      header {{ padding: 22px 18px 14px; }}
      .workspace {{ grid-template-columns: 1fr; padding: 10px; }}
      aside {{ position: static; max-height: none; }}
      .trace-options {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Daily ^NDX, TQQQ and SQQQ history from 2010. Choose a date range or drag the range slider; use the checkboxes to isolate price, trend, momentum, volatility and MACD data.</p>
    <p>For personal research and education only. Data may be delayed or revised; this chart is not investment advice.</p>
  </header>
  <main class="workspace">
    <aside aria-label="Chart controls">
      <p class="control-title">Date range</p>
      <div class="date-grid">
        <label>From<input id="date-from" type="date" min="{first_date}" max="{last_date}" value="{first_date}"></label>
        <label>To<input id="date-to" type="date" min="{first_date}" max="{last_date}" value="{last_date}"></label>
      </div>
      <button class="apply" id="apply-range" type="button">Apply date range</button>
      <div class="quick-ranges" aria-label="Quick date ranges">
        <button type="button" data-years="1">1Y</button><button type="button" data-years="3">3Y</button>
        <button type="button" data-years="5">5Y</button><button type="button" data-years="10">10Y</button>
        <button type="button" data-years="all">All</button>
      </div>
      <p class="control-title">Data shown</p>
      {''.join(fieldsets)}
      <section class="rules-panel" aria-labelledby="rules-title">
        <div class="rules-heading"><p class="control-title" id="rules-title">Ruleset highlights</p><button id="add-ruleset" type="button">+ Ruleset</button></div>
        <p class="hint">Simple rulesets highlight matching periods. Buy / Sell rulesets use separate expressions and plot green ▲ buy and red ▼ sell signals. Use + Group to create nested AND/OR parentheses.</p>
        <div id="rulesets"></div>
        <div class="rules-actions"><button id="copy-rules" type="button">Copy JSON</button><button id="import-rules" type="button">Import pasted JSON</button></div>
        <textarea class="rules-json" id="rules-json" aria-label="Ruleset JSON" placeholder="Copy rules here, or paste ruleset JSON to import"></textarea>
        <p class="rules-status" id="rules-status" role="status" aria-live="polite"></p>
      </section>
      <p class="hint">Hover a selector row or focus its i badge for an indicator explanation. Mouse wheel: zoom · drag: zoom box · double-click: reset. The crosshair, date, and values for every visible trace follow your pointer through every panel, and each vertical scale fits the visible date range. Slope values are the one-session percentage change of each NDX SMA. Chandelier Exit uses the standard 22-session, 3×ATR setting. Every ruleset group evaluates its own AND/OR items before its parent group, so nested groups behave like parentheses.</p>
    </aside>
    <section class="chart-card" aria-label="Interactive market chart"><div id="global-crosshair" class="global-crosshair" aria-hidden="true"></div><div id="global-tooltip" class="global-tooltip" role="status" aria-live="polite"></div>{plot_html}</section>
  </main>
  <script>
  (() => {{
    const plot = document.getElementById('{_PLOT_ID}');
    const from = document.getElementById('date-from');
    const to = document.getElementById('date-to');
    const crosshair = document.getElementById('global-crosshair');
    const tooltip = document.getElementById('global-tooltip');
    const rulesetsElement = document.getElementById('rulesets');
    const rulesJson = document.getElementById('rules-json');
    const rulesStatus = document.getElementById('rules-status');
    const first = '{first_date}';
    const last = '{last_date}';
    const xAxes = {list(_xaxis_layout_keys(len(datasets) + 4))!r};
    const ruleOperands = {rule_operands_json};
    let rulesets = {initial_rulesets_json}.rulesets;
    let nextRulesetId = 1;
    let ruleSections = [];
    let ruleSignals = [];
    const traceDateIndexes = new WeakMap();
    const escapeHtml = value => String(value)
      .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;').replaceAll("'", '&#039;');
    const dateKey = value => {{
      if (value instanceof Date) return value.toISOString().slice(0, 10);
      const text = String(value);
      if (/^\\d{{4}}-\\d{{2}}-\\d{{2}}/.test(text)) return text.slice(0, 10);
      const parsed = new Date(value);
      return Number.isFinite(parsed.getTime()) ? parsed.toISOString().slice(0, 10) : null;
    }};
    const traceValueAt = (trace, targetDate) => {{
      if (!trace.x || !trace.y) return null;
      let indexByDate = traceDateIndexes.get(trace);
      if (!indexByDate) {{
        indexByDate = new Map();
        trace.x.forEach((date, index) => indexByDate.set(dateKey(date), index));
        traceDateIndexes.set(trace, indexByDate);
      }}
      const index = indexByDate.get(targetDate);
      if (index === undefined) return null;
      const value = Number(trace.y[index]);
      return Number.isFinite(value) ? value : null;
    }};
    const formatHoverValue = (value, category) => {{
      if (category === 'NDX SMA slopes') return (value * 100).toFixed(3) + '%';
      if (category === 'NDX momentum') return value.toFixed(2);
      return value.toLocaleString(undefined, {{maximumFractionDigits: 4}});
    }};
    const formatRuleDate = value => new Date(value + 'T00:00:00Z').toLocaleDateString(undefined, {{
      year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC'
    }});
    const formatRulePrice = value => value === null
      ? 'N/A' : value.toLocaleString(undefined, {{style: 'currency', currency: 'USD', maximumFractionDigits: 2}});
    const formatRuleGain = value => value === null
      ? 'N/A' : (value >= 0 ? '+' : '') + value.toFixed(2) + '%';
    const baseShapes = JSON.parse(JSON.stringify(plot.layout.shapes || []));
    const ruleOperators = ['<', '=', '>', '>=', '<='];
    const operatorTests = {{
      '<': (left, right) => left < right,
      '=': (left, right) => left === right,
      '>': (left, right) => left > right,
      '>=': (left, right) => left >= right,
      '<=': (left, right) => left <= right
    }};
    const newRulesetId = () => 'ruleset-' + nextRulesetId++;
    const defaultOperand = () => ruleOperands.includes('SMA(50)') ? 'SMA(50)' : ruleOperands[0];
    const defaultRightOperand = () => ruleOperands.includes('SMA(200)') ? 'SMA(200)' : ruleOperands[0];
    const blankCondition = (operator = '>=') => ({{
      kind: 'condition', left: defaultOperand(), operator,
      right: {{type: 'indicator', value: defaultRightOperand()}}
    }});
    const blankGroup = (join = 'AND', operator = '>=') => ({{
      kind: 'group', join, items: [blankCondition(operator)]
    }});
    const cloneExpression = expression => JSON.parse(JSON.stringify(expression));
    const blankRuleset = () => ({{
      id: newRulesetId(), name: 'New ruleset', type: 'simple', enabled: true, color: '#74c0fc',
      expression: blankGroup()
    }});
    const MAX_RULE_DEPTH = 8;
    const MAX_RULE_NODES = 100;
    const normaliseCondition = (condition, context, state) => {{
      state.nodes += 1;
      if (state.nodes > MAX_RULE_NODES) {{
        throw new Error(context + ' exceeds the maximum of ' + MAX_RULE_NODES + ' groups and conditions.');
      }}
      if (!condition || !ruleOperands.includes(condition.left)) {{
        throw new Error(context + ' has an unknown left indicator.');
      }}
      if (!ruleOperators.includes(condition.operator)) {{
        throw new Error(context + ' has an unsupported operator.');
      }}
      const right = condition.right || {{}};
      const rightType = right.type === 'number' ? 'number' : 'indicator';
      if (rightType === 'indicator' && !ruleOperands.includes(right.value)) {{
        throw new Error(context + ' has an unknown right indicator.');
      }}
      const numericValue = Number(right.value);
      if (rightType === 'number' && !Number.isFinite(numericValue)) {{
        throw new Error(context + ' needs a numeric value.');
      }}
      return {{
        kind: 'condition',
        left: condition.left,
        operator: condition.operator,
        right: {{type: rightType, value: rightType === 'number' ? numericValue : right.value}}
      }};
    }};
    const normaliseGroup = (group, context, depth = 0, state = {{nodes: 0}}) => {{
      if (!group || typeof group !== 'object') throw new Error(context + ' is invalid.');
      if (depth >= MAX_RULE_DEPTH) {{
        throw new Error(context + ' exceeds the maximum nesting depth of ' + MAX_RULE_DEPTH + '.');
      }}
      state.nodes += 1;
      if (state.nodes > MAX_RULE_NODES) {{
        throw new Error(context + ' exceeds the maximum of ' + MAX_RULE_NODES + ' groups and conditions.');
      }}
      const items = Array.isArray(group.items) ? group.items : group.conditions;
      if (!Array.isArray(items) || !items.length) throw new Error(context + ' needs at least one item.');
      if (items.length > 20) throw new Error(context + ' can contain at most 20 direct items.');
      return {{
        kind: 'group',
        join: group.join === 'OR' ? 'OR' : 'AND',
        items: items.map((item, itemIndex) => {{
          const itemContext = 'Item ' + (itemIndex + 1) + ' in ' + context;
          const isGroup = item && typeof item === 'object' && (
            item.kind === 'group' || Array.isArray(item.items) || Array.isArray(item.conditions)
          );
          return isGroup
            ? normaliseGroup(item, itemContext, depth + 1, state)
            : normaliseCondition(item, itemContext, state);
        }})
      }};
    }};
    const normaliseRulesets = payload => {{
      const source = Array.isArray(payload) ? payload : payload && payload.rulesets;
      if (!Array.isArray(source)) throw new Error('Expected a rulesets array.');
      const version = payload && !Array.isArray(payload) && payload.version !== undefined
        ? Number(payload.version) : 1;
      if (![1, 2].includes(version)) {{
        throw new Error('Only ruleset JSON versions 1 and 2 are supported.');
      }}
      if (source.length > 50) throw new Error('A maximum of 50 rulesets can be imported.');
      return source.map((item, rulesetIndex) => {{
        if (!item || typeof item !== 'object') throw new Error('Ruleset ' + (rulesetIndex + 1) + ' is invalid.');
        const type = item.type === 'buy_sell' ? 'buy_sell' : 'simple';
        const ruleset = {{
          id: newRulesetId(),
          name: String(item.name || 'Ruleset ' + (rulesetIndex + 1)).slice(0, 80),
          type,
          enabled: item.enabled !== false,
          color: /^#[0-9a-f]{{6}}$/i.test(item.color || '') ? item.color : '#74c0fc'
        }};
        if (type === 'buy_sell') {{
          ruleset.buy = normaliseGroup(item.buy, 'buy rule in ruleset ' + (rulesetIndex + 1));
          ruleset.sell = normaliseGroup(item.sell, 'sell rule in ruleset ' + (rulesetIndex + 1));
        }} else {{
          ruleset.expression = normaliseGroup(
            item.expression || item, 'ruleset ' + (rulesetIndex + 1)
          );
        }}
        return ruleset;
      }});
    }};
    rulesets = normaliseRulesets({{version: 2, rulesets}});
    const optionMarkup = (values, selected) => values.map(value =>
      '<option value="' + escapeHtml(value) + '"' + (value === selected ? ' selected' : '') + '>' +
      escapeHtml(value) + '</option>'
    ).join('');
    const exportRulesets = () => ({{
      version: 2,
      rulesets: rulesets.map(ruleset => {{
        const exported = {{
          name: ruleset.name, type: ruleset.type, enabled: ruleset.enabled, color: ruleset.color
        }};
        const exportExpression = expression => expression.kind === 'group'
          ? {{
              join: expression.join,
              items: expression.items.map(exportExpression)
            }}
          : {{
              left: expression.left,
              operator: expression.operator,
              right: {{type: expression.right.type, value: expression.right.value}}
            }};
        if (ruleset.type === 'buy_sell') {{
          exported.buy = exportExpression(ruleset.buy);
          exported.sell = exportExpression(ruleset.sell);
        }} else {{
          exported.expression = exportExpression(ruleset.expression);
        }}
        return exported;
      }})
    }});
    const setRulesStatus = (message, isError = false) => {{
      rulesStatus.textContent = message;
      rulesStatus.classList.toggle('error', isError);
    }};
    const traceForRuleOperand = label => (plot._fullData || plot.data).find(trace =>
      (trace.meta && trace.meta.control_label) === label
    );
    const conditionMatches = (condition, targetDate) => {{
      const leftTrace = traceForRuleOperand(condition.left);
      const left = leftTrace ? traceValueAt(leftTrace, targetDate) : null;
      let right = Number(condition.right.value);
      if (condition.right.type === 'indicator') {{
        const rightTrace = traceForRuleOperand(condition.right.value);
        right = rightTrace ? traceValueAt(rightTrace, targetDate) : null;
      }}
      if (left === null || right === null || !Number.isFinite(right)) return false;
      return operatorTests[condition.operator](left, right);
    }};
    const matchingSpans = flags => {{
      const spans = [];
      let start = null;
      flags.forEach((matches, index) => {{
        if (matches && start === null) start = index;
        if (start !== null && (!matches || index === flags.length - 1)) {{
          spans.push([start, matches && index === flags.length - 1 ? index : index - 1]);
          start = null;
        }}
      }});
      return spans;
    }};
    const expressionMatches = (expression, targetDate) => {{
      if (expression.kind === 'condition') return conditionMatches(expression, targetDate);
      const results = expression.items.map(item => expressionMatches(item, targetDate));
      return expression.join === 'OR' ? results.some(Boolean) : results.every(Boolean);
    }};
    const applyRuleHighlights = () => {{
      const reference = traceForRuleOperand('^NDX price') || (plot._fullData || plot.data)[0];
      const timeline = reference && reference.x ? reference.x.map(dateKey) : [];
      const highlightShapes = [];
      ruleSections = [];
      ruleSignals = [];
      rulesets.forEach(ruleset => {{
        let matchCount = 0;
        let spanCount = 0;
        let buyCount = 0;
        let sellCount = 0;
        if (ruleset.enabled && ruleset.type === 'simple' && timeline.length) {{
          const flags = timeline.map(targetDate => expressionMatches(ruleset.expression, targetDate));
          matchCount = flags.filter(Boolean).length;
          const spans = matchingSpans(flags);
          spanCount = spans.length;
          spans.forEach(([startIndex, endIndex]) => {{
            const startDate = timeline[startIndex];
            const endDate = timeline[endIndex];
            const shapeEnd = timeline[endIndex + 1] || new Date(
              new Date(endDate + 'T00:00:00Z').getTime() + 86400000
            ).toISOString().slice(0, 10);
            const performance = ['TQQQ', 'SQQQ'].map(symbol => {{
              const trace = traceForRuleOperand(symbol + ' price');
              const entry = trace ? traceValueAt(trace, startDate) : null;
              const exit = trace ? traceValueAt(trace, endDate) : null;
              const gain = entry !== null && exit !== null && entry !== 0
                ? ((exit / entry) - 1) * 100 : null;
              return {{symbol, entry, exit, gain}};
            }});
            ruleSections.push({{
              rulesetId: ruleset.id, name: ruleset.name, color: ruleset.color,
              startDate, endDate, ongoing: endIndex === timeline.length - 1, performance
            }});
            highlightShapes.push({{
              type: 'rect', xref: 'x', yref: 'paper', x0: startDate, x1: shapeEnd,
              y0: 0, y1: 1, fillcolor: ruleset.color, opacity: 0.18,
              line: {{width: 0}}, layer: 'below'
            }});
          }});
        }} else if (ruleset.enabled && ruleset.type === 'buy_sell' && timeline.length) {{
          for (const signal of ['buy', 'sell']) {{
            const flags = timeline.map(targetDate => expressionMatches(ruleset[signal], targetDate));
            flags.forEach((matches, index) => {{
              if (!matches || (index > 0 && flags[index - 1])) return;
              const targetDate = timeline[index];
              const value = traceValueAt(reference, targetDate);
              if (value === null) return;
              ruleSignals.push({{
                rulesetId: ruleset.id, name: ruleset.name, signal, targetDate, value
              }});
              if (signal === 'buy') buyCount += 1; else sellCount += 1;
            }});
          }}
        }}
        const status = document.getElementById('rule-match-' + ruleset.id);
        if (status) {{
          if (!ruleset.enabled) status.textContent = 'Disabled';
          else if (ruleset.type === 'buy_sell') status.textContent =
            buyCount.toLocaleString() + ' buy · ' + sellCount.toLocaleString() + ' sell signals';
          else status.textContent =
            matchCount.toLocaleString() + ' sessions · ' + spanCount.toLocaleString() + ' sections';
        }}
      }});
      const markerUpdates = ['buy', 'sell'].map(signal => {{
        const traceIndex = plot.data.findIndex(trace => trace.meta && trace.meta.rule_signal === signal);
        if (traceIndex < 0) return Promise.resolve();
        const signals = ruleSignals.filter(item => item.signal === signal);
        return Plotly.restyle(plot, {{
          x: [signals.map(item => item.targetDate)],
          y: [signals.map(item => item.value)],
          text: [signals.map(item => (signal === 'buy' ? '▲ Buy · ' : '▼ Sell · ') + item.name)]
        }}, [traceIndex]);
      }});
      markerUpdates.push(Plotly.relayout(plot, {{shapes: baseShapes.concat(highlightShapes)}}));
      return Promise.all(markerUpdates).then(() => ({{
        sessions: timeline.length,
        highlightShapes: highlightShapes.length,
        buySignals: ruleSignals.filter(item => item.signal === 'buy').length,
        sellSignals: ruleSignals.filter(item => item.signal === 'sell').length
      }}));
    }};
    const expressionRoot = (ruleset, signal = '') => signal ? ruleset[signal] : ruleset.expression;
    const expressionAtPath = (root, path) => {{
      if (!path) return root;
      return path.split('.').reduce((node, index) => node.items[Number(index)], root);
    }};
    const renderRulesets = () => {{
      rulesetsElement.innerHTML = rulesets.map(ruleset => {{
        const conditionMarkup = (condition, path, signal, siblingCount) => {{
          const rightControl = condition.right.type === 'number'
            ? '<input type="number" step="any" data-field="right" value="' + escapeHtml(condition.right.value) + '" aria-label="Comparison number">'
            : '<select data-field="right" aria-label="Right indicator">' + optionMarkup(ruleOperands, condition.right.value) + '</select>';
          return '<div class="condition-row" data-path="' + path.join('.') + '" data-signal="' + signal + '">' +
            '<select data-field="left" aria-label="Left indicator">' + optionMarkup(ruleOperands, condition.left) + '</select>' +
            '<select data-field="operator" aria-label="Operator">' + optionMarkup(ruleOperators, condition.operator) + '</select>' +
            '<select data-field="rightType" aria-label="Comparison type">' + optionMarkup(['indicator', 'number'], condition.right.type) + '</select>' +
            rightControl +
            '<button class="remove-button" type="button" data-action="remove-condition" aria-label="Remove condition"' +
              (siblingCount === 1 ? ' disabled' : '') + '>×</button></div>';
        }};
        const expressionMarkup = (group, path = [], signal = '', isRoot = true, canRemove = false) => {{
          const items = group.items.map((item, itemIndex) => {{
            const itemPath = path.concat(itemIndex);
            return item.kind === 'group'
              ? expressionMarkup(item, itemPath, signal, false, group.items.length > 1)
              : conditionMarkup(item, itemPath, signal, group.items.length);
          }}).join('');
          const removeGroup = isRoot ? '' :
            '<button class="remove-button" type="button" data-action="remove-group" aria-label="Remove nested group"' +
              (canRemove ? '' : ' disabled') + '>× Group</button>';
          return '<section class="expression-group ' + (isRoot ? 'root' : 'nested') + '" data-path="' + path.join('.') + '" data-signal="' + signal + '">' +
            '<div class="expression-header"><label class="ruleset-join">Combine items with <select data-field="groupJoin" aria-label="Group join">' + optionMarkup(['AND', 'OR'], group.join) + '</select></label>' +
            '<div class="expression-actions"><button type="button" data-action="add-condition">+ Condition</button>' +
            '<button type="button" data-action="add-group">+ Group</button>' + removeGroup + '</div></div>' +
            '<div class="expression-items">' + items + '</div></section>';
        }};
        const simpleEditor = ruleset.type === 'simple'
          ? expressionMarkup(ruleset.expression) +
            '<div class="ruleset-footer"><span></span><span class="rule-match-status" id="rule-match-' + ruleset.id + '"></span></div>'
          : '';
        const signalEditor = signal => {{
          const label = signal === 'buy' ? 'Buy expression' : 'Sell expression';
          const arrow = signal === 'buy' ? '▲' : '▼';
          return '<section class="signal-group" data-signal="' + signal + '"><div class="signal-group-title"><span class="signal-arrow ' + signal + '">' + arrow + '</span>' + label + '</div>' +
            expressionMarkup(ruleset[signal], [], signal) + '</section>';
        }};
        const editor = ruleset.type === 'buy_sell'
          ? signalEditor('buy') + signalEditor('sell') + '<div class="ruleset-footer"><span></span><span class="rule-match-status" id="rule-match-' + ruleset.id + '"></span></div>'
          : simpleEditor;
        return '<article class="ruleset-card" data-ruleset="' + ruleset.id + '">' +
          '<div class="ruleset-header"><label class="ruleset-enabled" title="Enable ruleset"><input type="checkbox" data-field="enabled" aria-label="Enable ' + escapeHtml(ruleset.name) + '"' + (ruleset.enabled ? ' checked' : '') + '></label>' +
          '<input type="text" data-field="name" aria-label="Ruleset name" value="' + escapeHtml(ruleset.name) + '">' +
          '<input class="ruleset-color" type="color" data-field="color" aria-label="Highlight colour" title="Simple highlight colour; Buy / Sell markers use green and red" value="' + escapeHtml(ruleset.color) + '"' + (ruleset.type === 'buy_sell' ? ' disabled' : '') + '>' +
          '<button class="remove-button" type="button" data-action="remove-ruleset" aria-label="Remove ruleset">×</button></div>' +
          '<label class="ruleset-type">Type <select data-field="type" aria-label="Ruleset type"><option value="simple"' + (ruleset.type === 'simple' ? ' selected' : '') + '>Simple highlight</option><option value="buy_sell"' + (ruleset.type === 'buy_sell' ? ' selected' : '') + '>Buy / Sell</option></select></label>' +
          editor + '</article>';
      }}).join('');
      applyRuleHighlights();
    }};
    const updateGlobalTooltip = (hover, card, size) => {{
      const point = hover.points && hover.points[0];
      if (!point) return;
      const targetDate = dateKey(point.x);
      if (!targetDate) return;
      const grouped = new Map();
      (plot._fullData || plot.data).forEach(trace => {{
        if (trace.visible === false || trace.visible === 'legendonly') return;
        const metadata = trace.meta || {{}};
        if (metadata.rule_signal) return;
        const value = traceValueAt(trace, targetDate);
        if (value === null) return;
        const category = metadata.control_category || 'Other';
        const label = metadata.control_label || trace.name || 'Value';
        if (!grouped.has(category)) grouped.set(category, []);
        grouped.get(category).push({{label, value}});
      }});
      const groups = [...grouped.entries()].map(([category, values]) =>
        '<section><div class="hover-group-title">' + escapeHtml(category) + '</div>' +
        values.map(item => '<div class="hover-value"><span class="hover-value-name">' +
          escapeHtml(item.label) + '</span><span class="hover-value-number">' +
          escapeHtml(formatHoverValue(item.value, category)) + '</span></div>').join('') +
        '</section>'
      ).join('');
      const matchingRuleSections = ruleSections.filter(section =>
        targetDate >= section.startDate && targetDate <= section.endDate
      );
      const rules = matchingRuleSections.length ? '<div class="hover-rules">' +
        matchingRuleSections.map(section => {{
          const performance = '<div class="hover-rule-performance">' +
            '<span></span><span class="heading">Entry</span><span class="heading">' +
            (section.ongoing ? 'Current' : 'Exit') + '</span><span class="heading">Gain</span>' +
            section.performance.map(item => '<span class="symbol">' + escapeHtml(item.symbol) + '</span>' +
              '<span>' + escapeHtml(formatRulePrice(item.entry)) + '</span>' +
              '<span>' + escapeHtml(formatRulePrice(item.exit)) + '</span>' +
              '<span class="gain ' + (item.gain === null ? '' : item.gain >= 0 ? 'positive' : 'negative') + '">' +
              escapeHtml(formatRuleGain(item.gain)) + '</span>').join('') + '</div>';
          return '<section class="hover-rule" style="--rule-color:' + escapeHtml(section.color) + '">' +
            '<div class="hover-rule-name">' + escapeHtml(section.name) + '</div>' +
            '<div class="hover-rule-period">' + escapeHtml(formatRuleDate(section.startDate)) + ' – ' +
            escapeHtml(formatRuleDate(section.endDate)) + (section.ongoing ? ' · active' : '') + '</div>' +
            performance + '</section>';
        }}).join('') + '</div>' : '';
      const matchingSignals = ruleSignals.filter(signal => signal.targetDate === targetDate);
      const signals = matchingSignals.length ? '<div class="hover-signals">' +
        matchingSignals.map(signal => '<div class="hover-signal ' + signal.signal + '">' +
          (signal.signal === 'buy' ? '▲ Buy · ' : '▼ Sell · ') + escapeHtml(signal.name) +
          '</div>').join('') + '</div>' : '';
      tooltip.innerHTML = '<div class="hover-date">' +
        escapeHtml(formatRuleDate(targetDate)) + '</div><div class="hover-groups">' + groups + '</div>' + rules + signals;
      tooltip.style.display = 'block';
      const cursorX = hover.event.clientX - card.left;
      const cursorY = hover.event.clientY - card.top;
      const width = tooltip.offsetWidth; const height = tooltip.offsetHeight;
      const left = cursorX + width + 28 < card.width ? cursorX + 14 : cursorX - width - 14;
      const top = Math.min(Math.max(cursorY + 12, size.t), size.t + size.h - height);
      tooltip.style.left = Math.max(12, left) + 'px';
      tooltip.style.top = Math.max(size.t, top) + 'px';
    }};
    const rangeFromEvent = event => {{
      for (const axis of xAxes) {{
        const range = event[axis + '.range'];
        const start = event[axis + '.range[0]'];
        const end = event[axis + '.range[1]'];
        if (range && range.length === 2) return [range[0], range[1]];
        if (start && end) return [start, end];
      }}
      if (xAxes.some(axis => event[axis + '.autorange'])) return [first, last];
      return null;
    }};
    const adaptiveYRange = (start, end) => {{
      const startMs = new Date(start).getTime();
      const endMs = new Date(end).getTime();
      if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return Promise.resolve();
      const axes = {{}};
      const renderedTraces = plot._fullData || plot.data;
      renderedTraces.forEach(trace => {{
        if (trace.visible === false || trace.visible === 'legendonly' || !trace.x || !trace.y) return;
        const traceAxis = trace.yaxis || 'y';
        const layoutAxis = traceAxis === 'y' ? 'yaxis' : 'yaxis' + traceAxis.slice(1);
        const category = trace.meta && trace.meta.control_category;
        const entry = axes[layoutAxis] || (axes[layoutAxis] = {{values: [], categories: new Set()}});
        if (category) entry.categories.add(category);
        for (let index = 0; index < trace.x.length; index += 1) {{
          const time = new Date(trace.x[index]).getTime();
          const value = Number(trace.y[index]);
          if (time >= startMs && time <= endMs && Number.isFinite(value)) entry.values.push(value);
        }}
      }});
      const update = {{}};
      Object.entries(axes).forEach(([layoutAxis, entry]) => {{
        const axis = plot.layout[layoutAxis] || {{}};
        let values = entry.values;
        if (axis.type === 'log') values = values.filter(value => value > 0);
        if (!values.length) return;
        let minimum = Math.min(...values);
        let maximum = Math.max(...values);
        if (entry.categories.has('NDX SMA slopes') || entry.categories.has('NDX MACD')) {{
          minimum = Math.min(minimum, 0); maximum = Math.max(maximum, 0);
        }}
        if (axis.type === 'log') {{
          minimum = Math.log10(minimum); maximum = Math.log10(maximum);
          const padding = Math.max((maximum - minimum) * 0.08, 0.025);
          update[layoutAxis + '.range'] = [minimum - padding, maximum + padding];
        }} else {{
          const padding = Math.max((maximum - minimum) * 0.08, Math.abs(maximum || 1) * 0.01);
          let lower = minimum - padding; let upper = maximum + padding;
          if (entry.categories.has('NDX momentum')) {{
            lower = Math.max(0, lower); upper = Math.min(100, upper);
          }}
          update[layoutAxis + '.range'] = [lower, upper];
        }}
      }});
      return Object.keys(update).length ? Plotly.relayout(plot, update) : Promise.resolve();
    }};
    window.chartAdaptiveYRange = adaptiveYRange;
    const applyRange = (start, end) => {{
      if (!start || !end || start > end) return;
      from.value = start.slice(0, 10); to.value = end.slice(0, 10);
      const update = {{}};
      xAxes.forEach(axis => update[axis + '.range'] = [start, end]);
      Plotly.relayout(plot, update).then(() => adaptiveYRange(start, end));
    }};
    document.getElementById('apply-range').addEventListener('click', () => applyRange(from.value, to.value));
    document.querySelectorAll('[data-years]').forEach(button => button.addEventListener('click', () => {{
      if (button.dataset.years === 'all') return applyRange(first, last);
      const end = new Date(last + 'T00:00:00Z');
      const start = new Date(end); start.setUTCFullYear(start.getUTCFullYear() - Number(button.dataset.years));
      applyRange(start.toISOString().slice(0, 10), last);
    }}));
    document.querySelectorAll('input[data-traces]').forEach(input => input.addEventListener('change', () => {{
      const indices = input.dataset.traces.split(',').map(Number);
      Plotly.restyle(plot, {{visible: input.checked}}, indices)
        .then(() => adaptiveYRange(from.value, to.value));
    }}));
    document.querySelectorAll('[data-group-action]').forEach(button => button.addEventListener('click', () => {{
      const checked = button.dataset.groupAction === 'all';
      button.closest('fieldset').querySelectorAll('input[data-traces]').forEach(input => {{
        if (input.checked !== checked) {{ input.checked = checked; input.dispatchEvent(new Event('change')); }}
      }});
    }}));
    rulesetsElement.addEventListener('input', event => {{
      const target = event.target;
      const card = target.closest('.ruleset-card');
      if (!card || !target.dataset.field) return;
      const ruleset = rulesets.find(item => item.id === card.dataset.ruleset);
      if (!ruleset) return;
      const field = target.dataset.field;
      const conditionRow = target.closest('.condition-row');
      if (!conditionRow) {{
        if (field === 'type') {{
          if (target.value === 'buy_sell' && ruleset.type !== 'buy_sell') {{
            ruleset.type = 'buy_sell';
            ruleset.buy = cloneExpression(ruleset.expression);
            ruleset.sell = blankGroup('AND', '<=');
            delete ruleset.expression;
          }} else if (target.value === 'simple' && ruleset.type !== 'simple') {{
            ruleset.type = 'simple';
            ruleset.expression = cloneExpression(ruleset.buy);
            delete ruleset.buy; delete ruleset.sell;
          }}
          renderRulesets();
          return;
        }}
        if (field === 'groupJoin') {{
          const groupElement = target.closest('.expression-group');
          const signal = groupElement.dataset.signal;
          const group = expressionAtPath(
            expressionRoot(ruleset, signal), groupElement.dataset.path
          );
          group.join = target.value;
          applyRuleHighlights();
          return;
        }}
        ruleset[field] = field === 'enabled' ? target.checked : target.value;
        applyRuleHighlights();
        return;
      }}
      const signal = conditionRow.dataset.signal;
      const condition = expressionAtPath(
        expressionRoot(ruleset, signal), conditionRow.dataset.path
      );
      if (field === 'rightType') {{
        condition.right = target.value === 'number'
          ? {{type: 'number', value: 0}}
          : {{type: 'indicator', value: defaultRightOperand()}};
        renderRulesets();
        return;
      }}
      if (field === 'right') {{
        condition.right.value = condition.right.type === 'number' && target.value !== ''
          ? Number(target.value) : target.value;
      }} else {{
        condition[field] = target.value;
      }}
      applyRuleHighlights();
    }});
    rulesetsElement.addEventListener('click', event => {{
      const button = event.target.closest('button[data-action]');
      if (!button) return;
      const card = button.closest('.ruleset-card');
      const rulesetIndex = rulesets.findIndex(item => item.id === card.dataset.ruleset);
      if (rulesetIndex < 0) return;
      const ruleset = rulesets[rulesetIndex];
      if (button.dataset.action === 'remove-ruleset') {{
        rulesets.splice(rulesetIndex, 1);
        renderRulesets();
        return;
      }}
      const groupElement = button.closest('.expression-group');
      if (!groupElement) return;
      const signal = groupElement.dataset.signal;
      const root = expressionRoot(ruleset, signal);
      const group = expressionAtPath(root, groupElement.dataset.path);
      const countNodes = node => 1 + (node.kind === 'group'
        ? node.items.reduce((total, item) => total + countNodes(item), 0) : 0);
      if (button.dataset.action === 'add-condition' || button.dataset.action === 'add-group') {{
        const addedNodes = button.dataset.action === 'add-group' ? 2 : 1;
        const depth = groupElement.dataset.path
          ? groupElement.dataset.path.split('.').length : 0;
        if (group.items.length >= 20) {{
          setRulesStatus('A group can contain at most 20 direct items.', true);
          return;
        }}
        if (countNodes(root) + addedNodes > MAX_RULE_NODES) {{
          setRulesStatus('An expression can contain at most ' + MAX_RULE_NODES + ' groups and conditions.', true);
          return;
        }}
        if (button.dataset.action === 'add-group' && depth >= MAX_RULE_DEPTH - 1) {{
          setRulesStatus('An expression can contain at most ' + MAX_RULE_DEPTH + ' nested group levels.', true);
          return;
        }}
        group.items.push(button.dataset.action === 'add-group' ? blankGroup() : blankCondition());
        setRulesStatus('');
      }}
      if (button.dataset.action === 'remove-condition') {{
        const path = button.closest('.condition-row').dataset.path.split('.');
        const itemIndex = Number(path.pop());
        const parent = expressionAtPath(root, path.join('.'));
        if (parent.items.length > 1) parent.items.splice(itemIndex, 1);
      }}
      if (button.dataset.action === 'remove-group') {{
        const path = groupElement.dataset.path.split('.');
        const itemIndex = Number(path.pop());
        const parent = expressionAtPath(root, path.join('.'));
        if (parent.items.length > 1) parent.items.splice(itemIndex, 1);
      }}
      renderRulesets();
    }});
    document.getElementById('add-ruleset').addEventListener('click', () => {{
      rulesets.push(blankRuleset());
      renderRulesets();
    }});
    document.getElementById('copy-rules').addEventListener('click', async () => {{
      const text = JSON.stringify(exportRulesets(), null, 2);
      rulesJson.value = text;
      try {{
        await navigator.clipboard.writeText(text);
        setRulesStatus('Ruleset JSON copied to the clipboard.');
      }} catch (error) {{
        rulesJson.focus(); rulesJson.select();
        const copied = document.execCommand && document.execCommand('copy');
        setRulesStatus(copied ? 'Ruleset JSON copied to the clipboard.' : 'Ruleset JSON is ready to copy from the box.');
      }}
    }});
    document.getElementById('import-rules').addEventListener('click', () => {{
      try {{
        const imported = JSON.parse(rulesJson.value);
        rulesets = normaliseRulesets(imported);
        renderRulesets();
        setRulesStatus('Imported ' + rulesets.length + ' ruleset' + (rulesets.length === 1 ? '.' : 's.'));
      }} catch (error) {{
        setRulesStatus('Import failed: ' + error.message, true);
      }}
    }});
    window.chartRulesets = {{
      export: exportRulesets,
      import: payload => {{ rulesets = normaliseRulesets(payload); renderRulesets(); }},
      apply: applyRuleHighlights
    }};
    plot.on('plotly_relayout', event => {{
      const range = rangeFromEvent(event);
      if (!range) {{
        if (Object.keys(event).some(key => key.endsWith('.type'))) {{
          adaptiveYRange(from.value, to.value);
        }}
        return;
      }}
      const start = String(range[0]).slice(0, 10); const end = String(range[1]).slice(0, 10);
      from.value = start; to.value = end;
      adaptiveYRange(start, end);
    }});
    plot.on('plotly_hover', hover => {{
      if (!hover.event) return;
      const card = plot.parentElement.getBoundingClientRect();
      const plotRect = plot.getBoundingClientRect();
      const size = plot._fullLayout._size;
      crosshair.style.left = (hover.event.clientX - card.left) + 'px';
      crosshair.style.top = (plotRect.top - card.top + size.t) + 'px';
      crosshair.style.height = size.h + 'px';
      crosshair.style.display = 'block';
      updateGlobalTooltip(hover, card, size);
    }});
    plot.on('plotly_unhover', () => {{
      crosshair.style.display = 'none'; tooltip.style.display = 'none';
    }});
    renderRulesets();
    adaptiveYRange(first, last);
  }})();
  </script>
</body>
</html>"""


def build_chart_figure(
    datasets: dict[str, pd.DataFrame],
    *,
    title: str = "Nasdaq-100 market & indicator explorer",
) -> go.Figure:
    """Build synced price and NDX-indicator panels for the HTML explorer."""
    if not datasets:
        raise ValueError("At least one dataset is required")
    primary_symbol = "NDX" if "NDX" in datasets else next(iter(datasets))
    symbols = [primary_symbol, *(symbol for symbol in datasets if symbol != primary_symbol)]
    primary = datasets[primary_symbol].sort_values("session_date")
    price_rows = len(symbols)
    total_rows = price_rows + 4
    titles = [f"{_display_symbol(symbol)} — adjusted close" for symbol in symbols]
    titles += [
        f"{_display_symbol(primary_symbol)} — RSI",
        f"{_display_symbol(primary_symbol)} — SMA slope",
        f"{_display_symbol(primary_symbol)} — ATR",
        f"{_display_symbol(primary_symbol)} — MACD(12, 26, 9)",
    ]
    base_heights = [0.24, *([0.13] * (price_rows - 1)), 0.14, 0.13, 0.10, 0.13]
    height_total = sum(base_heights)
    row_heights = [value / height_total for value in base_heights]
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=row_heights,
        subplot_titles=titles,
    )

    for row, symbol in enumerate(symbols, start=1):
        frame = datasets[symbol].sort_values("session_date")
        fig.add_trace(
            _line(
                frame,
                "price",
                f"{_display_symbol(symbol)} price",
                _INK if symbol == primary_symbol else (_GREEN if symbol == "TQQQ" else _RED),
                category="Prices",
                width=2.2,
                visible=True,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(type="log", title_text="Price", row=row, col=1)

    for window in CHART_SMA_WINDOWS:
        column = moving_average_column(window)
        fig.add_trace(
            _line(
                primary,
                column,
                f"SMA({window})",
                _SMA_COLORS[window],
                category="NDX trend",
                width=1.35,
                visible=window in (50, 200),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        _line(primary, "ema_5", "EMA(5)", _TEAL, category="NDX trend", visible=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        _line(
            primary,
            "chandelier_long",
            "Chandelier long exit",
            _RED,
            category="NDX trend",
            dash="dot",
            visible=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _line(
            primary,
            "chandelier_short",
            "Chandelier short exit",
            _GREEN,
            category="NDX trend",
            dash="dot",
            visible=False,
        ),
        row=1,
        col=1,
    )

    rsi_row = price_rows + 1
    for period, color, visible in ((2, _RED, False), (3, _GOLD, False), (14, _VIOLET, True)):
        fig.add_trace(
            _line(
                primary,
                f"rsi_{period}",
                f"RSI({period})",
                color,
                category="NDX momentum",
                visible=visible,
            ),
            row=rsi_row,
            col=1,
        )
    fig.update_yaxes(range=[0, 100], title_text="RSI", row=rsi_row, col=1)
    for level, color in ((30, "#adb5bd"), (50, "#ced4da"), (70, "#adb5bd")):
        fig.add_hline(y=level, line_width=1, line_dash="dot", line_color=color, row=rsi_row, col=1)

    slope_row = price_rows + 2
    for window in CHART_SLOPE_WINDOWS:
        fig.add_trace(
            _line(
                primary,
                f"sma_slope_{window}",
                f"SMA slope({window})",
                _SMA_COLORS[window],
                category="NDX SMA slopes",
                visible=window in (20, 50),
            ),
            row=slope_row,
            col=1,
        )
    fig.add_hline(y=0, line_width=1, line_color=_AXIS, row=slope_row, col=1)
    fig.update_yaxes(tickformat=".1%", title_text="1-day change", row=slope_row, col=1)

    atr_row = price_rows + 3
    for period in ATR_PERIODS:
        fig.add_trace(
            _line(
                primary,
                f"atr_{period}",
                f"ATR({period})",
                _ATR_COLORS[period],
                category="NDX volatility",
                visible=period == 14,
            ),
            row=atr_row,
            col=1,
        )
    fig.update_yaxes(title_text="Index points", row=atr_row, col=1)

    macd_row = price_rows + 4
    fig.add_trace(
        go.Bar(
            x=primary["session_date"],
            y=primary["macd_histogram"],
            name="MACD histogram",
            marker_color=[_GREEN if value >= 0 else _RED for value in primary["macd_histogram"].fillna(0)],
            opacity=0.5,
            visible=True,
            showlegend=False,
            meta={"control_category": "NDX MACD", "control_label": "MACD histogram"},
        ),
        row=macd_row,
        col=1,
    )
    fig.add_trace(
        _line(primary, "macd_line", "MACD line", _BLUE, category="NDX MACD", visible=True),
        row=macd_row,
        col=1,
    )
    fig.add_trace(
        _line(primary, "macd_signal", "MACD signal", _GOLD, category="NDX MACD", visible=True),
        row=macd_row,
        col=1,
    )
    for signal, label, color, symbol in (
        ("buy", "Buy signals", _GREEN, "triangle-up"),
        ("sell", "Sell signals", _RED, "triangle-down"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=label,
                mode="markers",
                marker={
                    "color": color,
                    "line": {"color": _PANEL, "width": 1},
                    "size": 14,
                    "symbol": symbol,
                },
                showlegend=False,
                meta={"rule_signal": signal},
                hovertemplate=f"%{{text}}<br>%{{x|%Y-%m-%d}}<extra>{label}</extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_hline(y=0, line_width=1, line_color=_AXIS, row=macd_row, col=1)
    fig.update_yaxes(title_text="Index points", row=macd_row, col=1)

    fig.update_xaxes(
        gridcolor=_GRID,
        zerolinecolor=_AXIS,
        tickfont={"color": _MUTED},
        matches="x",
        showspikes=True,
        spikecolor=_MUTED,
        spikedash="solid",
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
    )
    fig.update_xaxes(rangeslider={"visible": True, "thickness": 0.045}, row=total_rows, col=1)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_AXIS, tickfont={"color": _MUTED})
    for annotation in fig.layout.annotations:
        annotation.font = {"color": _INK, "size": 12}

    price_axis_keys = ["yaxis", *[f"yaxis{row}" for row in range(2, price_rows + 1)]]
    scale_buttons = [
        {
            "label": label,
            "method": "relayout",
            "args": [{f"{key}.type": scale for key in price_axis_keys}],
        }
        for label, scale in (("Log", "log"), ("Linear", "linear"))
    ]
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=_PANEL,
        plot_bgcolor=_PANEL,
        font={"family": "Inter, ui-sans-serif, system-ui, sans-serif", "color": _MUTED},
        hovermode="x unified",
        hoversubplots="axis",
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": scale_buttons,
                "active": 0,
                "x": 1.0,
                "xanchor": "right",
                "y": 1.012,
                "yanchor": "bottom",
                "bgcolor": _PANEL,
                "bordercolor": _AXIS,
                "font": {"color": _INK},
            }
        ],
        height=max(1400, 235 * total_rows),
        margin={"t": 74, "b": 50, "l": 72, "r": 28},
        title={"text": title, "font": {"color": _INK, "size": 19}, "x": 0.01},
        bargap=0,
    )
    return fig


def _line(
    frame: pd.DataFrame,
    column: str,
    label: str,
    color: str,
    *,
    category: str,
    visible: bool,
    width: float = 1.5,
    dash: str = "solid",
) -> go.Scattergl:
    if column not in frame.columns:
        raise ValueError(f"Chart dataset is missing required column {column!r}")
    return go.Scattergl(
        x=frame["session_date"],
        y=frame[column],
        name=label,
        mode="lines",
        line={"color": color, "width": width, "dash": dash},
        visible=True if visible else "legendonly",
        showlegend=False,
        meta={"control_category": category, "control_label": label},
        hovertemplate=f"%{{y:,.4g}}<extra>{html.escape(label)}</extra>",
    )


def _display_symbol(symbol: str) -> str:
    return "^NDX" if symbol == "NDX" else symbol


def _xaxis_layout_keys(count: int) -> tuple[str, ...]:
    return ("xaxis", *(f"xaxis{index}" for index in range(2, count + 1)))
