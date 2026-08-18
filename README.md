# NDX market chart

Public, interactive historical chart for the Nasdaq-100 (`^NDX`), TQQQ, and
SQQQ. It includes technical indicators, synchronized crosshairs, adaptive
vertical scales, date-range controls, and colour-coded Simple and Buy/Sell
rulesets.

The published chart is generated after the US market closes each weekday and
can also be regenerated manually from GitHub Actions.

## View the chart

After the publishing pull request is merged, the chart will be available at:

<https://peteoh.github.io/ndx-market-chart/>

The page is static. Rulesets created in the browser are not sent to a server;
use **Copy JSON** and **Import pasted JSON** to save or transfer them.

## Build from scratch

Python 3.12 or later is required.

```powershell
git clone https://github.com/PeteOH/ndx-market-chart.git
Set-Location ndx-market-chart
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[dev]"
ndx-chart --output dist/index.html --verbose
Start-Process dist/index.html
```

The default history begins on `2010-01-01` and ends today. Override either
date when reproducing an older chart:

```powershell
ndx-chart --start 2010-01-01 --end 2025-12-31 --output dist/index.html --verbose
```

Every command that performs the update supports `--verbose`, which reports
each download, calculation, and output step.

## Indicators

The NDX panels include:

- SMA(3), SMA(5), SMA(7), SMA(10), SMA(20), SMA(50), SMA(200), SMA(250)
- EMA(5)
- RSI(2), RSI(3), RSI(14)
- SMA slope(5), (7), (10), (20), (50), (200), and (250)
- Chandelier Exit, ATR(5), ATR(7), ATR(10), ATR(14), and MACD(12,26,9)

Hover the information badge beside an indicator for a concise explanation.
The ruleset editor can compare an indicator with a number or another indicator
using `<`, `=`, `>`, `>=`, or `<=`.

## Nested AND/OR rulesets

Every Simple, Buy, and Sell expression starts with a root group. Each group has
its own AND/OR selector and can contain conditions or more groups. Select
**+ Group** to create parentheses and nest combinations to multiple levels.

For example, `(a OR b) AND (c OR d)` is represented by an AND root containing
two OR groups:

```json
{
  "version": 2,
  "rulesets": [
    {
      "name": "Nested example",
      "type": "simple",
      "enabled": true,
      "color": "#51cf66",
      "expression": {
        "join": "AND",
        "items": [
          {
            "join": "OR",
            "items": [
              {
                "left": "SMA(50)",
                "operator": ">=",
                "right": {"type": "indicator", "value": "SMA(200)"}
              },
              {
                "left": "EMA(5)",
                "operator": ">=",
                "right": {"type": "indicator", "value": "SMA(20)"}
              }
            ]
          },
          {
            "join": "OR",
            "items": [
              {
                "left": "RSI(14)",
                "operator": ">=",
                "right": {"type": "number", "value": 70}
              },
              {
                "left": "ATR(5)",
                "operator": ">=",
                "right": {"type": "indicator", "value": "ATR(14)"}
              }
            ]
          }
        ]
      }
    }
  ]
}
```

For `(a AND b) OR (c AND d)`, change the root to `"join": "OR"` and each
nested group to `"join": "AND"`. Buy/Sell rulesets use this same recursive
group shape independently under `buy` and `sell`.

**Copy JSON** now exports version 2. Existing version-1 flat rulesets remain
importable and are automatically converted to a one-level expression.

## Automated publication

`.github/workflows/pages.yml` runs at 18:35 New York time on weekdays, more
than two hours after the regular US market close. It performs these steps:

1. Installs the chart generator on a clean GitHub-hosted runner.
2. Downloads the full NDX, TQQQ, and SQQQ history.
3. Recalculates every indicator.
4. Builds `site/index.html` with `--verbose` enabled.
5. Uploads the generated site as a GitHub Pages artifact and deploys it.

The generated 9–10 MB HTML file is deliberately not committed on every run;
this keeps Git history small. To update immediately, open **Actions**, choose
**Update and publish NDX chart**, and select **Run workflow**.

## Test and lint

```powershell
python -m pytest
python -m ruff check .
```

## Data and investment disclaimer

Data is obtained through the unofficial `yfinance` library and Yahoo Finance.
Review the data provider's terms before redistributing the chart or using it
commercially. The data can be delayed, incomplete, or revised.

This project is for personal research and education. It is not investment
advice, and an indicator or ruleset match is not a recommendation to buy, hold,
or sell any security.
