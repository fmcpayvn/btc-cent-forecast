"""btc-cent-forecast-01 entry (Python 3.13, standard library only).

Forecasts Bitcoin's US-dollar price 240 minutes after each checkpoint.

Model: one-hour momentum. The forecast is the checkpoint's last close
moved a fraction of the last hour's log return:

    forecast = last_close * exp(0.30 * (log(last) - log(close 60 min ago)))

Why: the score is mean absolute error, so the best guess is the conditional
median of the price 4 hours out. Across 107 four-hour checkpoints of public
1-minute BTC closes (21 days, Coinbase BTC-USD) the 1-hour return kept a
positive correlation with the 4-hour-ahead return (+0.53 in the first half
of the sample, +0.28 in the second), while the 4-hour return's own
correlation flipped sign between halves and a fitted ridge on 13 features
was much worse out of sample. On the kit's five practice cases with known
answers (the real Orca series) this rule cut the mean error from 47,055
cents (persistence) to 41,140 cents at this coefficient. Larger coefficients
kept helping on the trending practice day but stopped helping out of sample
on the public series, so 0.30 is the conservative middle: about a 0.7%
improvement on choppy data, much more when the market trends.

Runs offline, no network, deterministic: the same input always writes the
same bytes. Reads every <id>.in in the input directory and writes one
<id>.out per case: {"forecasts": [{"case": "<id>", "price": <number>}] }.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999

# Fraction of the last hour's log return carried into the 4-hour forecast.
MOMENTUM = 0.30
# The return is measured over this many minutes (60 minutes back from the
# checkpoint close, i.e. close[-61] to close[-1]).
LOOKBACK_MINUTES = 60
# Cap on the hourly log return used: the largest hourly move in 21 days of
# public 1-minute closes was 4.5%, so a move beyond 5% is a wild print in
# this thin pool, not a trend to chase; capping it keeps one bad minute
# from moving the forecast by more than about 1.5%.
MAX_HOURLY_MOVE = 0.05


def price5(x):
    """Keep a price on the grid's 5 decimals and inside (0, 10,000,000)."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def forecast(prices):
    """One-hour-momentum forecast from a case's closes (oldest to newest)."""
    if not prices:
        raise ValueError("market has no prices")
    last = prices[-1]
    if not isinstance(last, (int, float)) or not math.isfinite(last) or last <= 0:
        raise ValueError("last price is not a positive finite number")
    n = min(LOOKBACK_MINUTES, len(prices) - 1)
    if n <= 0:
        return price5(last)
    ref = prices[-1 - n]
    if not isinstance(ref, (int, float)) or not math.isfinite(ref) or ref <= 0:
        return price5(last)
    r60 = math.log(last) - math.log(ref)
    r60 = max(-MAX_HOURLY_MOVE, min(MAX_HOURLY_MOVE, r60))
    return price5(last * math.exp(MOMENTUM * r60))


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(INPUT_DIR)
    inputs = sorted(f for f in in_dir.iterdir() if f.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        case = json.loads(path.read_text(encoding="utf-8"))
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError(f"{path.name} does not hold exactly one market")
        guess = forecast(markets[0]["prices"])
        out = {"forecasts": [{"case": case_id, "price": guess}]}
        (out_dir / f"{case_id}.out").write_text(
            json.dumps(out) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
