"""Generate synthetic strategy artefacts for the README walkthrough.

Produces a mild positive-edge momentum rule that usually clears integrity /
cost / OOS-hold checks but fails DSR at an honest trial count — the common
"code is honest, edge not proven" outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(21)
    idx = pd.date_range("2018-01-01", "2025-09-30", freq="B", tz="UTC")

    # Persistent trend + noise → momentum has a real (modest) edge
    shock = rng.normal(0.0, 0.008, len(idx))
    drift = np.cumsum(rng.normal(0.0, 0.00015, len(idx)))
    asset_r = pd.Series(0.00015 + drift / 50.0 + shock, index=idx, name="asset_returns")

    window = 60
    mom = asset_r.rolling(window).mean().shift(1)
    pos = np.sign(mom).fillna(0.0)
    pos.name = "positions"

    held = pos.shift(1)
    turnover = held.diff().abs().fillna(held.abs())
    cost_bps = 5.0
    strat = held * asset_r - (cost_bps / 1e4) * turnover
    strat.name = "returns"

    strat.to_csv(OUT / "returns.csv", header=True)
    pos.to_csv(OUT / "positions.csv", header=True)
    asset_r.to_csv(OUT / "asset_returns.csv", header=True)

    import sys

    sys.path.insert(0, str(OUT.parents[1]))
    from rtf.metrics import sharpe, summarize

    claimed = summarize(strat[strat.index < "2025-10-01"])
    (OUT / "claimed_metrics.json").write_text(
        json.dumps(
            {
                "sharpe": claimed["sharpe"],
                "cagr": claimed["cagr"],
                "maxdd": claimed["maxdd"],
            },
            indent=2,
        )
    )

    oos_start = pd.Timestamp("2024-01-01", tz="UTC")
    grid = []
    for w in range(20, 101, 5):
        m = asset_r.rolling(w).mean().shift(1)
        p = np.sign(m).fillna(0.0)
        h = p.shift(1)
        s = h * asset_r - (cost_bps / 1e4) * h.diff().abs().fillna(h.abs())
        is_s = s[s.index < oos_start].dropna()
        oo_s = s[(s.index >= oos_start) & (s.index < "2025-10-01")].dropna()
        grid.append(
            {
                "window": w,
                "sharpe": sharpe(s.dropna()),
                "is_sharpe": sharpe(is_s),
                "oos_sharpe": sharpe(oo_s),
            }
        )
    (OUT / "param_grid.json").write_text(json.dumps(grid, indent=2))

    print(f"Wrote sample data to {OUT}")
    print(
        f"n={len(strat)}, full_sharpe≈{claimed['sharpe']:.2f}, "
        f"IS≈{sharpe(strat[strat.index < oos_start]):.2f}, "
        f"OOS≈{sharpe(strat[(strat.index >= oos_start) & (strat.index < '2025-10-01')]):.2f}"
    )


if __name__ == "__main__":
    main()
