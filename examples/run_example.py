#!/usr/bin/env python3
"""End-to-end example: evaluate sample strategy + show kill-policy thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtf import StrategyBundle, calibrate_kill_policy, evaluate_strategy  # noqa: E402
from rtf.pipeline import render_markdown  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    # Ensure sample data exists
    if not (DATA / "returns.csv").exists():
        from generate_sample_data import main as gen

        gen()

    import pandas as pd

    returns = pd.read_csv(DATA / "returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
    positions = pd.read_csv(DATA / "positions.csv", index_col=0, parse_dates=True).iloc[:, 0]
    asset = pd.read_csv(DATA / "asset_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
    claimed = json.loads((DATA / "claimed_metrics.json").read_text())
    grid = json.loads((DATA / "param_grid.json").read_text())

    bundle = StrategyBundle(
        returns=returns,
        oos_start="2024-01-01",
        positions=positions,
        asset_returns=asset,
        n_trials=50,  # pretend we tried 50 windows / variants
        claimed_metrics=claimed,
        cost_bps_base=5.0,
        parameter_grid_results=grid,
        meta={"strategy_id": "demo_momentum", "symbol": "SYNTH", "owner": "lab"},
    )

    report = evaluate_strategy(
        bundle,
        calibrate_kills=True,
        n_null=300,
        n_kill_paths=1500,
        seed=42,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    report.save(OUT / "report.json")
    print(render_markdown(report))

    # Also dump a compact kill summary
    if report.kill_policy:
        cal = report.kill_policy.calibration
        print("\n--- Kill calibration snapshot ---")
        print(json.dumps(cal["drawdown_simultaneous"], indent=2))
        print(json.dumps(cal["tuw_simultaneous_bars"], indent=2))


if __name__ == "__main__":
    main()
