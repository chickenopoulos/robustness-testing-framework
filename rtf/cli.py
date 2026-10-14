"""Command-line interface: `rtf evaluate` and `rtf calibrate-kills`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .bundle import load_bundle
from .kill_switch.policy import calibrate_kill_policy, evaluate_live
from .pipeline import evaluate_strategy, render_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rtf",
        description="Robustness Testing Framework — post-search battery + kill-switch calibration",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("evaluate", help="Run Part A (+ Part C calibration)")
    p_eval.add_argument("--returns", required=True, help="CSV of strategy net returns")
    p_eval.add_argument("--oos-start", required=True, help="OOS start date (YYYY-MM-DD)")
    p_eval.add_argument("--positions", default=None, help="CSV of positions [-1,1]")
    p_eval.add_argument("--asset-returns", default=None, help="CSV of underlying returns")
    p_eval.add_argument("--n-trials", type=int, default=1, help="Honest trial count N")
    p_eval.add_argument("--cost-bps", type=float, default=0.0, help="Baseline one-way cost (bps)")
    p_eval.add_argument("--sealed-start", default="2025-10-01")
    p_eval.add_argument("--claimed-metrics", default=None, help="JSON ledger to reconcile")
    p_eval.add_argument("--strategy-id", default="strategy")
    p_eval.add_argument("--out", default="rtf_report.json")
    p_eval.add_argument("--no-kills", action="store_true", help="Skip kill-policy calibration")
    p_eval.add_argument("--n-null", type=int, default=400)
    p_eval.add_argument("--n-kill-paths", type=int, default=2000)
    p_eval.add_argument("--seed", type=int, default=42)

    p_kill = sub.add_parser("calibrate-kills", help="Calibrate Part C kill policy only")
    p_kill.add_argument("--returns", required=True)
    p_kill.add_argument("--strategy-id", default="strategy")
    p_kill.add_argument("--haircut", type=float, default=0.5)
    p_kill.add_argument("--horizon-years", type=float, default=3.0)
    p_kill.add_argument("--n-paths", type=int, default=5000)
    p_kill.add_argument("--false-kill", type=float, default=0.05)
    p_kill.add_argument("--out", default="kill_policy.yaml")
    p_kill.add_argument("--seed", type=int, default=42)

    p_live = sub.add_parser("check-live", help="Apply a kill policy to live returns")
    p_live.add_argument("--live-returns", required=True)
    p_live.add_argument("--policy", required=True, help="JSON or YAML kill policy")
    p_live.add_argument("--tracking-z", type=float, default=None)
    p_live.add_argument("--out", default="live_check.json")

    args = parser.parse_args(argv)

    if args.cmd == "evaluate":
        return _cmd_evaluate(args)
    if args.cmd == "calibrate-kills":
        return _cmd_calibrate(args)
    if args.cmd == "check-live":
        return _cmd_live(args)
    return 1


def _cmd_evaluate(args: argparse.Namespace) -> int:
    bundle = load_bundle(
        args.returns,
        args.oos_start,
        positions_path=args.positions,
        asset_returns_path=args.asset_returns,
        n_trials=args.n_trials,
        cost_bps_base=args.cost_bps,
        sealed_start=args.sealed_start,
        claimed_metrics_path=args.claimed_metrics,
        meta={"strategy_id": args.strategy_id},
    )
    report = evaluate_strategy(
        bundle,
        calibrate_kills=not args.no_kills,
        n_null=args.n_null,
        n_kill_paths=args.n_kill_paths,
        seed=args.seed,
    )
    out = Path(args.out)
    report.save(out)
    print(render_markdown(report))
    print(f"\nWrote {out} and {out.with_suffix('.md')}")
    if report.kill_policy is not None:
        print(f"Wrote {out.with_suffix('.kill_policy.yaml')}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    r = _read_returns(args.returns)
    report = calibrate_kill_policy(
        r,
        strategy_id=args.strategy_id,
        sharpe_haircut=args.haircut,
        horizon_years=args.horizon_years,
        n_paths=args.n_paths,
        false_kill_budget=args.false_kill,
        seed=args.seed,
    )
    out = Path(args.out)
    out.write_text(report.to_yaml())
    meta = out.with_suffix(".calibration.json")
    meta.write_text(json.dumps(report.calibration, indent=2, default=str))
    print(f"policy_hash={report.policy_hash}")
    print(f"Wrote {out} and {meta}")
    return 0


def _cmd_live(args: argparse.Namespace) -> int:
    import yaml

    policy_path = Path(args.policy)
    text = policy_path.read_text()
    if policy_path.suffix in {".yaml", ".yml"}:
        policy = yaml.safe_load(text)
    else:
        raw = json.loads(text)
        policy = raw.get("policy", raw)

    live = _read_returns(args.live_returns)
    result = evaluate_live(live, policy, tracking_z=args.tracking_z)
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    return 0


def _read_returns(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if len(df.columns) >= 2:
        return pd.Series(
            df.iloc[:, 1].values,
            index=pd.to_datetime(df.iloc[:, 0], utc=True),
            name="returns",
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0]


if __name__ == "__main__":
    sys.exit(main())
