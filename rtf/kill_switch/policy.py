"""Default kill policy template (C.11) with quantitative calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from ..metrics import (
    equity_curve,
    max_drawdown,
    periods_per_year,
    probabilistic_sharpe,
    sharpe,
    time_under_water,
)
from ..types import ResponseLevel
from .simulate import (
    cusum_threshold_calibration,
    haircut_returns,
    path_maxdd_to_date,
    path_tuw_to_date,
    politis_white_mean_block_length,
    simultaneous_threshold,
    stationary_bootstrap_paths,
)


@dataclass
class KillPolicyReport:
    policy: dict[str, Any]
    policy_hash: str
    calibration: dict[str, Any]
    bands: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "policy": self.policy,
            "calibration": self.calibration,
            "bands": self.bands,
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.policy, sort_keys=False)


def calibrate_kill_policy(
    backtest_returns: pd.Series,
    *,
    strategy_id: str = "strategy",
    frozen_code: str = "unknown",
    owner: str = "research",
    sharpe_haircut: float = 0.5,
    horizon_years: float = 3.0,
    n_paths: int = 5000,
    false_kill_budget: float = 0.05,
    seed: int = 42,
    holding_period_bars: int = 1,
) -> KillPolicyReport:
    """Build a hashed kill policy with simulation-calibrated thresholds.

    Joint false-kill budget is split across the three statistical monitors
    (MaxDD, TuW, CUSUM) via Bonferroni: α_i = α / 3.
    """
    r = pd.Series(backtest_returns, dtype=float).dropna()
    if len(r) < 60:
        raise ValueError("Need ≥60 backtest returns to calibrate kill policy")

    q = periods_per_year(r.index)
    horizon = int(round(horizon_years * q))
    haircut_r = haircut_returns(r, haircut=sharpe_haircut)
    mean_block = max(
        politis_white_mean_block_length(haircut_r.values),
        holding_period_bars,
    )

    paths = stationary_bootstrap_paths(
        haircut_r.values,
        horizon=horizon,
        n_paths=n_paths,
        mean_block=mean_block,
        seed=seed,
    )

    alpha_i = false_kill_budget / 3.0

    # Drawdown bands (additive % of capital for calibration tables)
    dd_paths = path_maxdd_to_date(paths, additive=True)
    # dd is negative; more negative = worse. Simultaneous path-wise bands.
    dd_pause = simultaneous_threshold(dd_paths, alpha_i, worse="low")
    dd_reduce = simultaneous_threshold(dd_paths, min(0.10, 3 * alpha_i), worse="low")
    dd_warn = simultaneous_threshold(dd_paths, min(0.20, 5 * alpha_i), worse="low")

    # Time under water
    tuw_paths = path_tuw_to_date(paths, additive=True)
    tuw_pause = simultaneous_threshold(tuw_paths.astype(float), alpha_i, worse="high")
    tuw_reduce = simultaneous_threshold(
        tuw_paths.astype(float), min(0.10, 3 * alpha_i), worse="high"
    )
    tuw_warn = simultaneous_threshold(
        tuw_paths.astype(float), min(0.20, 5 * alpha_i), worse="high"
    )

    # CUSUM: healthy mean = haircut mean; dead = 0
    mu_h = float(haircut_r.mean())
    h_cusum = cusum_threshold_calibration(paths, mu_h, 0.0, alpha_i)

    # Hard capital cap: 2 × annualised vol of haircut series
    vol = float(haircut_r.std(ddof=1) * np.sqrt(q))
    hard_cap = -2.0 * vol  # as fractional equity loss (compounding approx)

    # Snapshot bands at 1y / 2y / 3y for the report
    def _snap(stat, idxs, qs, worse="low"):
        out = {}
        for name, idx in idxs.items():
            t = min(idx, stat.shape[1] - 1)
            col = stat[:, : t + 1]
            if worse == "low":
                extreme = col.min(axis=1)
                out[name] = {f"q{int(q_ * 100)}": float(np.quantile(extreme, q_)) for q_ in qs}
            else:
                extreme = col.max(axis=1)
                out[name] = {
                    f"q{int(q_ * 100)}": float(np.quantile(extreme, q_)) for q_ in qs
                }
        return out

    year_idx = {
        "1y": int(q) - 1,
        "2y": int(2 * q) - 1,
        "3y": horizon - 1,
    }

    policy = {
        "version": 1,
        "strategy_id": strategy_id,
        "frozen_code": frozen_code,
        "owner": owner,
        "reference": {
            "returns": "backtest_net_daily",
            "sharpe_haircut": sharpe_haircut,
            "simulator": "stationary_bootstrap",
            "mean_block_length": mean_block,
            "paths": n_paths,
            "horizon_years": horizon_years,
            "horizon_bars": horizon,
            "periods_per_year": q,
            "false_kill_budget": false_kill_budget,
            "per_monitor_alpha": alpha_i,
            "seed": seed,
            "backtest_sharpe": sharpe(r),
            "haircut_sharpe": sharpe(haircut_r),
        },
        "monitors": {
            "max_drawdown": {
                "warn": dd_warn,
                "reduce": dd_reduce,
                "pause": dd_pause,
                "hard_cap": hard_cap,
                "unit": "additive_return_drawdown",
                "note": "More negative = worse. Simultaneous bands for joint false-kill budget.",
            },
            "time_under_water": {
                "warn": int(round(tuw_warn)),
                "reduce": int(round(tuw_reduce)),
                "pause": int(round(tuw_pause)),
                "unit": "bars",
            },
            "cusum_mean_drop": {
                "mu_healthy": mu_h,
                "mu_dead": 0.0,
                "h": h_cusum,
                "warn": 0.5 * h_cusum,
            },
            "bayes_p_negative": {"reduce": 0.5, "pause": 0.8, "retire_after_mintrl": 0.9},
            "tracking_z": {"warn": -2.0, "pause": -3.0, "sign_agreement_min": 0.95},
            "slippage_bps_20fills": {"reduce_mult": 2.0, "pause_mult": 3.0},
            "operational": {
                "data_staleness": {"warn_bars": 1, "pause_bars": 2},
                "tracker_staleness": {"warn_rebalances": 1, "pause_rebalances": 2},
                "config_drift": "block_deploy_and_pause",
                "position_mismatch": {"warn_bars": 1, "flatten_bars": 2},
            },
        },
        "responses": [e.value for e in ResponseLevel],
        "re_entry": {
            "after_operational_pause": (
                "cause fixed + 20 shadow days with |tracking_z| < 1 -> half size"
            ),
            "after_statistical_pause": (
                ">= 60 paper days, paper CUSUM < h/2, paper MaxDD < warn -> half size"
            ),
            "full_size": "40 further days with all monitors below WARN",
        },
        "retire": (
            "paused > 6 months OR bayes_p_negative > 0.9 after MinTRL "
            "OR alpha break with post-break alpha <= 0"
        ),
        "sizing": (
            "start at 25-50% of target risk; scale up only as live PSR(0) rises "
            "(e.g. 50% at PSR 0.7, 100% at PSR 0.9)"
        ),
        "amendments": "dated, reasoned, restart the false-kill budget",
    }

    payload = json.dumps(policy, sort_keys=True, default=str).encode()
    policy_hash = hashlib.sha256(payload).hexdigest()
    policy["policy_hash"] = policy_hash

    calibration = {
        "n_paths": n_paths,
        "mean_block_length": mean_block,
        "joint_false_kill_budget": false_kill_budget,
        "per_monitor_alpha": alpha_i,
        "drawdown_simultaneous": {
            "warn": dd_warn,
            "reduce": dd_reduce,
            "pause": dd_pause,
            "hard_cap": hard_cap,
        },
        "tuw_simultaneous_bars": {
            "warn": int(round(tuw_warn)),
            "reduce": int(round(tuw_reduce)),
            "pause": int(round(tuw_pause)),
        },
        "cusum_h": h_cusum,
        "horizon_snapshots_additive_dd": _snap(
            dd_paths, year_idx, (0.05, 0.01), worse="low"
        ),
        "horizon_snapshots_tuw": _snap(
            tuw_paths.astype(float), year_idx, (0.95, 0.99), worse="high"
        ),
    }

    # Pointwise bands (for plotting) — downsampled for report size
    step = max(1, horizon // 60)
    bands = {
        "elapsed_bars": list(range(0, horizon, step)),
        "maxdd_q05": np.quantile(dd_paths, 0.05, axis=0)[::step].tolist(),
        "maxdd_q01": np.quantile(dd_paths, 0.01, axis=0)[::step].tolist(),
        "tuw_q95": np.quantile(tuw_paths, 0.95, axis=0)[::step].tolist(),
        "tuw_q99": np.quantile(tuw_paths, 0.99, axis=0)[::step].tolist(),
    }

    return KillPolicyReport(
        policy=policy,
        policy_hash=policy_hash,
        calibration=calibration,
        bands=bands,
    )


def evaluate_live(
    live_returns: pd.Series,
    policy: dict[str, Any],
    *,
    tracking_z: Optional[float] = None,
    sign_agreement: Optional[float] = None,
    operational_flags: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Apply a calibrated policy to live returns; return graduated response."""
    r = pd.Series(live_returns, dtype=float).dropna()
    monitors = policy["monitors"]
    level = ResponseLevel.NORMAL
    reasons: list[str] = []

    # Operational first (fastest / most informative)
    ops = operational_flags or {}
    if ops.get("config_drift"):
        level = ResponseLevel.PAUSE_FLAT_KEEP_SHADOW
        reasons.append("operational:config_drift")
    if ops.get("data_staleness_bars", 0) >= monitors["operational"]["data_staleness"]["pause_bars"]:
        level = _max_level(level, ResponseLevel.PAUSE_FLAT_KEEP_SHADOW)
        reasons.append("operational:data_staleness")
    elif ops.get("data_staleness_bars", 0) >= monitors["operational"]["data_staleness"]["warn_bars"]:
        level = _max_level(level, ResponseLevel.WARN)
        reasons.append("operational:data_staleness_warn")

    # Drawdown (additive for consistency with calibration)
    dd = max_drawdown(r, additive=True)
    mdd = monitors["max_drawdown"]
    if dd is not None:
        if dd <= mdd["hard_cap"] or dd <= mdd["pause"]:
            level = _max_level(level, ResponseLevel.PAUSE_FLAT_KEEP_SHADOW)
            reasons.append(f"maxdd={dd:.4f} ≤ pause/hard_cap")
        elif dd <= mdd["reduce"]:
            level = _max_level(level, ResponseLevel.REDUCE_HALF)
            reasons.append(f"maxdd={dd:.4f} ≤ reduce")
        elif dd <= mdd["warn"]:
            level = _max_level(level, ResponseLevel.WARN)
            reasons.append(f"maxdd={dd:.4f} ≤ warn")

    # Time under water
    tuw = time_under_water(r)
    mt = monitors["time_under_water"]
    if tuw >= mt["pause"]:
        level = _max_level(level, ResponseLevel.PAUSE_FLAT_KEEP_SHADOW)
        reasons.append(f"tuw={tuw} ≥ pause")
    elif tuw >= mt["reduce"]:
        level = _max_level(level, ResponseLevel.REDUCE_HALF)
        reasons.append(f"tuw={tuw} ≥ reduce")
    elif tuw >= mt["warn"]:
        level = _max_level(level, ResponseLevel.WARN)
        reasons.append(f"tuw={tuw} ≥ warn")

    # CUSUM
    mc = monitors["cusum_mean_drop"]
    cusum = _running_cusum(r.values, mc["mu_healthy"], mc["mu_dead"])
    if cusum >= mc["h"]:
        level = _max_level(level, ResponseLevel.PAUSE_FLAT_KEEP_SHADOW)
        reasons.append(f"cusum={cusum:.4f} ≥ h")
    elif cusum >= mc["warn"]:
        level = _max_level(level, ResponseLevel.WARN)
        reasons.append(f"cusum={cusum:.4f} ≥ h/2")

    # Bayes-ish: PSR(0) inverted as P(negative skill) ≈ 1 - PSR
    psr = probabilistic_sharpe(r, sr_star_annual=0.0)
    if psr is not None:
        p_neg = 1.0 - psr
        mb = monitors["bayes_p_negative"]
        if p_neg >= mb["pause"]:
            level = _max_level(level, ResponseLevel.PAUSE_FLAT_KEEP_SHADOW)
            reasons.append(f"P(SR<0)≈{p_neg:.2f}")
        elif p_neg >= mb["reduce"]:
            level = _max_level(level, ResponseLevel.REDUCE_HALF)
            reasons.append(f"P(SR<0)≈{p_neg:.2f}")

    # Tracking
    if tracking_z is not None:
        tz = monitors["tracking_z"]
        if tracking_z <= tz["pause"]:
            level = _max_level(level, ResponseLevel.PAUSE_FLAT_KEEP_SHADOW)
            reasons.append(f"tracking_z={tracking_z:.2f}")
        elif tracking_z <= tz["warn"]:
            level = _max_level(level, ResponseLevel.WARN)
            reasons.append(f"tracking_z={tracking_z:.2f}")
    if sign_agreement is not None and sign_agreement < monitors["tracking_z"]["sign_agreement_min"]:
        level = _max_level(level, ResponseLevel.PAUSE_FLAT_KEEP_SHADOW)
        reasons.append(f"sign_agreement={sign_agreement:.2f}")

    return {
        "response": level.value,
        "reasons": reasons,
        "live_summary": {
            "n": int(len(r)),
            "sharpe": sharpe(r),
            "maxdd_additive": dd,
            "maxdd_compounding": max_drawdown(r, additive=False),
            "time_under_water": tuw,
            "cusum": cusum,
            "psr_0": psr,
            "equity_last": float(equity_curve(r).iloc[-1]) if len(r) else None,
        },
        "policy_hash": policy.get("policy_hash"),
    }


_LEVEL_ORDER = [
    ResponseLevel.NORMAL,
    ResponseLevel.WARN,
    ResponseLevel.REDUCE_HALF,
    ResponseLevel.PAUSE_FLAT_KEEP_SHADOW,
    ResponseLevel.RETIRE,
]


def _max_level(a: ResponseLevel, b: ResponseLevel) -> ResponseLevel:
    return a if _LEVEL_ORDER.index(a) >= _LEVEL_ORDER.index(b) else b


def _running_cusum(x: np.ndarray, mu_h: float, mu_d: float) -> float:
    k = 0.5 * (mu_h + mu_d)
    s = 0.0
    peak = 0.0
    for v in x:
        s = max(0.0, s + (k - v))
        peak = max(peak, s)
    return float(peak)
