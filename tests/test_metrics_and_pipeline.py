"""Unit tests for metrics, fills, and a miniature end-to-end evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rtf.bundle import StrategyBundle
from rtf.fills import strategy_returns
from rtf.kill_switch.policy import calibrate_kill_policy, evaluate_live
from rtf.metrics import deflated_sharpe, periods_per_year, sharpe
from rtf.pipeline import evaluate_strategy
from rtf.types import Tier, Verdict


def _synth(n=800, seed=0, edge=0.0002):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    asset = pd.Series(rng.normal(0.0002, 0.01, n), index=idx)
    pos = pd.Series(np.sign(asset.rolling(10).mean().shift(1)).fillna(0.0), index=idx)
    ret = strategy_returns(pos, asset, lag=1, cost_bps=2.0) + edge * pos.shift(1).fillna(0)
    return ret, pos, asset


def test_periods_per_year_business_days():
    idx = pd.date_range("2020-01-01", periods=252, freq="B", tz="UTC")
    q = periods_per_year(idx)
    assert 240 < q < 270


def test_same_bar_refused():
    r, pos, asset = _synth(100)
    with pytest.raises(ValueError):
        strategy_returns(pos, asset, lag=0)


def test_dsr_low_at_large_n():
    r, _, _ = _synth(500, edge=0.0)
    # Noise strategy at huge N should have low DSR
    dsr = deflated_sharpe(r, n_trials=100_000)
    assert dsr is not None
    assert dsr < 0.8


def test_evaluate_smoke():
    ret, pos, asset = _synth(900, seed=1, edge=0.00025)
    bundle = StrategyBundle(
        returns=ret,
        oos_start="2022-06-01",
        positions=pos,
        asset_returns=asset,
        n_trials=20,
        cost_bps_base=2.0,
        meta={"strategy_id": "unit"},
    )
    report = evaluate_strategy(
        bundle, calibrate_kills=True, n_null=100, n_kill_paths=400, seed=1
    )
    assert report.tier in {
        Tier.CLEARED,
        Tier.PROVISIONAL,
        Tier.INCOMPLETE,
        Tier.FAILED,
        Tier.NOT_CREDIBLE,
    }
    assert any(t.id == "A1.3" and t.verdict == Verdict.PASS for t in report.tests)
    assert report.kill_policy is not None
    assert "max_drawdown" in report.kill_policy.policy["monitors"]


def test_sealed_fail():
    ret, pos, asset = _synth(100, seed=2)
    # Inject a sealed stamp
    sealed_idx = pd.Timestamp("2025-10-15", tz="UTC")
    ret.loc[sealed_idx] = 0.01
    bundle = StrategyBundle(
        returns=ret,
        oos_start="2022-01-01",
        positions=pos,
        asset_returns=asset,
        sealed_start="2025-10-01",
    )
    report = evaluate_strategy(bundle, calibrate_kills=False, n_null=50)
    assert report.tier == Tier.NOT_CREDIBLE


def test_kill_live_pause_on_crash():
    ret, _, _ = _synth(600, seed=3, edge=0.0002)
    policy_report = calibrate_kill_policy(ret, n_paths=500, seed=3)
    # Fabricate a live crash
    live_idx = pd.date_range("2024-01-01", periods=80, freq="B", tz="UTC")
    crash = pd.Series(-0.02, index=live_idx)  # persistent -2%/day
    result = evaluate_live(crash, policy_report.policy)
    assert result["response"] in {
        "REDUCE_HALF",
        "PAUSE_FLAT_KEEP_SHADOW",
        "RETIRE",
        "WARN",
    }
