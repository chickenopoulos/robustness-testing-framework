"""Next-bar fills and delay sweep (A2.1 / A2.2)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .metrics import sharpe


def strategy_returns(
    positions: pd.Series,
    asset_returns: pd.Series,
    lag: int = 1,
    cost_bps: float = 0.0,
    allow_same_bar: bool = False,
) -> pd.Series:
    """P&L from positions decided at bar close.

    r_t = pos_{t-L} * asset_r_t - cost * |Δpos|

    lag < 1 is refused unless allow_same_bar=True (recorded by caller).
    """
    if lag < 1 and not allow_same_bar:
        raise ValueError(
            "lag < 1 is same-bar look-ahead; pass allow_same_bar=True only as a diagnostic"
        )
    pos = pd.Series(positions, dtype=float).fillna(0.0)
    ret = pd.Series(asset_returns, dtype=float).reindex(pos.index).fillna(0.0)
    held = pos.shift(lag)
    turnover = held.diff().abs().fillna(held.abs())
    cost = (cost_bps / 1e4) * turnover
    out = held * ret - cost
    out.name = "strategy_return"
    return out


def delay_sweep(
    positions: pd.Series,
    asset_returns: pd.Series,
    lags: tuple[int, ...] = (0, 1, 2, 3),
    cost_bps: float = 0.0,
) -> dict:
    """Sharpe at L=0..3; D2/D3 decay ratios vs L=1 baseline."""
    results: dict[int, Optional[float]] = {}
    for L in lags:
        sr = strategy_returns(
            positions, asset_returns, lag=L, cost_bps=cost_bps, allow_same_bar=(L == 0)
        )
        results[L] = sharpe(sr.dropna())

    base = results.get(1)
    d2 = d3 = look_ahead_ratio = None
    if base is not None and abs(base) > 1e-12:
        if results.get(2) is not None:
            d2 = results[2] / base
        if results.get(3) is not None:
            d3 = results[3] / base
        if results.get(0) is not None:
            look_ahead_ratio = results[0] / base

    sign_flip = False
    if base is not None:
        for L in (2, 3):
            v = results.get(L)
            if v is not None and np.sign(v) != np.sign(base) and abs(v) > 1e-9:
                sign_flip = True

    return {
        "sharpe_by_lag": {str(k): v for k, v in results.items()},
        "D2": d2,
        "D3": d3,
        "look_ahead_ratio_L0_over_L1": look_ahead_ratio,
        "sign_flip": sign_flip,
    }
