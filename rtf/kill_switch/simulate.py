"""Reference-path simulation for kill thresholds (C.5).

Uses a stationary bootstrap with a simple automatic mean block length
(Politis–White style autocorrelation rule) applied to haircut backtest returns.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def haircut_returns(r: pd.Series, haircut: float = 0.5) -> pd.Series:
    """Scale the mean toward zero: r' = r - mean + mean * haircut."""
    x = pd.Series(r, dtype=float).dropna()
    mu = float(x.mean())
    return x - mu + mu * haircut


def politis_white_mean_block_length(r: np.ndarray) -> int:
    """Heuristic mean block length from lag-1 autocorrelation (floor at 1).

    Full Politis–White requires spectral density estimates; this practical
    surrogate tracks short-range dependence well enough for kill calibration.
    """
    x = np.asarray(r, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 20:
        return 5
    c0 = float(np.dot(x, x) / n)
    c1 = float(np.dot(x[:-1], x[1:]) / n)
    rho = c1 / c0 if c0 > 0 else 0.0
    rho = float(np.clip(rho, 0.0, 0.99))
    # Expected geometric block length rises with persistence
    b = int(np.ceil((1.0 + rho) / (1.0 - rho)))
    # Cap: don't let blocks eat the whole sample
    return int(np.clip(max(b, 1), 1, max(1, n // 10)))


def stationary_bootstrap_paths(
    returns: np.ndarray,
    horizon: int,
    n_paths: int,
    mean_block: int,
    seed: int = 42,
) -> np.ndarray:
    """Return array shape (n_paths, horizon) of bootstrapped returns."""
    rng = np.random.default_rng(seed)
    x = np.asarray(returns, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("empty returns")
    p = 1.0 / max(mean_block, 1)
    out = np.empty((n_paths, horizon), dtype=float)
    for i in range(n_paths):
        idx = rng.integers(0, n)
        for t in range(horizon):
            out[i, t] = x[idx]
            if rng.random() < p:
                idx = rng.integers(0, n)
            else:
                idx = (idx + 1) % n
    return out


def path_maxdd_to_date(paths: np.ndarray, additive: bool = True) -> np.ndarray:
    """Max drawdown-to-date for each path and elapsed day. Shape (n_paths, T)."""
    if additive:
        eq = np.cumsum(paths, axis=1)
    else:
        eq = np.cumprod(1.0 + paths, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = eq - peak  # ≤ 0
    # maxdd-to-date = most negative dd so far
    return np.minimum.accumulate(dd, axis=1)


def path_tuw_to_date(paths: np.ndarray, additive: bool = True) -> np.ndarray:
    """Current time-under-water (bars) for each path/elapsed day."""
    if additive:
        eq = np.cumsum(paths, axis=1)
    else:
        eq = np.cumprod(1.0 + paths, axis=1)
    n_paths, T = eq.shape
    out = np.zeros((n_paths, T), dtype=int)
    for i in range(n_paths):
        peak_val = eq[i, 0]
        peak_t = 0
        for t in range(T):
            if eq[i, t] >= peak_val - 1e-15:
                peak_val = eq[i, t]
                peak_t = t
                out[i, t] = 0
            else:
                out[i, t] = t - peak_t
    return out


def percentile_bands(
    stat_paths: np.ndarray,
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> dict[str, np.ndarray]:
    """Pointwise percentile bands over paths. stat_paths: (n_paths, T)."""
    # For drawdowns (negative), "worse" = more negative → use lower percentiles.
    # Caller decides sign convention. We compute raw quantiles of the statistic.
    out = {}
    for q in quantiles:
        out[f"q{int(q * 100)}"] = np.quantile(stat_paths, q, axis=0)
    return out


def simultaneous_threshold(
    stat_paths: np.ndarray,
    target_false_kill: float,
    worse: str = "low",
) -> float:
    """Path-wise calibrated threshold so P(ever breach within horizon) ≈ α.

    For MaxDD (more negative is worse): find level L such that the fraction of
    paths whose min(stat) < L equals α.
    For TuW (larger is worse): fraction whose max(stat) > L equals α.
    """
    if worse == "low":
        extremes = stat_paths.min(axis=1)
        # We want P(extreme < L) = α → L = quantile(extremes, α)
        return float(np.quantile(extremes, target_false_kill))
    extremes = stat_paths.max(axis=1)
    return float(np.quantile(extremes, 1.0 - target_false_kill))


def cusum_threshold_calibration(
    paths: np.ndarray,
    mu_healthy: float,
    mu_dead: float,
    target_false_kill: float,
) -> float:
    """Calibrate two-sided CUSUM threshold h so P(ever fire | healthy) ≈ α.

    Uses the standard Page CUSUM on demeaned returns toward detecting a drop
    from mu_healthy to mu_dead (per-period means).
    """
    # Likelihood-ratio increment for detecting mean drop
    # k = (mu_h + mu_d) / 2; fire when cumsum of (k - r) exceeds h
    k = 0.5 * (mu_healthy + mu_dead)
    n_paths, T = paths.shape
    max_cusum = np.zeros(n_paths)
    for i in range(n_paths):
        s = 0.0
        peak = 0.0
        for t in range(T):
            s = max(0.0, s + (k - paths[i, t]))
            peak = max(peak, s)
        max_cusum[i] = peak
    # h such that P(max_cusum > h) = α
    return float(np.quantile(max_cusum, 1.0 - target_false_kill))
