"""Canonical performance metrics.

Annualisation uses observed observation density (periods per year), never a
hard-coded constant — in the programme audits that single mistake drove most
ledger mismatches.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def periods_per_year(index: pd.DatetimeIndex | pd.Index) -> float:
    idx = pd.DatetimeIndex(index)
    if len(idx) < 3:
        return 252.0
    span_years = (idx[-1] - idx[0]).total_seconds() / (365.25 * 86400.0)
    if span_years <= 0:
        return 252.0
    return float((len(idx) - 1) / span_years)


def _clean(r: pd.Series) -> pd.Series:
    return pd.Series(r, dtype=float).dropna()


def per_period_sharpe(r: pd.Series) -> Optional[float]:
    r = _clean(r)
    if len(r) < 2:
        return None
    s = float(r.std(ddof=1))
    if s == 0 or not np.isfinite(s):
        return None
    return float(r.mean() / s)


def sharpe(r: pd.Series, q: Optional[float] = None) -> Optional[float]:
    r = _clean(r)
    pp = per_period_sharpe(r)
    if pp is None:
        return None
    if q is None:
        q = periods_per_year(r.index)
    return float(pp * np.sqrt(q))


def cagr(r: pd.Series, q: Optional[float] = None) -> Optional[float]:
    r = _clean(r)
    if len(r) < 2:
        return None
    if q is None:
        q = periods_per_year(r.index)
    years = len(r) / q
    growth = float((1.0 + r).prod())
    if growth <= 0 or years <= 0:
        return None
    return growth ** (1.0 / years) - 1.0


def max_drawdown(r: pd.Series, additive: bool = False) -> Optional[float]:
    """Compounding MaxDD by default; additive = sum of returns peak-to-trough."""
    r = _clean(r)
    if len(r) == 0:
        return None
    if additive:
        eq = r.cumsum()
        return float((eq - eq.cummax()).min())
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def equity_curve(r: pd.Series) -> pd.Series:
    r = _clean(r)
    return (1.0 + r).cumprod()


def time_under_water(r: pd.Series) -> int:
    """Current underwater spell length (bars since last equity high)."""
    r = _clean(r)
    if len(r) == 0:
        return 0
    eq = equity_curve(r)
    peak_idx = int(eq.values.argmax())
    return int(len(eq) - 1 - peak_idx)


def longest_underwater(r: pd.Series) -> int:
    r = _clean(r)
    if len(r) == 0:
        return 0
    eq = equity_curve(r)
    peak = eq.cummax()
    underwater = eq < peak
    longest = cur = 0
    for flag in underwater.values:
        if flag:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return int(longest)


def skewness(r: pd.Series) -> float:
    r = _clean(r)
    if len(r) < 3:
        return 0.0
    return float(stats.skew(r, bias=False))


def kurtosis_non_excess(r: pd.Series) -> float:
    """Non-excess kurtosis γ4 (normal = 3)."""
    r = _clean(r)
    if len(r) < 4:
        return 3.0
    return float(stats.kurtosis(r, fisher=False, bias=False))


def n_trades_from_positions(pos: pd.Series) -> int:
    p = pd.Series(pos).fillna(0.0)
    return int((p.diff().fillna(p).abs() > 1e-12).sum())


def summarize(r: pd.Series) -> dict:
    r = _clean(r)
    q = periods_per_year(r.index) if len(r) else 252.0
    return {
        "n": int(len(r)),
        "periods_per_year": q,
        "sharpe": sharpe(r, q),
        "cagr": cagr(r, q),
        "maxdd": max_drawdown(r),
        "mean": float(r.mean()) if len(r) else None,
        "vol": float(r.std(ddof=1) * np.sqrt(q)) if len(r) > 1 else None,
        "skew": skewness(r),
        "kurtosis": kurtosis_non_excess(r),
    }


# ---------------------------------------------------------------------------
# Probabilistic / deflated Sharpe (Bailey & López de Prado)
# ---------------------------------------------------------------------------

def _psr_z(sr_hat: float, sr_star: float, T: int, skew: float, kurt: float) -> float:
    """z-stat for PSR; all Sharpe values are *per-period*."""
    if T < 2:
        return float("nan")
    num = (sr_hat - sr_star) * np.sqrt(T - 1)
    den = np.sqrt(max(1e-12, 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat**2))
    return float(num / den)


def probabilistic_sharpe(
    r: pd.Series,
    sr_star_annual: float = 0.0,
    q: Optional[float] = None,
) -> Optional[float]:
    """PSR = Φ(z): Prob(true SR > sr_star | observed sample)."""
    r = _clean(r)
    if len(r) < 3:
        return None
    if q is None:
        q = periods_per_year(r.index)
    sr_hat = per_period_sharpe(r)
    if sr_hat is None:
        return None
    sr_star = sr_star_annual / np.sqrt(q)
    z = _psr_z(sr_hat, sr_star, len(r), skewness(r), kurtosis_non_excess(r))
    if not np.isfinite(z):
        return None
    return float(stats.norm.cdf(z))


def min_track_record_length(
    r: pd.Series,
    sr_star_annual: float = 0.0,
    alpha: float = 0.05,
    q: Optional[float] = None,
) -> Optional[float]:
    """MinTRL in number of observations (Bailey & López de Prado)."""
    r = _clean(r)
    if len(r) < 3:
        return None
    if q is None:
        q = periods_per_year(r.index)
    sr_hat = per_period_sharpe(r)
    if sr_hat is None:
        return None
    sr_star = sr_star_annual / np.sqrt(q)
    if abs(sr_hat - sr_star) < 1e-12:
        return float("inf")
    g3 = skewness(r)
    g4 = kurtosis_non_excess(r)
    z_a = float(stats.norm.ppf(1.0 - alpha))
    factor = 1.0 - g3 * sr_hat + ((g4 - 1.0) / 4.0) * sr_hat**2
    return float(1.0 + factor * (z_a / (sr_hat - sr_star)) ** 2)


def deflated_sharpe(
    r: pd.Series,
    n_trials: int,
    q: Optional[float] = None,
    variance_of_sr: Optional[float] = None,
) -> Optional[float]:
    """DSR at honest trial count N (Bailey & López de Prado).

    SR0 benchmark uses the expected maximum of N zero-skill Sharpes.
    Returns DSR = PSR(SR0) in [0, 1].
    """
    r = _clean(r)
    if len(r) < 3 or n_trials < 1:
        return None
    if q is None:
        q = periods_per_year(r.index)
    sr_hat = per_period_sharpe(r)
    if sr_hat is None:
        return None
    # Expected max of N independent N(0, V) draws; V ≈ 1/(T-1) under iid.
    T = len(r)
    V = variance_of_sr if variance_of_sr is not None else 1.0 / max(T - 1, 1)
    # Euler-Mascheroni approximation for E[max of N standard normals]
    euler = 0.5772156649
    z_n = float(stats.norm.ppf(1.0 - 1.0 / n_trials)) if n_trials > 1 else 0.0
    z_n_e = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))) if n_trials > 1 else 0.0
    # SR0 (per-period) = √V * [(1-γ) z_{1-1/N} + γ z_{1-1/(N e)}] with γ≈euler
    sr0 = np.sqrt(V) * ((1.0 - euler) * z_n + euler * z_n_e)
    z = _psr_z(sr_hat, float(sr0), T, skewness(r), kurtosis_non_excess(r))
    if not np.isfinite(z):
        return None
    return float(stats.norm.cdf(z))
