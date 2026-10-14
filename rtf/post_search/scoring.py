"""Gates G0–G9, calibrated score proxy, and fail-closed tiers (A.13)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy import stats

from ..bundle import StrategyBundle
from ..metrics import (
    max_drawdown,
    n_trades_from_positions,
    periods_per_year,
    sharpe,
)
from ..types import GateResult, TestResult, Tier, Verdict


def evaluate_gates(
    bundle: StrategyBundle,
    tests: list[TestResult],
) -> list[GateResult]:
    by_id = {t.id: t for t in tests}
    is_sr = sharpe(bundle.is_returns)
    oos_sr = sharpe(bundle.oos_returns)
    full_sr = sharpe(bundle.full_dev_returns)
    oos_dd = max_drawdown(bundle.oos_returns)

    # G0 — run integrity: no blocking FAIL
    blocking_fail = [
        t for t in tests if t.check_type.value == "blocking" and t.verdict == Verdict.FAIL
    ]
    g0 = GateResult(
        "G0",
        "Run integrity: no blocking defect",
        passed=(len(blocking_fail) == 0),
        detail="ok" if not blocking_fail else f"failed: {[t.id for t in blocking_fail]}",
    )

    # G1 — ≥ 0.5 yr OOS before sealed
    q = periods_per_year(bundle.oos_returns.index) if len(bundle.oos_returns) else 252.0
    oos_years = len(bundle.oos_returns) / q if q else 0.0
    g1 = GateResult(
        "G1",
        "≥ 0.5 yr OOS before sealed boundary",
        passed=oos_years >= 0.5,
        detail=f"oos_years={oos_years:.2f}",
    )

    # G2 — trade counts
    if bundle.positions is not None:
        n_full = n_trades_from_positions(bundle.positions)
        n_oos = n_trades_from_positions(
            bundle.positions[
                (bundle.positions.index >= bundle.oos_start)
                & (bundle.positions.index < bundle.sealed_start)
            ]
        )
        g2 = GateResult(
            "G2",
            "≥100 trades, ≥30 OOS trades",
            passed=(n_full >= 100 and n_oos >= 30),
            detail=f"trades_full={n_full}, trades_oos={n_oos}",
        )
    else:
        g2 = GateResult("G2", "≥100 trades, ≥30 OOS trades", passed=None, detail="no positions")

    # G3 — OOS SR ≥ 0.5
    g3 = GateResult(
        "G3",
        "OOS SR ≥ 0.5",
        passed=(oos_sr is not None and oos_sr >= 0.5),
        detail=f"oos_sharpe={oos_sr}",
    )

    # G4 — IS ≥ 0.3; OOS ≥ 50% IS
    g4_ok = (
        is_sr is not None
        and oos_sr is not None
        and is_sr >= 0.3
        and oos_sr >= 0.5 * is_sr
    )
    g4 = GateResult(
        "G4",
        "IS ≥ 0.3 and OOS ≥ 50% of IS",
        passed=g4_ok,
        detail=f"is={is_sr}, oos={oos_sr}",
    )

    # G5 — cost 2×
    a31 = by_id.get("A3.1")
    if a31 is None or a31.verdict == Verdict.INCOMPLETE:
        g5 = GateResult("G5", "Cost 2× > 0 and ≥ 50% base", passed=None, detail="not assessed")
    else:
        g5 = GateResult(
            "G5",
            "Cost 2× > 0 and ≥ 50% base",
            passed=(a31.verdict == Verdict.PASS),
            detail=a31.detail,
        )

    # G6 — parameter stability
    a61 = by_id.get("A6.1")
    if a61 is None or a61.verdict == Verdict.INCOMPLETE:
        g6 = GateResult("G6", "Parameter plateau", passed=None, detail="not assessed")
    else:
        g6 = GateResult(
            "G6",
            "Parameter plateau",
            passed=(a61.verdict == Verdict.PASS),
            detail=a61.detail,
        )

    # G7 — OOS MaxDD ≥ −25%
    g7 = GateResult(
        "G7",
        "OOS MaxDD ≥ −25%",
        passed=(oos_dd is not None and oos_dd >= -0.25),
        detail=f"oos_maxdd={oos_dd}",
    )

    # G8 — DSR ≥ 0.95
    a43 = by_id.get("A4.3")
    if a43 is None or a43.verdict == Verdict.INCOMPLETE:
        g8 = GateResult("G8", "DSR ≥ 0.95 at honest N", passed=None, detail="not assessed")
    else:
        g8 = GateResult(
            "G8",
            "DSR ≥ 0.95 at honest N",
            passed=(a43.verdict == Verdict.PASS),
            detail=a43.detail,
        )

    # G9 — null p < 0.05
    nulls = [by_id.get("A4.1"), by_id.get("A4.2")]
    if any(t is None or t.verdict == Verdict.INCOMPLETE for t in nulls):
        g9 = GateResult(
            "G9",
            "Shift/permutation null p < 0.05",
            passed=None,
            detail="nulls not stored / incomplete",
        )
    else:
        ok = all(t.verdict == Verdict.PASS for t in nulls if t is not None)
        g9 = GateResult(
            "G9",
            "Shift/permutation null p < 0.05",
            passed=ok,
            detail="; ".join(t.detail for t in nulls if t),
        )

    # silence unused
    _ = full_sr
    return [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9]


def assign_tier(gates: list[GateResult], tests: list[TestResult]) -> Tier:
    """Fail-closed tier rules from A.13."""
    # Not credible: sealed leak or look-ahead blocking fails
    for t in tests:
        if t.id in {"A1.3", "A2.1"} and t.verdict == Verdict.FAIL:
            return Tier.NOT_CREDIBLE
        if t.id == "A2.3" and t.verdict == Verdict.FAIL:
            return Tier.NOT_CREDIBLE

    by = {g.gate: g for g in gates}
    # Any assessed gate that failed (except pure missing) → Failed
    # Incomplete: only G1 fails OR required artefacts absent (any passed is None among mandatory research)
    failed = [g for g in gates if g.passed is False]
    incomplete_assess = [g for g in gates if g.passed is None]

    if any(g.gate == "G0" and g.passed is False for g in gates):
        return Tier.NOT_CREDIBLE if by["G0"].detail.startswith("failed: ['A1.3'") else Tier.FAILED

    # If the only failure is G1 (short OOS) and nothing else failed → Incomplete
    if failed and all(g.gate == "G1" for g in failed) and incomplete_assess:
        return Tier.INCOMPLETE
    if failed and all(g.gate == "G1" for g in failed):
        return Tier.INCOMPLETE

    if failed:
        return Tier.FAILED

    if incomplete_assess:
        # No gate failed; at least one not assessed → Provisional
        return Tier.PROVISIONAL

    return Tier.CLEARED


def robustness_score(bundle: StrategyBundle, n_trials: Optional[int] = None) -> dict[str, Any]:
    """Score = 10 × P(OOS holds | IS evidence) via Bayesian shrinkage (A13.1).

    Event: one-year OOS Sharpe > 0 and ≥ ½ IS Sharpe.
    Prior θ ~ N(0, 0.8²); haircut λ unused (MAP finding λ=0).
    Always labelled as weakly-calibrated on the programme data.
    """
    is_r = bundle.is_returns.dropna()
    if len(is_r) < 30:
        return {
            "score": None,
            "probability": None,
            "status": "INCOMPLETE",
            "note": "Insufficient IS returns",
            "calibration": "not evaluated",
        }
    is_sr = sharpe(is_r)
    if is_sr is None or is_sr <= 0:
        return {
            "score": 0.0,
            "probability": 0.0,
            "status": "No IS edge",
            "note": "IS SR ≤ 0",
            "calibration": "n/a",
        }

    q = periods_per_year(is_r.index)
    # Lo SE adjusted roughly via PSR denominator
    se = _lo_se(is_r, is_sr, q)
    # Prior
    mu0, tau = 0.0, 0.8
    post_var = 1.0 / (1.0 / tau**2 + 1.0 / se**2)
    post_mean = post_var * (mu0 / tau**2 + is_sr / se**2)
    # One-year predictive: add horizon noise (1 + m²/2)/h with h=1, m≈post_mean/vol_unit
    # Approximate annual SR predictive sd
    h = 1.0
    pred_sd = np.sqrt(post_var + (1.0 + 0.5 * post_mean**2) / h)
    # P(SR_oos > 0 and SR_oos >= 0.5 * SR_is)
    thresh = max(0.0, 0.5 * is_sr)
    # Both constraints → SR_oos >= thresh (since thresh >= 0 when is_sr>0)
    z = (thresh - post_mean) / pred_sd
    # P(SR > thresh) = 1 - Φ(z) but we need upper tail from thresh
    # Actually P(SR_oos >= thresh) = 1 - Φ((thresh - mu)/sd)
    p = float(1.0 - stats.norm.cdf(z))
    # Optional trial-count haircut unused (λ=0)
    _ = n_trials
    return {
        "score": round(10.0 * p, 2),
        "probability": round(p, 4),
        "posterior_mean_sr": round(float(post_mean), 3),
        "posterior_sd_sr": round(float(np.sqrt(post_var)), 3),
        "predictive_sd_1y": round(float(pred_sd), 3),
        "is_sharpe": round(float(is_sr), 3),
        "status": "ok",
        "calibration": (
            "Weakly calibrated on programme data (Brier≈base rate, AUC≈0.55). "
            "Present as a ranked probability, not a profit forecast."
        ),
    }


def _lo_se(r, sr_annual: float, q: float) -> float:
    """Rough Lo (2002) standard error for annualised Sharpe."""
    T = len(r)
    if T < 3:
        return 1.0
    # SE(SR_annual) ≈ sqrt((1 + 0.5 SR^2) * q / T) under iid
    return float(np.sqrt(max(1e-8, (1.0 + 0.5 * sr_annual**2) * q / T)))
