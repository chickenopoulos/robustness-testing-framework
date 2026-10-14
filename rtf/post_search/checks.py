"""Part A test cards implemented as pure functions over a StrategyBundle."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..bundle import StrategyBundle
from ..fills import delay_sweep, strategy_returns
from ..metrics import (
    cagr,
    deflated_sharpe,
    min_track_record_length,
    probabilistic_sharpe,
    sharpe,
    summarize,
)
from ..types import CheckType, TestResult, Verdict


def run_post_search_battery(
    bundle: StrategyBundle,
    *,
    n_null: int = 500,
    seed: int = 42,
) -> list[TestResult]:
    """Run all evaluable Part A checks; missing inputs → INCOMPLETE (fail-closed)."""
    results: list[TestResult] = []
    results.append(_a13_sealed(bundle))
    results.append(_a21_next_bar(bundle))
    results.append(_a22_delay_sweep(bundle))
    results.append(_a23_reconcile(bundle))
    results.append(_a25_equity_sanity(bundle))
    results.append(_a31_cost_sweep(bundle))
    results.extend(_a4_nulls(bundle, n_null=n_null, seed=seed))
    results.append(_a43_dsr(bundle))
    results.append(_a44_psr_mintrl(bundle))
    results.append(_a46_pbo_proxy(bundle))
    results.append(_a5_oos_hold(bundle))
    results.append(_a61_param_neighbourhood(bundle))
    results.append(_a91_concentration(bundle))
    results.append(_a101_regimes(bundle))
    results.append(_a111_attribution(bundle))
    results.append(_a121_book_corr(bundle))
    return results


# ---------------------------------------------------------------------------
# A1.3 Sealed-window guard
# ---------------------------------------------------------------------------

def _a13_sealed(b: StrategyBundle) -> TestResult:
    sealed = b.returns.index[b.returns.index >= b.sealed_start]
    if len(sealed) == 0:
        return TestResult(
            "A1.3",
            "Sealed-window guard",
            Verdict.PASS,
            CheckType.BLOCKING,
            detail=f"No rows on/after {b.sealed_start.date()}",
        )
    return TestResult(
        "A1.3",
        "Sealed-window guard",
        Verdict.FAIL,
        CheckType.BLOCKING,
        metric=float(len(sealed)),
        detail=f"{len(sealed)} sealed rows present in returns — Not credible for OOS claim",
        extras={"first_sealed": str(sealed.min())},
    )


# ---------------------------------------------------------------------------
# A2.1 Next-bar fills
# ---------------------------------------------------------------------------

def _a21_next_bar(b: StrategyBundle) -> TestResult:
    if b.allow_same_bar:
        return TestResult(
            "A2.1",
            "Next-bar fills",
            Verdict.FAIL,
            CheckType.BLOCKING,
            detail="allow_same_bar=True — same-bar fills in reported numbers",
        )
    if b.positions is None or b.asset_returns is None:
        return TestResult(
            "A2.1",
            "Next-bar fills",
            Verdict.INCOMPLETE,
            CheckType.BLOCKING,
            detail="Need positions + asset_returns to verify lag≥1 construction",
        )
    # Recompute lag-1 net returns (zero incremental cost) and correlate with stored.
    recon = strategy_returns(b.positions, b.asset_returns, lag=1, cost_bps=0.0)
    aligned = pd.concat([recon, b.returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return TestResult(
            "A2.1",
            "Next-bar fills",
            Verdict.INCOMPLETE,
            CheckType.BLOCKING,
            detail="Insufficient overlap to verify fill lag",
        )
    rho = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    # High correlation with lag-1 reconstruction is expected if costs are small.
    # Also check lag-0 would fit *better* — look-ahead signature.
    recon0 = strategy_returns(
        b.positions, b.asset_returns, lag=0, cost_bps=0.0, allow_same_bar=True
    )
    a0 = pd.concat([recon0, b.returns], axis=1, join="inner").dropna()
    rho0 = float(a0.iloc[:, 0].corr(a0.iloc[:, 1])) if len(a0) > 30 else None
    if rho0 is not None and rho0 > rho + 0.05 and rho0 > 0.95:
        return TestResult(
            "A2.1",
            "Next-bar fills",
            Verdict.FAIL,
            CheckType.BLOCKING,
            metric=rho0,
            detail=f"Stored returns match lag-0 (ρ={rho0:.3f}) better than lag-1 (ρ={rho:.3f})",
            extras={"rho_lag1": rho, "rho_lag0": rho0},
        )
    if rho >= 0.90:
        return TestResult(
            "A2.1",
            "Next-bar fills",
            Verdict.PASS,
            CheckType.BLOCKING,
            metric=rho,
            detail=f"Stored returns consistent with lag≥1 fills (ρ={rho:.3f})",
            extras={"rho_lag1": rho, "rho_lag0": rho0},
        )
    if rho >= 0.70:
        return TestResult(
            "A2.1",
            "Next-bar fills",
            Verdict.BORDERLINE,
            CheckType.BLOCKING,
            metric=rho,
            detail=f"Moderate match to lag-1 (ρ={rho:.3f}); costs or transforms may differ",
            extras={"rho_lag1": rho, "rho_lag0": rho0},
        )
    return TestResult(
        "A2.1",
        "Next-bar fills",
        Verdict.BORDERLINE,
        CheckType.BLOCKING,
        metric=rho,
        detail=f"Low correlation with lag-1 rebuild (ρ={rho:.3f}); disclose fill convention",
        extras={"rho_lag1": rho, "rho_lag0": rho0},
    )


# ---------------------------------------------------------------------------
# A2.2 Delay sweep
# ---------------------------------------------------------------------------

def _a22_delay_sweep(b: StrategyBundle) -> TestResult:
    if b.positions is None or b.asset_returns is None:
        return TestResult(
            "A2.2",
            "Execution-delay sweep 0–3",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Need positions + asset_returns",
        )
    sweep = delay_sweep(b.positions, b.asset_returns, cost_bps=b.cost_bps_base)
    d2, d3 = sweep["D2"], sweep["D3"]
    la = sweep["look_ahead_ratio_L0_over_L1"]
    if d2 is None:
        return TestResult(
            "A2.2",
            "Execution-delay sweep 0–3",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Could not compute D2 (baseline Sharpe missing/zero)",
            extras=sweep,
        )
    if (la is not None and la > 3) or sweep["sign_flip"] or d2 < 0.3:
        return TestResult(
            "A2.2",
            "Execution-delay sweep 0–3",
            Verdict.FAIL,
            CheckType.RESEARCH,
            metric=d2,
            threshold_pass=0.5,
            threshold_fail=0.3,
            detail="Look-ahead signature, sign flip, or D2 < 0.3",
            extras=sweep,
        )
    if d2 >= 0.5 and (d3 is None or d3 >= 0.3):
        return TestResult(
            "A2.2",
            "Execution-delay sweep 0–3",
            Verdict.PASS,
            CheckType.RESEARCH,
            metric=d2,
            threshold_pass=0.5,
            threshold_fail=0.3,
            detail=f"D2={d2:.2f}, D3={None if d3 is None else round(d3, 2)}",
            extras=sweep,
        )
    return TestResult(
        "A2.2",
        "Execution-delay sweep 0–3",
        Verdict.BORDERLINE,
        CheckType.RESEARCH,
        metric=d2,
        threshold_pass=0.5,
        threshold_fail=0.3,
        detail=f"Fragile delay profile D2={d2:.2f}",
        extras=sweep,
    )


# ---------------------------------------------------------------------------
# A2.3 Ledger reconciliation
# ---------------------------------------------------------------------------

def _a23_reconcile(b: StrategyBundle) -> TestResult:
    if not b.claimed_metrics:
        return TestResult(
            "A2.3",
            "Ledger reconciliation",
            Verdict.INCOMPLETE,
            CheckType.BLOCKING,
            detail="No claimed_metrics supplied — nothing to reconcile",
        )
    recomputed = summarize(b.full_dev_returns)
    mismatches = []
    for key, tol_abs, tol_rel in (
        ("sharpe", 0.05, 0.05),
        ("cagr", 0.005, 0.05),
        ("maxdd", 0.005, 0.0),
    ):
        claimed = b.claimed_metrics.get(key)
        got = recomputed.get(key)
        if claimed is None or got is None:
            continue
        lim = max(tol_abs, abs(claimed) * tol_rel) if tol_rel else tol_abs
        if abs(float(claimed) - float(got)) > lim:
            mismatches.append(
                f"{key}: claimed={claimed} recomputed={got:.6f} (tol={lim})"
            )
    if mismatches:
        return TestResult(
            "A2.3",
            "Ledger reconciliation",
            Verdict.FAIL,
            CheckType.BLOCKING,
            detail="; ".join(mismatches),
            extras={"recomputed": recomputed, "claimed": b.claimed_metrics},
        )
    return TestResult(
        "A2.3",
        "Ledger reconciliation",
        Verdict.PASS,
        CheckType.BLOCKING,
        detail="All claimed metrics within tolerance",
        extras={"recomputed": recomputed},
    )


# ---------------------------------------------------------------------------
# A2.5 Equity sanity
# ---------------------------------------------------------------------------

def _a25_equity_sanity(b: StrategyBundle) -> TestResult:
    r = b.full_dev_returns.dropna()
    flags = []
    if (r.abs() > 0.5).any():
        flags.append("single_period_return>50%")
    sr = sharpe(r)
    if sr is not None and sr > 3.0:
        flags.append(f"sharpe>{sr:.2f}>3")
    cg = cagr(r)
    if cg is not None and cg > 1.0:
        flags.append("cagr>100%")
    zero_frac = float((r.abs() < 1e-12).mean()) if len(r) else 1.0
    if zero_frac > 0.97:
        flags.append("nearly_flat")
    if not flags:
        return TestResult(
            "A2.5",
            "Equity sanity",
            Verdict.PASS,
            CheckType.BLOCKING,
            detail="No sanity flags",
        )
    # Advisory-style but listed blocking for extreme flatness
    verdict = Verdict.FAIL if "nearly_flat" in flags else Verdict.BORDERLINE
    return TestResult(
        "A2.5",
        "Equity sanity",
        verdict,
        CheckType.BLOCKING,
        detail=", ".join(flags),
        extras={"flags": flags},
    )


# ---------------------------------------------------------------------------
# A3.1 Cost 1×/2×/3×
# ---------------------------------------------------------------------------

def _a31_cost_sweep(b: StrategyBundle) -> TestResult:
    if b.positions is None or b.asset_returns is None:
        return TestResult(
            "A3.1",
            "Cost 1×/2×/3×",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Need positions + asset_returns for cost sweep",
        )
    base = max(b.cost_bps_base, 1.0)  # at least 1 bp so multipliers are meaningful
    levels = {
        "1x": strategy_returns(b.positions, b.asset_returns, lag=1, cost_bps=base),
        "2x": strategy_returns(b.positions, b.asset_returns, lag=1, cost_bps=2 * base),
        "3x": strategy_returns(b.positions, b.asset_returns, lag=1, cost_bps=3 * base),
    }
    srs = {k: sharpe(v.dropna()) for k, v in levels.items()}
    s1, s2 = srs["1x"], srs["2x"]
    extras = {"cost_bps_base": base, "sharpe_by_cost": srs}
    if s1 is None or s2 is None:
        return TestResult(
            "A3.1",
            "Cost 1×/2×/3×",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Could not compute cost-sweep Sharpes",
            extras=extras,
        )
    if s2 <= 0:
        return TestResult(
            "A3.1",
            "Cost 1×/2×/3×",
            Verdict.FAIL,
            CheckType.RESEARCH,
            metric=s2,
            threshold_pass=0.0,
            detail=f"SR(2×)={s2:.3f} ≤ 0 (G5 fail)",
            extras=extras,
        )
    if s1 > 0 and s2 >= 0.5 * s1:
        return TestResult(
            "A3.1",
            "Cost 1×/2×/3×",
            Verdict.PASS,
            CheckType.RESEARCH,
            metric=s2 / s1 if s1 else None,
            detail=f"SR(2×)={s2:.3f} ≥ 50% of SR(1×)={s1:.3f}",
            extras=extras,
        )
    return TestResult(
        "A3.1",
        "Cost 1×/2×/3×",
        Verdict.BORDERLINE,
        CheckType.RESEARCH,
        metric=s2 / s1 if s1 else None,
        detail=f"SR(2×)={s2:.3f} positive but < 50% of base",
        extras=extras,
    )


# ---------------------------------------------------------------------------
# A4.1–2 Shift / permutation nulls
# ---------------------------------------------------------------------------

def _a4_nulls(b: StrategyBundle, n_null: int, seed: int) -> list[TestResult]:
    if b.positions is None or b.asset_returns is None:
        incomplete = TestResult(
            "A4.1",
            "Shift null",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Need positions + asset_returns",
        )
        incomplete2 = TestResult(
            "A4.2",
            "Block-permutation null",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Need positions + asset_returns",
        )
        return [incomplete, incomplete2]

    pos = b.positions.fillna(0.0)
    # Use IS only for nulls (selection happened on IS)
    mask = pos.index < b.oos_start
    pos_is = pos[mask]
    ret_is = b.asset_returns.reindex(pos_is.index).fillna(0.0)
    shift = _shift_null(pos_is, ret_is, n=n_null, seed=seed)
    perm = _perm_null(pos_is, ret_is, n=n_null, seed=seed + 1)
    return [
        _null_result("A4.1", "Shift null", shift),
        _null_result("A4.2", "Block-permutation null", perm),
    ]


def _sr_raw(pos: np.ndarray, ret: np.ndarray) -> float:
    x = pos * ret
    s = x.std(ddof=1)
    return 0.0 if s == 0 or not np.isfinite(s) else float(x.mean() / s)


def _shift_null(pos: pd.Series, ret: pd.Series, n: int, seed: int) -> dict:
    p = pos.values.astype(float)
    r = ret.values.astype(float)
    N = len(p)
    obs = _sr_raw(p, r)
    rng = np.random.default_rng(seed)
    lo = max(1, int(0.05 * N))
    if N <= 2 * lo + 1:
        return {"p": None, "n": 0, "observed": obs}
    ks = rng.integers(lo, N - lo, size=n)
    null = np.array([_sr_raw(np.roll(p, int(k)), r) for k in ks])
    pval = float((1 + (null >= obs).sum()) / (n + 1))
    return {
        "p": pval,
        "n": n,
        "observed": obs,
        "null_q50": float(np.quantile(null, 0.5)),
        "null_q95": float(np.quantile(null, 0.95)),
        "seed": seed,
    }


def _perm_null(pos: pd.Series, ret: pd.Series, n: int, seed: int) -> dict:
    p = pos.values.astype(float)
    r = ret.values.astype(float)
    changes = np.flatnonzero(np.diff(p) != 0)
    block = int(max(1, np.median(np.diff(changes)))) if len(changes) > 1 else 5
    blocks = [p[i : i + block] for i in range(0, len(p), block)]
    rng = np.random.default_rng(seed)
    obs = _sr_raw(p, r)
    null = []
    for _ in range(n):
        order = rng.permutation(len(blocks))
        shuffled = np.concatenate([blocks[i] for i in order])[: len(p)]
        null.append(_sr_raw(shuffled, r))
    null_a = np.array(null)
    pval = float((1 + (null_a >= obs).sum()) / (n + 1))
    return {
        "p": pval,
        "n": n,
        "observed": obs,
        "block": block,
        "null_q50": float(np.quantile(null_a, 0.5)),
        "null_q95": float(np.quantile(null_a, 0.95)),
        "seed": seed,
    }


def _null_result(id_: str, name: str, pack: dict) -> TestResult:
    p = pack.get("p")
    if p is None:
        return TestResult(id_, name, Verdict.INCOMPLETE, CheckType.RESEARCH, extras=pack)
    if p < 0.05:
        v = Verdict.PASS
    elif p >= 0.10:
        v = Verdict.FAIL
    else:
        v = Verdict.BORDERLINE
    return TestResult(
        id_,
        name,
        v,
        CheckType.RESEARCH,
        metric=p,
        threshold_pass=0.05,
        threshold_fail=0.10,
        detail=f"p={p:.4f}",
        extras=pack,
    )


# ---------------------------------------------------------------------------
# A4.3 DSR
# ---------------------------------------------------------------------------

def _a43_dsr(b: StrategyBundle) -> TestResult:
    dsr = deflated_sharpe(b.is_returns, n_trials=max(1, b.n_trials))
    if dsr is None:
        return TestResult(
            "A4.3",
            "DSR at honest N",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Insufficient IS returns for DSR",
        )
    if dsr >= 0.95:
        v = Verdict.PASS
    elif dsr < 0.80:
        v = Verdict.FAIL
    else:
        v = Verdict.BORDERLINE
    return TestResult(
        "A4.3",
        "DSR at honest N",
        v,
        CheckType.RESEARCH,
        metric=dsr,
        threshold_pass=0.95,
        threshold_fail=0.80,
        detail=f"DSR={dsr:.3f} at N={b.n_trials}",
        extras={"n_trials": b.n_trials},
    )


# ---------------------------------------------------------------------------
# A4.4 PSR / MinTRL
# ---------------------------------------------------------------------------

def _a44_psr_mintrl(b: StrategyBundle) -> TestResult:
    oos = b.oos_returns
    if len(oos) < 10:
        return TestResult(
            "A4.4",
            "PSR / MinTRL (OOS)",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Need ≥10 OOS observations",
        )
    psr = probabilistic_sharpe(oos, sr_star_annual=0.0)
    mintrl = min_track_record_length(oos, sr_star_annual=0.0, alpha=0.05)
    T = len(oos)
    if mintrl is None or psr is None:
        return TestResult(
            "A4.4",
            "PSR / MinTRL (OOS)",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Could not compute PSR/MinTRL",
        )
    ratio = T / mintrl if mintrl > 0 else 0.0
    if ratio >= 1.0:
        v = Verdict.PASS
    elif ratio >= 0.5:
        v = Verdict.BORDERLINE
    else:
        v = Verdict.FAIL
    return TestResult(
        "A4.4",
        "PSR / MinTRL (OOS)",
        v,
        CheckType.RESEARCH,
        metric=ratio,
        detail=f"T={T}, MinTRL={mintrl:.0f}, T/MinTRL={ratio:.2f}, PSR(0)={psr:.3f}",
        extras={"T": T, "mintrl": mintrl, "psr": psr, "ratio": ratio},
    )


# ---------------------------------------------------------------------------
# A4.6 PBO proxy via CSCV on optional grid; else INCOMPLETE
# ---------------------------------------------------------------------------

def _a46_pbo_proxy(b: StrategyBundle) -> TestResult:
    grid = b.parameter_grid_results
    if not grid or len(grid) < 4:
        return TestResult(
            "A4.6",
            "PBO (CSCV)",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Need parameter_grid_results with ≥4 variants (each with is_returns & oos_returns)",
        )
    # Expect each item: {"id", "is_sharpe", "oos_sharpe"} or return series
    pairs = []
    for item in grid:
        if "is_sharpe" in item and "oos_sharpe" in item:
            pairs.append((float(item["is_sharpe"]), float(item["oos_sharpe"])))
        elif "is_returns" in item and "oos_returns" in item:
            pairs.append(
                (
                    float(sharpe(pd.Series(item["is_returns"])) or 0.0),
                    float(sharpe(pd.Series(item["oos_returns"])) or 0.0),
                )
            )
    if len(pairs) < 4:
        return TestResult(
            "A4.6",
            "PBO (CSCV)",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Grid variants missing is/oos Sharpe fields",
        )
    # Simplified PBO: fraction where IS rank is top-half but OOS rank is bottom-half
    is_s = np.array([p[0] for p in pairs])
    oo_s = np.array([p[1] for p in pairs])
    is_rank = is_s.argsort().argsort()
    oo_rank = oo_s.argsort().argsort()
    mid = len(pairs) / 2.0
    # Combinatorial symmetric cross-validation approximation:
    # PBO ≈ P(OOS rank of IS-best is in bottom half)
    best_is = int(is_s.argmax())
    pbo_simple = 1.0 if oo_rank[best_is] < mid else 0.0
    # Soften with Kendall-style: corr of ranks; map to [0,1] overfitting score
    if len(pairs) >= 8:
        # Leave-one-out style: for each held-out, check if IS winner is below median OOS
        fails = 0
        for i in range(len(pairs)):
            mask = np.ones(len(pairs), dtype=bool)
            mask[i] = False
            winner = int(is_s[mask].argmax())
            # map winner index back
            idxs = np.flatnonzero(mask)
            w = idxs[winner]
            if oo_s[w] <= np.median(oo_s[mask]):
                fails += 1
        pbo = fails / len(pairs)
    else:
        pbo = pbo_simple

    if pbo <= 0.3:
        v = Verdict.PASS
    elif pbo > 0.5:
        v = Verdict.FAIL
    else:
        v = Verdict.BORDERLINE
    return TestResult(
        "A4.6",
        "PBO (CSCV)",
        v,
        CheckType.RESEARCH,
        metric=pbo,
        threshold_pass=0.3,
        threshold_fail=0.5,
        detail=f"PBO≈{pbo:.2f} over {len(pairs)} variants",
        extras={"n_variants": len(pairs), "is_oos_rank_corr": float(np.corrcoef(is_rank, oo_rank)[0, 1])},
    )


# ---------------------------------------------------------------------------
# A5 OOS hold (G3/G4)
# ---------------------------------------------------------------------------

def _a5_oos_hold(b: StrategyBundle) -> TestResult:
    is_sr = sharpe(b.is_returns)
    oos_sr = sharpe(b.oos_returns)
    if is_sr is None or oos_sr is None:
        return TestResult(
            "A5.1",
            "OOS hold (IS→OOS)",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Need IS and OOS returns",
        )
    extras = {"is_sharpe": is_sr, "oos_sharpe": oos_sr}
    if oos_sr <= 0:
        return TestResult(
            "A5.1",
            "OOS hold (IS→OOS)",
            Verdict.FAIL,
            CheckType.RESEARCH,
            metric=oos_sr,
            detail=f"OOS SR={oos_sr:.3f} ≤ 0",
            extras=extras,
        )
    if is_sr >= 0.3 and oos_sr >= 0.5 and oos_sr >= 0.5 * is_sr:
        return TestResult(
            "A5.1",
            "OOS hold (IS→OOS)",
            Verdict.PASS,
            CheckType.RESEARCH,
            metric=oos_sr / is_sr if is_sr else None,
            detail=f"IS={is_sr:.2f}, OOS={oos_sr:.2f} (≥50% of IS, ≥0.5)",
            extras=extras,
        )
    if oos_sr >= 0.5 * max(is_sr, 1e-9):
        return TestResult(
            "A5.1",
            "OOS hold (IS→OOS)",
            Verdict.BORDERLINE,
            CheckType.RESEARCH,
            metric=oos_sr,
            detail=f"Partial hold IS={is_sr:.2f} OOS={oos_sr:.2f}",
            extras=extras,
        )
    return TestResult(
        "A5.1",
        "OOS hold (IS→OOS)",
        Verdict.FAIL,
        CheckType.RESEARCH,
        metric=oos_sr,
        detail=f"OOS decay IS={is_sr:.2f} → OOS={oos_sr:.2f}",
        extras=extras,
    )


# ---------------------------------------------------------------------------
# A6.1 Parameter neighbourhood
# ---------------------------------------------------------------------------

def _a61_param_neighbourhood(b: StrategyBundle) -> TestResult:
    grid = b.parameter_grid_results
    if not grid or len(grid) < 3:
        return TestResult(
            "A6.1",
            "Parameter neighbourhood",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Need parameter_grid_results (≥3 cells with 'sharpe')",
        )
    srs = np.array([float(x["sharpe"]) for x in grid if "sharpe" in x])
    if len(srs) < 3:
        return TestResult(
            "A6.1",
            "Parameter neighbourhood",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Grid cells missing sharpe",
        )
    best = float(srs.max())
    # Stability S = fraction within 20% of best; P+ = fraction positive
    if abs(best) < 1e-12:
        S = float((srs >= 0).mean())
    else:
        S = float((srs >= 0.8 * best).mean())
    Pp = float((srs > 0).mean())
    extras = {"S": S, "P_plus": Pp, "best": best, "n_cells": len(srs)}
    if S >= 0.7 and Pp >= 0.8:
        v = Verdict.PASS
    elif S < 0.5:
        v = Verdict.FAIL
    else:
        v = Verdict.BORDERLINE
    return TestResult(
        "A6.1",
        "Parameter neighbourhood",
        v,
        CheckType.RESEARCH,
        metric=S,
        threshold_pass=0.7,
        threshold_fail=0.5,
        detail=f"S={S:.2f}, P+={Pp:.2f}",
        extras=extras,
    )


# ---------------------------------------------------------------------------
# A9.1 Concentration
# ---------------------------------------------------------------------------

def _a91_concentration(b: StrategyBundle) -> TestResult:
    r = b.full_dev_returns.dropna()
    if len(r) < 60:
        return TestResult(
            "A9.1",
            "Profit concentration",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Need ≥60 returns",
        )
    gains = r.clip(lower=0.0)
    total = float(gains.sum())
    if total <= 0:
        return TestResult(
            "A9.1",
            "Profit concentration",
            Verdict.FAIL,
            CheckType.RESEARCH,
            detail="No positive returns",
        )
    top_n = max(1, int(0.01 * len(r)))
    c1 = float(gains.nlargest(top_n).sum() / total)
    # Leave-one-year-out
    years = r.index.year.unique()
    loyo = []
    base_sr = sharpe(r) or 0.0
    for y in years:
        sr_y = sharpe(r[r.index.year != y])
        if sr_y is not None and abs(base_sr) > 1e-12:
            loyo.append(sr_y / base_sr)
    loyo_min = float(min(loyo)) if loyo else None
    extras = {"C1pct": c1, "LOYOmin": loyo_min, "n_years": int(len(years))}
    if c1 > 1.0 or (loyo_min is not None and loyo_min <= 0):
        v = Verdict.FAIL
    elif c1 <= 0.5 and (loyo_min is None or loyo_min >= 0.5):
        v = Verdict.PASS
    else:
        v = Verdict.BORDERLINE
    return TestResult(
        "A9.1",
        "Profit concentration",
        v,
        CheckType.RESEARCH,
        metric=c1,
        detail=f"C1%={c1:.1%}, LOYOmin={loyo_min}",
        extras=extras,
    )


# ---------------------------------------------------------------------------
# A10.1 Regimes (vol quartile)
# ---------------------------------------------------------------------------

def _a101_regimes(b: StrategyBundle) -> TestResult:
    r = b.full_dev_returns.dropna()
    if len(r) < 120:
        return TestResult(
            "A10.1",
            "Regime-conditional performance",
            Verdict.INCOMPLETE,
            CheckType.ADVISORY,
            detail="Need ≥120 returns",
        )
    vol = r.rolling(21).std().shift(1)  # lagged — no lookahead
    ok = vol.dropna()
    r2 = r.reindex(ok.index)
    qs = pd.qcut(ok, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    regime_sr = {}
    for lab in qs.cat.categories:
        regime_sr[str(lab)] = sharpe(r2[qs == lab])
    positive = sum(1 for v in regime_sr.values() if v is not None and v > 0)
    extras = {"regime_sharpe": regime_sr, "positive_regimes": positive}
    if positive >= 3:
        v = Verdict.PASS
    elif positive <= 1:
        v = Verdict.FAIL
    else:
        v = Verdict.BORDERLINE
    return TestResult(
        "A10.1",
        "Regime-conditional performance",
        v,
        CheckType.ADVISORY,
        metric=float(positive),
        detail=f"SR>0 in {positive}/{len(regime_sr)} vol regimes",
        extras=extras,
    )


# ---------------------------------------------------------------------------
# A11.1 Attribution vs benchmark
# ---------------------------------------------------------------------------

def _a111_attribution(b: StrategyBundle) -> TestResult:
    if b.benchmark_returns is None:
        return TestResult(
            "A11.1",
            "Benchmark attribution",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="No benchmark_returns supplied",
        )
    df = pd.concat(
        [b.full_dev_returns.rename("r"), b.benchmark_returns.rename("b")],
        axis=1,
    ).dropna()
    if len(df) < 60:
        return TestResult(
            "A11.1",
            "Benchmark attribution",
            Verdict.INCOMPLETE,
            CheckType.RESEARCH,
            detail="Need ≥60 overlapping observations",
        )
    cov = np.cov(df["r"], df["b"])
    var_b = cov[1, 1]
    beta = float(cov[0, 1] / var_b) if var_b > 0 else 0.0
    resid = df["r"] - beta * df["b"]
    sr_res = sharpe(resid)
    rho = float(df["r"].corr(df["b"]))
    extras = {"beta": beta, "rho": rho, "sr_resid": sr_res}
    if sr_res is not None and sr_res >= 0.5 and abs(rho) < 0.3:
        v = Verdict.PASS
    elif sr_res is not None and sr_res <= 0:
        v = Verdict.FAIL
    else:
        v = Verdict.BORDERLINE
    return TestResult(
        "A11.1",
        "Benchmark attribution",
        v,
        CheckType.RESEARCH,
        metric=sr_res,
        detail=f"SRresid={sr_res}, ρ={rho:.2f}, β={beta:.2f}",
        extras=extras,
    )


# ---------------------------------------------------------------------------
# A12.1 Book correlation
# ---------------------------------------------------------------------------

def _a121_book_corr(b: StrategyBundle) -> TestResult:
    if not b.book_returns:
        return TestResult(
            "A12.1",
            "Book correlation",
            Verdict.INCOMPLETE,
            CheckType.PORTFOLIO,
            detail="No book_returns supplied",
        )
    corrs = {}
    for name, series in b.book_returns.items():
        df = pd.concat([b.full_dev_returns, series], axis=1).dropna()
        if len(df) < 60:
            continue
        corrs[name] = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
    if not corrs:
        return TestResult(
            "A12.1",
            "Book correlation",
            Verdict.INCOMPLETE,
            CheckType.PORTFOLIO,
            detail="Insufficient overlap with book series",
        )
    max_rho = max(corrs.values(), key=lambda x: abs(x))
    max_name = max(corrs, key=lambda k: abs(corrs[k]))
    if abs(max_rho) < 0.7:
        v = Verdict.PASS
    elif abs(max_rho) >= 0.8:
        v = Verdict.FAIL
    else:
        v = Verdict.BORDERLINE
    return TestResult(
        "A12.1",
        "Book correlation",
        v,
        CheckType.PORTFOLIO,
        metric=max_rho,
        detail=f"max |ρ|={abs(max_rho):.2f} vs {max_name}",
        extras={"correlations": corrs},
    )
