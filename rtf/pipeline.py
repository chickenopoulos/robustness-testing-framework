"""Top-level evaluation: Part A battery + optional Part C kill calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .bundle import StrategyBundle
from .kill_switch.policy import KillPolicyReport, calibrate_kill_policy
from .post_search.checks import run_post_search_battery
from .post_search.scoring import assign_tier, evaluate_gates, robustness_score
from .types import GateResult, TestResult, Tier


@dataclass
class EvaluationReport:
    summary: dict[str, Any]
    tests: list[TestResult]
    gates: list[GateResult]
    tier: Tier
    score: dict[str, Any]
    kill_policy: Optional[KillPolicyReport] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "score": self.score,
            "summary": self.summary,
            "gates": [g.to_dict() for g in self.gates],
            "tests": [t.to_dict() for t in self.tests],
            "kill_policy": None if self.kill_policy is None else self.kill_policy.to_dict(),
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        if self.kill_policy is not None:
            yml = path.with_suffix(".kill_policy.yaml")
            yml.write_text(self.kill_policy.to_yaml())
        md = path.with_suffix(".md")
        md.write_text(render_markdown(self))
        return path


def evaluate_strategy(
    bundle: StrategyBundle,
    *,
    calibrate_kills: bool = True,
    n_null: int = 500,
    n_kill_paths: int = 3000,
    kill_horizon_years: float = 3.0,
    false_kill_budget: float = 0.05,
    sharpe_haircut: float = 0.5,
    seed: int = 42,
) -> EvaluationReport:
    """Run the post-search battery and (optionally) calibrate a kill policy."""
    notes: list[str] = []
    tests = run_post_search_battery(bundle, n_null=n_null, seed=seed)
    gates = evaluate_gates(bundle, tests)
    tier = assign_tier(gates, tests)
    score = robustness_score(bundle, n_trials=bundle.n_trials)

    # Caps from A13: verification defect caps score at 3.0
    if tier == Tier.NOT_CREDIBLE:
        score = {
            **score,
            "score": None,
            "status": "Not credible",
            "note": "Leakage / look-ahead / unreproducible — score withheld",
        }
        notes.append("Score withheld: Not credible")
    elif any(t.verdict.value == "FAIL" and t.check_type.value == "blocking" for t in tests):
        if score.get("score") is not None and score["score"] > 3.0:
            score["score"] = 3.0
            score["capped"] = True
            notes.append("Score capped at 3.0 due to verification defect")

    kill: Optional[KillPolicyReport] = None
    if calibrate_kills:
        try:
            kill = calibrate_kill_policy(
                bundle.full_dev_returns,
                strategy_id=str(bundle.meta.get("strategy_id", "strategy")),
                frozen_code=str(bundle.meta.get("frozen_code", "unknown")),
                owner=str(bundle.meta.get("owner", "research")),
                sharpe_haircut=sharpe_haircut,
                horizon_years=kill_horizon_years,
                n_paths=n_kill_paths,
                false_kill_budget=false_kill_budget,
                seed=seed,
            )
        except ValueError as exc:
            notes.append(f"Kill policy not calibrated: {exc}")

    incomplete = [t.id for t in tests if t.verdict.value == "INCOMPLETE"]
    if incomplete:
        notes.append(
            f"Incomplete checks (fail-closed, not imputed): {', '.join(incomplete)}"
        )

    return EvaluationReport(
        summary=bundle.summary(),
        tests=tests,
        gates=gates,
        tier=tier,
        score=score,
        kill_policy=kill,
        notes=notes,
    )


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        f"# Robustness evaluation — **{report.tier.value}**",
        "",
        f"Score: `{report.score.get('score')}` "
        f"(P={report.score.get('probability')}) — {report.score.get('calibration', '')}",
        "",
        "## Gates",
        "",
        "| Gate | Passed | Detail |",
        "|------|--------|--------|",
    ]
    for g in report.gates:
        flag = {True: "PASS", False: "FAIL", None: "N/A"}[g.passed]
        lines.append(f"| {g.gate} | {flag} | {g.detail} |")

    lines += ["", "## Tests", "", "| ID | Verdict | Detail |", "|----|---------|--------|"]
    for t in report.tests:
        lines.append(f"| {t.id} | {t.verdict.value} | {t.detail} |")

    if report.kill_policy is not None:
        kp = report.kill_policy.policy
        m = kp["monitors"]
        lines += [
            "",
            "## Kill policy (calibrated)",
            "",
            f"Hash: `{report.kill_policy.policy_hash[:16]}…`",
            "",
            f"- Sharpe haircut: {kp['reference']['sharpe_haircut']}",
            f"- False-kill budget (joint, {kp['reference']['horizon_years']}y): "
            f"{kp['reference']['false_kill_budget']}",
            f"- MaxDD pause (additive): {m['max_drawdown']['pause']:.4f}",
            f"- TuW pause (bars): {m['time_under_water']['pause']}",
            f"- CUSUM h: {m['cusum_mean_drop']['h']:.6f}",
            f"- Hard capital cap: {m['max_drawdown']['hard_cap']:.4f}",
        ]

    if report.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in report.notes]

    lines.append("")
    return "\n".join(lines)
