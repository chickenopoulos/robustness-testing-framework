"""Robustness Testing Framework (RTF).

Implements Part A (post-search validation) and Part C (kill-switch
calibration) from the Robustness Testing Frameworks research document.
"""

from .bundle import StrategyBundle, load_bundle
from .pipeline import evaluate_strategy, EvaluationReport
from .kill_switch.policy import calibrate_kill_policy, KillPolicyReport

__all__ = [
    "StrategyBundle",
    "load_bundle",
    "evaluate_strategy",
    "EvaluationReport",
    "calibrate_kill_policy",
    "KillPolicyReport",
]

__version__ = "1.0.0"
