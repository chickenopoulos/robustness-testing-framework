"""Shared enums and result containers (fail-closed vocabulary)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Verdict(str, Enum):
    PASS = "PASS"
    BORDERLINE = "BORDERLINE"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"


class CheckType(str, Enum):
    BLOCKING = "blocking"
    RESEARCH = "research"
    ADVISORY = "advisory"
    DISPLAY = "display"
    PORTFOLIO = "portfolio"


class Tier(str, Enum):
    CLEARED = "Cleared"
    PROVISIONAL = "Provisional"
    INCOMPLETE = "Incomplete"
    FAILED = "Failed"
    NOT_CREDIBLE = "Not credible"


class ResponseLevel(str, Enum):
    NORMAL = "NORMAL"
    WARN = "WARN"
    REDUCE_HALF = "REDUCE_HALF"
    PAUSE_FLAT_KEEP_SHADOW = "PAUSE_FLAT_KEEP_SHADOW"
    RETIRE = "RETIRE"


@dataclass
class TestResult:
    id: str
    name: str
    verdict: Verdict
    check_type: CheckType
    metric: Optional[float] = None
    threshold_pass: Optional[float] = None
    threshold_fail: Optional[float] = None
    detail: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["check_type"] = self.check_type.value
        return d


@dataclass
class GateResult:
    gate: str
    rule: str
    passed: Optional[bool]  # None = not assessable / incomplete
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
