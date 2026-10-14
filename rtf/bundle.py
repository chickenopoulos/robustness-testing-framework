"""Strategy input contract.

Minimum inputs for a quick evaluation:

Required
--------
- returns: pd.Series of net strategy returns (after costs), DatetimeIndex
- oos_start: timestamp splitting IS / OOS (IS ends strictly before this)

Strongly recommended
---------------------
- positions: per-bar position in [-1, 1] (enables delay sweep, nulls, trade counts)
- asset_returns: underlying close-to-close returns aligned to positions
- n_trials: honest count of configurations tried on the same data (for DSR)
- claimed_metrics: optional ledger dict to reconcile against recomputed metrics

Optional
--------
- cost_bps_base: baseline one-way cost in bps used to build returns
- sealed_start: calendar guard (default 2025-10-01)
- parameter_grid_results: list of {params, sharpe} for neighbourhood stability
- benchmark_returns: for residual / attribution checks
- book_returns: dict of other live strategy return series for correlation
- meta: free-form dict (symbol, timeframe, strategy_id, …)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .metrics import equity_curve, n_trades_from_positions, summarize


DEFAULT_SEALED_START = "2025-10-01"


@dataclass
class StrategyBundle:
    returns: pd.Series
    oos_start: pd.Timestamp
    positions: Optional[pd.Series] = None
    asset_returns: Optional[pd.Series] = None
    n_trials: int = 1
    claimed_metrics: Optional[dict[str, Any]] = None
    cost_bps_base: float = 0.0
    sealed_start: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp(DEFAULT_SEALED_START, tz="UTC")
    )
    parameter_grid_results: Optional[list[dict[str, Any]]] = None
    benchmark_returns: Optional[pd.Series] = None
    book_returns: Optional[dict[str, pd.Series]] = None
    meta: dict[str, Any] = field(default_factory=dict)
    allow_same_bar: bool = False

    def __post_init__(self) -> None:
        self.returns = _as_utc_series(self.returns, "returns")
        self.oos_start = _as_utc_ts(self.oos_start)
        self.sealed_start = _as_utc_ts(self.sealed_start)
        if self.positions is not None:
            self.positions = _as_utc_series(self.positions, "positions").reindex(
                self.returns.index
            )
        if self.asset_returns is not None:
            self.asset_returns = _as_utc_series(
                self.asset_returns, "asset_returns"
            ).reindex(self.returns.index)
        if self.benchmark_returns is not None:
            self.benchmark_returns = _as_utc_series(
                self.benchmark_returns, "benchmark_returns"
            ).reindex(self.returns.index)

    @property
    def is_returns(self) -> pd.Series:
        return self.returns[self.returns.index < self.oos_start]

    @property
    def oos_returns(self) -> pd.Series:
        return self.returns[
            (self.returns.index >= self.oos_start)
            & (self.returns.index < self.sealed_start)
        ]

    @property
    def full_dev_returns(self) -> pd.Series:
        """All returns strictly before the sealed boundary."""
        return self.returns[self.returns.index < self.sealed_start]

    def returns_hash(self) -> str:
        payload = self.returns.dropna().round(12).to_csv().encode()
        return hashlib.sha256(payload).hexdigest()

    def summary(self) -> dict[str, Any]:
        trades = None
        if self.positions is not None:
            trades = {
                "full": n_trades_from_positions(self.positions),
                "is": n_trades_from_positions(
                    self.positions[self.positions.index < self.oos_start]
                ),
                "oos": n_trades_from_positions(
                    self.positions[
                        (self.positions.index >= self.oos_start)
                        & (self.positions.index < self.sealed_start)
                    ]
                ),
            }
        return {
            "meta": self.meta,
            "n_trials": self.n_trials,
            "oos_start": str(self.oos_start.date()),
            "sealed_start": str(self.sealed_start.date()),
            "returns_hash": self.returns_hash(),
            "is": summarize(self.is_returns),
            "oos": summarize(self.oos_returns),
            "full_dev": summarize(self.full_dev_returns),
            "trades": trades,
            "has_positions": self.positions is not None,
            "has_asset_returns": self.asset_returns is not None,
        }


def _as_utc_ts(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _as_utc_series(s: pd.Series, name: str) -> pd.Series:
    out = pd.Series(s, dtype=float).copy()
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError(f"{name}: duplicate timestamps are not allowed")
    out.name = name
    return out


def load_bundle(
    returns_path: str | Path,
    oos_start: str,
    *,
    positions_path: str | Path | None = None,
    asset_returns_path: str | Path | None = None,
    n_trials: int = 1,
    cost_bps_base: float = 0.0,
    sealed_start: str = DEFAULT_SEALED_START,
    claimed_metrics_path: str | Path | None = None,
    meta: Optional[dict[str, Any]] = None,
) -> StrategyBundle:
    """Load a strategy bundle from CSV files (date,value columns or a Series CSV)."""
    returns = _read_series(returns_path, "returns")
    positions = _read_series(positions_path, "positions") if positions_path else None
    asset_returns = (
        _read_series(asset_returns_path, "asset_returns") if asset_returns_path else None
    )
    claimed = None
    if claimed_metrics_path:
        claimed = json.loads(Path(claimed_metrics_path).read_text())
    return StrategyBundle(
        returns=returns,
        oos_start=pd.Timestamp(oos_start),
        positions=positions,
        asset_returns=asset_returns,
        n_trials=int(n_trials),
        cost_bps_base=float(cost_bps_base),
        sealed_start=pd.Timestamp(sealed_start),
        claimed_metrics=claimed,
        meta=meta or {},
    )


def _read_series(path: str | Path, name: str) -> pd.Series:
    path = Path(path)
    df = pd.read_csv(path)
    if len(df.columns) < 2:
        # single-column with date index written by to_csv
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        s = df.iloc[:, 0]
    else:
        date_col = df.columns[0]
        val_col = df.columns[1]
        s = pd.Series(
            df[val_col].values,
            index=pd.to_datetime(df[date_col], utc=True),
            name=name,
        )
    return s


def rebuild_equity_from_returns(returns: pd.Series) -> pd.Series:
    return equity_curve(returns)
