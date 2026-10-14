# Robustness Testing Framework (RTF)

Standalone library to **evaluate any strategy quickly** after a search, and to **define a quantitative kill switch** before (paper) deployment.

Implements **Part A (post-search validation)** and **Part C (kill signals)** from *Robustness Testing Frameworks* (v1.0, 2026-10-14): fail-closed checks, honest trial-count statistics, G0–G9 gates / tiers, and a simulation-calibrated kill policy (stationary bootstrap, joint false-kill budget).

```text
integrity → implementation → costs → stats → OOS / stability → score & tier
                                                                    ↓
                                              kill-policy calibration (Part C)
```

---

## Install

```bash
cd /root/robustness-testing-framework
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## What you need as input

### Required

| Input | Type | Meaning |
|-------|------|---------|
| `returns` | `pd.Series`, DatetimeIndex (UTC) | **Net** strategy returns after costs |
| `oos_start` | timestamp | First bar of out-of-sample (IS is everything strictly before) |

### Strongly recommended (unlocks most of the battery)

| Input | Type | Unlocks |
|-------|------|---------|
| `positions` | `pd.Series` in `[-1, 1]` | Delay sweep, nulls, trade counts (G2), cost sweep |
| `asset_returns` | underlying close-to-close returns | Rebuild fills / verify lag ≥ 1 |
| `n_trials` | int ≥ 1 | **Honest** count of configs tried on the same data (DSR / G8) |
| `cost_bps_base` | float | Baseline one-way cost for 1×/2×/3× sweep (G5) |

### Optional

| Input | Unlocks |
|-------|---------|
| `claimed_metrics` `{sharpe,cagr,maxdd}` | Ledger reconciliation (A2.3) |
| `parameter_grid_results` | Neighbourhood stability (A6.1), PBO proxy (A4.6) |
| `benchmark_returns` | Residual attribution (A11.1) |
| `book_returns` | Correlation to live book (A12.1) |
| `sealed_start` | Default `2025-10-01` — any row on/after → **Not credible** |
| `meta` | `strategy_id`, `frozen_code`, `owner`, symbol, … |

**CSV layout** (also accepted by the CLI): two columns `date,value`, or a Series CSV with a date index.

```csv
date,returns
2019-01-02,0.0012
2019-01-03,-0.0004
```

---

## Quick start

### Python

```python
from rtf import StrategyBundle, evaluate_strategy

bundle = StrategyBundle(
    returns=returns,                 # net returns
    oos_start="2024-01-01",
    positions=positions,             # optional but recommended
    asset_returns=asset_returns,     # optional but recommended
    n_trials=50,                     # honest N
    cost_bps_base=5.0,
    claimed_metrics={"sharpe": 1.1, "cagr": 0.18, "maxdd": -0.12},
    meta={"strategy_id": "my_strat"},
)

report = evaluate_strategy(bundle)   # Part A + Part C
print(report.tier)                   # Cleared | Provisional | Incomplete | Failed | Not credible
report.save("out/report.json")       # also writes .md and .kill_policy.yaml
```

### CLI

```bash
rtf evaluate \
  --returns examples/data/returns.csv \
  --positions examples/data/positions.csv \
  --asset-returns examples/data/asset_returns.csv \
  --oos-start 2024-01-01 \
  --n-trials 50 \
  --cost-bps 5 \
  --claimed-metrics examples/data/claimed_metrics.json \
  --strategy-id demo_momentum \
  --out examples/output/report.json
```

Kill policy only:

```bash
rtf calibrate-kills --returns examples/data/returns.csv --out kill_policy.yaml
```

Apply policy to live (or shadow) returns:

```bash
rtf check-live --live-returns live.csv --policy kill_policy.yaml
```

### Example walkthrough

```bash
python examples/generate_sample_data.py
python examples/run_example.py
```

---

## What the framework runs

### Part A — post-search battery

| ID | Test | Pass (default) | Fail | Type |
|----|------|----------------|------|------|
| A1.3 | Sealed-window guard | no sealed rows | sealed row in returns | Blocking |
| A2.1 | Next-bar fills | lag≥1 consistent | same-bar signature | Blocking |
| A2.2 | Delay sweep 0–3 | D2≥0.5, D3≥0.3 | D2\<0.3 / sign flip / L0≫L1 | Research |
| A2.3 | Ledger reconciliation | within tolerance | mismatch | Blocking |
| A2.5 | Equity sanity | no flags | nearly-flat / extremes | Blocking |
| A3.1 | Cost 1×/2×/3× | SR(2×)\>0 & ≥50% base | SR(2×)≤0 | Research (G5) |
| A4.1–2 | Shift / block nulls | p\<0.05 | p≥0.10 | Research (G9) |
| A4.3 | DSR at honest N | ≥0.95 | \<0.80 | Research (G8) |
| A4.4 | PSR / MinTRL (OOS) | T ≥ MinTRL | T \< ½ MinTRL | Research |
| A4.6 | PBO (grid proxy) | ≤0.3 | \>0.5 | Research |
| A5.1 | OOS hold | OOS≥0.5 & ≥50% IS | OOS≤0 | Research (G3/G4) |
| A6.1 | Parameter neighbourhood | S≥0.7, P+≥0.8 | S\<0.5 | Research (G6) |
| A9.1 | Concentration | C1%≤50%, LOYO≥50% | C1%\>100% or LOYO≤0 | Research |
| A10.1 | Vol regimes | SR\>0 in ≥3/4 | one-regime edge | Advisory |
| A11.1 | Benchmark attribution | SRresid≥0.5, \|ρ\|\<0.3 | α≤0 | Research |
| A12.1 | Book correlation | max \|ρ\|\<0.7 | ≥0.8 | Portfolio |

**Fail-closed:** missing inputs return `INCOMPLETE`, never a default pass. Blocking `FAIL`s stop “Cleared”.

### Gates & tiers (A.13)

Gates **G0–G9** mirror the document (integrity, OOS length, trades, OOS SR, IS→OOS, cost 2×, parameter plateau, OOS MaxDD, DSR, nulls).

| Tier | Rule |
|------|------|
| **Cleared** | Every gate evaluated and passed |
| **Provisional** | No gate failed; ≥1 not yet assessed |
| **Incomplete** | Missing evidence (e.g. short OOS / missing artefacts) |
| **Failed** | Any other gate failed |
| **Not credible** | Leakage, sealed access, look-ahead, unreproducible ledger |

Score = `10 × P(OOS holds | IS)` via Bayesian shrinkage. Always labelled **weakly calibrated** (programme finding: barely better than base rate).

### Part C — kill-switch battery

Before live capital, RTF calibrates a **hashed** policy from haircut backtest returns (`sharpe_haircut=0.5` by default):

1. Stationary bootstrap, mean block ≈ Politis–White heuristic (floored at holding period)
2. Joint false-kill budget (default **5%** over **3 years**), Bonferroni-split across MaxDD / TuW / CUSUM
3. Simultaneous (path-wise) bands — not naive pointwise percentiles
4. Graduated ladder: `NORMAL → WARN → REDUCE_HALF → PAUSE_FLAT_KEEP_SHADOW → RETIRE`
5. Operational monitors (staleness, config drift, tracking z) included in the template

---

## Example outputs

### Markdown summary (from `examples/run_example.py`)

The bundled synthetic momentum rule clears plumbing and several research bars, then **fails the research bar** on honest DSR / trade count / parameter plateau — the usual “code is fine, edge not proven” outcome:

```markdown
# Robustness evaluation — **Failed**

Score: `5.47` (P=0.547) — Weakly calibrated …

## Gates
| Gate | Passed | Detail |
|------|--------|--------|
| G0 | PASS | ok |
| G1 | PASS | oos_years=1.75 |
| G2 | FAIL | trades_full=81, trades_oos=7 |
| G3 | PASS | oos_sharpe=1.87 |
| G4 | PASS | is=0.47, oos=1.87 |
| G5 | PASS | SR(2×)=0.72 ≥ 50% of SR(1×)=0.80 |
| G6 | FAIL | S=0.24, P+=0.94 |
| G7 | PASS | oos_maxdd=-0.15 |
| G8 | FAIL | DSR=0.13 at N=50 |
| G9 | PASS | p=0.02; p=0.03 |

## Kill policy (calibrated)
- Sharpe haircut: 0.5
- False-kill budget (joint, 3y): 0.05
- MaxDD pause (additive): -0.42
- TuW pause (bars): 777
- CUSUM h: 0.47
- Hard capital cap: -0.24
```

### `report.json` (shape)

```json
{
  "tier": "Failed",
  "score": {
    "score": 5.47,
    "probability": 0.547,
    "calibration": "Weakly calibrated on programme data …"
  },
  "gates": [{"gate": "G8", "passed": false, "detail": "DSR=0.131 at N=50"}],
  "tests": [{"id": "A4.3", "verdict": "FAIL", "metric": 0.131}],
  "kill_policy": {
    "policy_hash": "…",
    "policy": { "monitors": { "max_drawdown": { "pause": -0.419 } } },
    "calibration": { "per_monitor_alpha": 0.0167 }
  },
  "notes": ["Incomplete checks (fail-closed, not imputed): A11.1, A12.1"]
}
```

### Live check

```json
{
  "response": "PAUSE_FLAT_KEEP_SHADOW",
  "reasons": ["maxdd=-0.31 ≤ pause/hard_cap", "cusum=0.11 ≥ h"],
  "live_summary": {"sharpe": -1.4, "time_under_water": 54, "psr_0": 0.08},
  "policy_hash": "…"
}
```

---

## Design rules (from the document)

1. **Fail closed** — missing evidence is `INCOMPLETE`, never a soft pass.
2. **Artefacts are truth** — metrics are recomputed from stored returns.
3. **Blocking vs research** — integrity defects ≠ “no edge”; statistical weakness ≠ “buggy code”.
4. **Honest N** — DSR/PBO are only as good as the trial count you declare.
5. **Kill rules catch disasters / ops**, not slow decay — size down with live PSR; don’t expect a 5% false-kill budget to spot Sharpe 1→0 inside three years.

---

## Layout

```text
rtf/
  bundle.py              # StrategyBundle input contract
  fills.py               # next-bar fills + delay sweep
  metrics.py             # Sharpe, DSR, PSR, MinTRL, …
  pipeline.py            # evaluate_strategy()
  post_search/           # Part A checks + G0–G9 scoring
  kill_switch/           # Part C simulation + policy + live check
  cli.py                 # `rtf` entry point
examples/                # sample data + walkthrough
tests/
```

---

## Not in this package (by design)

- In-search / GA mining controls (**Part B**) — belongs in the search harness.
- Live brokerage wiring — RTF only calibrates and evaluates the policy.
- Automatic packaging / deployment of “Cleared” strategies.

---

## License

Proprietary — internal research tooling.
