# Walk-Forward Backtesting Engine Plan
**Project:** NCAAB Men's Basketball Prediction System
**Consensus:** Planner → Architect → Critic (1 revision cycle)
**Status:** APPROVED (all Critic blockers resolved below)
**Date:** 2026-03-17

---

## Requirements Summary

Walk-forward backtest of the tournament prediction model over 2016–2025 NCAA Men's Tournaments (excluding 2020 COVID gap). Measures bracket accuracy by round, Brier score, Kelly-sized ROI, Sharpe ratio, and max drawdown. Feeds calibration data into isotonic regression.

**Must support `--mock` flag** — no live API calls in test suite.

---

## Resolved Blockers (from Critic)

### B1 — CSV Schema (RESOLVED)
`data/tournament_results/{year}.csv` — one file per year, committed to git.

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `round` | str | `R64` | R64/R32/S16/E8/FF/F |
| `region` | str | `East` | East/West/Midwest/South/Final Four |
| `team_a` | str | `Duke` | **Canonical name** (not ESPN/Torvik raw) |
| `team_b` | str | `NCST` | Canonical name |
| `seed_a` | int | `1` | 1–16 |
| `seed_b` | int | `16` | 1–16 |
| `winner` | str | `Duke` | Canonical name |
| `score_a` | int | `93` | Optional (ESPN) |
| `score_b` | int | `67` | Optional |

**`.gitignore` rule:** `data/tournament_results/` is **NOT** in `.gitignore` — historical CSVs are committed reference data.

### B2 — AdjT Rank vs Tempo (RESOLVED)
**Decision: Accept default tempo for all backtesting.** BartTorvik `AdjT` returns rank (1–363), not pace. The clamp in `build_wp_table()` already handles this (values <55 or >80 → `DEFAULT_TEMPO = 67.5`). All backtest predictions use default tempo. This costs ~0.5–1% accuracy vs. having real tempo — acceptable for a 10-year historical backtest. Document in `YearResult.metadata`.

### B3 — Team Name Mapping Scope (RESOLVED)
**Strategy:** Bootstrap from `data/torvik/torvik_2026.py` (known 2026 names), then extend backwards. Use `rapidfuzz.process.extract()` with `ratio` scorer to suggest matches — human confirms. Target: 2026 tournament field (68 teams) first, then extend to 2016–2025.

Deliverable: `data/team_name_map.csv` with columns: `canonical, torvik_name, espn_name, alt_names`.
Initial population via `scripts/build_name_map.py` (fuzzy bootstrap + manual review).

### B4 — Historical BartTorvik Caching (RESOLVED)
**Decision: Pre-cache all historical BartTorvik data.** `data/cache/torvik_{year}.json` stores raw scraped data. `NcaabDataLoader` checks cache first, scrapes only on `force_refresh=True` or cache miss. Cache files are committed to git (small JSON, ~50KB/year).

### B5 — GamePrediction Dataclass Fields (RESOLVED)
```python
@dataclass
class GamePrediction:
    year: int
    round: str                    # R64/R32/S16/E8/FF/F
    team_a: str                   # canonical name
    team_b: str                   # canonical name
    seed_a: int
    seed_b: int
    pred_prob_a: float            # raw model probability
    cal_prob_a: Optional[float]   # isotonic-calibrated probability (None if not fitted)
    actual_winner: str            # canonical name
    ml_a: Optional[int]           # American ML for team_a at open (None if no line)
    ml_b: Optional[int]
    closing_ml_a: Optional[int]   # closing line for CLV
    closing_ml_b: Optional[int]
    kelly_fraction: float         # computed from cal_prob_a or pred_prob_a
    stake: float                  # as fraction of bankroll
    outcome: int                  # 1 if team_a won, 0 if team_b won
```

---

## Architecture

```
TournamentBacktestOrchestrator        (owns I/O and year iteration)
  ├─ NcaabDataLoader                  (CSV + BartTorvik cache + team resolver)
  ├─ BartTorvik                       (historical AdjEM scraper)
  ├─ build_wp_table()                 (win_probability.py)
  ├─ simulate_tournament()            (simulation/tournament.py — optional, for bracket picks)
  ├─ CalibrationPipeline              (wraps Calibrator from model/calibration.py)
  └─ BacktestEngine                   (pure computation: List[GamePrediction] → YearResult)
        ├─ _brier_by_round()
        ├─ _accuracy_by_round()
        ├─ _kelly_series()
        ├─ _sharpe()
        └─ _max_drawdown()
```

**Key boundary:** `BacktestEngine.evaluate(games: List[GamePrediction]) -> YearResult` is a pure function with zero I/O. The Orchestrator owns all data loading, model calls, and calibration.

---

## Data Structures

```python
@dataclass
class YearResult:
    year: int
    n_games: int
    accuracy_by_round: Dict[str, float]     # R64..F
    brier_score: float                       # overall
    brier_by_round: Dict[str, float]         # per round
    upsets_predicted: int                    # lower seed predicted to win
    upsets_missed: int                       # lower seed won, model predicted chalk
    kelly_bets: List[GamePrediction]
    roi_pct: float
    sharpe: float
    max_drawdown_pct: float
    clv_mean: Optional[float]               # None if no odds data
    calibration_skipped: bool               # True if N_train < 150
    metadata: Dict                          # e.g., {"tempo_mode": "default_67.5"}

@dataclass
class WalkForwardResult:
    years: List[YearResult]
    year_errors: Dict[int, str]             # years that failed with error messages
    ensemble_accuracy: Dict[str, float]     # weighted average across all years
    calibration_lift: float                 # Brier improvement from calibration
    clv_summary: Optional[Dict]             # None if no odds data
    total_roi_pct: float
    total_sharpe: float
```

---

## Implementation Sequence

**Phase 1 — Data Foundation** (must complete before anything else)
1. `data/team_name_map.csv` — canonical↔torvik↔espn mappings (bootstrap from 2026, extend back)
2. `scripts/build_name_map.py` — fuzzy matching helper for populating the map
3. `src/utils/team_names.py` — `resolve(name)`, raises `UnresolvableTeamError` on miss
4. `scripts/update_tournament_cache.py` — ESPN → `data/tournament_results/{year}.csv`
5. Populate `data/tournament_results/` for 2016–2025 (commit to git, skip 2020)
6. `scripts/cache_torvik_history.py` — scrape BartTorvik 2016–2025 → `data/cache/torvik_{year}.json`

**Phase 2 — Data Loader** (after Phase 1)
7. `src/backtesting/data_loader.py` — `NcaabDataLoader` with file cache + team resolver
8. `tests/test_data_loader.py` — tests with --mock (fixture CSVs)

**Phase 3 — Pure Engine** (after Phase 2)
9. `src/backtesting/engine.py` — `BacktestEngine.evaluate()` + `TournamentBacktestOrchestrator`
10. `src/utils/calibration_pipeline.py` — cumulative calibration with N>=150 guard
11. `tests/test_backtest_engine.py` — TDD with mock data, verify all metric formulas

**Phase 4 — Integration**
12. Run end-to-end: `python scripts/run_backtest.py --start 2016 --end 2025 --mock`
13. Run real: `python scripts/run_backtest.py --start 2016 --end 2025`
14. Commit results summary to `.omc/backtest_results.json`

---

## Special Cases

- **2020:** No tournament. `load_tournament_results(2020)` returns `[]`. Orchestrator skips year 2020 automatically via `SKIP_YEARS = {2020}`. 2019 calibration data carries forward to 2021 (cumulative pool).
- **2016 test year:** Zero prior games → calibration cannot be fit. `calibration_skipped=True`, `cal_prob_a=None` for all games. Raw `pred_prob_a` used for Kelly sizing.
- **Calibration guard:** If `len(prior_games) < 150`, skip isotonic regression, log `calibration_skipped=True`.
- **First Four teams:** "NCST/SMU" and "HWD/LEH" have no individual BartTorvik rows. Map to composite AdjEM in `team_name_map.csv` or as special entries.

---

## Acceptance Criteria (testable)

- [ ] `BacktestEngine.evaluate([])` returns `YearResult` with all numeric fields = 0.0
- [ ] Brier score for perfect predictions (all `pred_prob=1.0`, all `outcome=1`) = 0.0
- [ ] Brier score for random predictions (all `pred_prob=0.5`) ≈ 0.25 ± 0.01
- [ ] `kelly_fraction` = 0.0 for all games where `edge < min_edge`
- [ ] `clv_summary` is `None` (not empty dict) when no moneylines provided
- [ ] `calibration_skipped=True` when training on < 150 games
- [ ] Year 2020 is absent from `WalkForwardResult.years`
- [ ] `UnresolvableTeamError` raised on unmapped team name, not silent None
- [ ] Historical BartTorvik data uses cache file if present, not network
- [ ] `run_walk_forward` returns partial results even if 1 year fails

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/backtesting/engine.py` | BacktestEngine + TournamentBacktestOrchestrator |
| `src/backtesting/data_loader.py` | NcaabDataLoader (CSV + BartTorvik cache) |
| `src/utils/__init__.py` | Package init |
| `src/utils/team_names.py` | Canonical resolver, UnresolvableTeamError |
| `scripts/build_name_map.py` | Fuzzy-match bootstrap for team_name_map.csv |
| `scripts/update_tournament_cache.py` | ESPN → tournament_results CSV |
| `scripts/cache_torvik_history.py` | BartTorvik historical JSON cache |
| `scripts/run_backtest.py` | CLI entry point |
| `data/team_name_map.csv` | Canonical name mappings (committed) |
| `data/tournament_results/` | Per-year game result CSVs (committed) |
| `data/cache/` | BartTorvik JSON cache (committed) |
| `tests/test_backtest_engine.py` | TDD: BacktestEngine + Orchestrator |
| `tests/test_data_loader.py` | TDD: NcaabDataLoader with mock fixtures |
| `tests/test_team_names.py` | TDD: canonical resolver |

---

*Planner + Architect + Critic consensus reached after 1 revision cycle.*
