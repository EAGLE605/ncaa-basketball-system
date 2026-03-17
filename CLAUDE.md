# CLAUDE.md — ncaa-basketball-system

## IMPORTANT: NCAA Men's Basketball Only
This system is for **NCAA Men's Basketball (NCAAB Men's)** only.
- BartTorvik sport key: `basketball_ncaab` (Men's)
- The Odds API sport key: `basketball_ncaab` (Men's)
- Women's basketball (NCAAW) is a future expansion — P4 in Beads backlog, NOT in scope now

## Project Purpose
Long-term NCAA Men's Basketball prediction and betting system:
1. 2026 March Madness bracket simulation (immediate)
2. Walk-forward backtesting 2016-2025 (P2)
3. Regular season prediction pipeline (P2)
4. Kelly criterion bet sizing with CLV tracking
5. Automated betting signals via ntfy

## Data Sources
| Source | What | Key |
|--------|------|-----|
| BartTorvik `trank.php?year=YYYY&json=1` | AdjOE/DE/T/Barthag | No key |
| The Odds API `basketball_ncaab` | Pinnacle market lines | `ODDS_API_KEY` |
| ESPN `site.api.espn.com` | Scores, results, seeds | No key |
| `shoenot/march-madness-games-csv` | Historical tournament results 1985-2024 | GitHub |

## BartTorvik Column Mapping (CRITICAL — verified 2026-03-17)
The JSON array uses these positions (37 cols, stable 2016-2025):
- `col[0]` = team name (NOT rank — rows are in internal DB order, NOT sorted)
- `col[1]` = AdjOE, `col[2]` = AdjDE
- `col[3]` = Barthag, `col[4]` = record
- `col[15]` = AdjT (actual tempo, possessions/40 min, range ~60-75)
- `AdjEM = col[1] - col[2]` (computed, not a direct column)

**DO NOT** use col[12] as AdjT — that's DRB (defensive rebound rate).

## Win Probability Model
```
z = (AdjEM_a - AdjEM_b) * ((tempo_a + tempo_b) / 200) * 1.07 / sigma
P(a wins) = NormCDF(z)
sigma = 12.5 (3pt-heavy matchup) or 11.0 (standard)
Ensemble: 0.65 * P_market + 0.35 * P_torvik (when Pinnacle line available)
```

## Testing
```bash
# All tests (56 green):
python -m pytest tests/ --timeout=30

# TDD gate — run before any betting-critical change:
python -m pytest tests/test_win_probability.py tests/test_kelly.py -v
```
All tests use `--mock` internally — no paid API calls.

## Key Commands
```bash
# Run 2026 tournament simulation (mock):
python scripts/run_pipeline.py --year 2026 --mock

# Run 2026 with live Pinnacle lines:
python scripts/run_pipeline.py --year 2026

# Build tournament results cache:
python scripts/update_tournament_cache.py --all

# Run walk-forward backtest (mock):
python scripts/run_backtest.py --start 2016 --end 2025 --mock
```

## Environment
- `.env` in project root with `ODDS_API_KEY` — never commit
- `src/config/secrets.py` loads from `.env` via python-dotenv

## Architecture
```
src/api/          — BartTorvik, ESPN, The Odds API clients
src/model/        — win_probability, calibration
src/simulation/   — vectorized 250k Monte Carlo
src/betting/      — kelly, parlay
src/backtesting/  — engine, data_loader
src/utils/        — team_names canonical resolver
brackets/2026/    — 2026 tournament bracket + results
data/             — tournament_results/, cache/, odds/
scripts/          — run_pipeline, run_backtest, update_tournament_cache
```

## Beads Task Tracker
Run `~/.claude/beads/bd.exe list` from project root to see open tasks.
