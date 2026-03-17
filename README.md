# NCAAB Men's Basketball Prediction System

Walk-forward backtesting, Monte Carlo tournament simulation, and Kelly criterion bet sizing for **NCAA Men's Basketball**.

> **Scope:** Men's basketball only. Women's basketball (NCAAW) is a future expansion.

## Features

- **BartTorvik AdjEM** data ingestion — historical 2016-2025, scraper with verified column mapping
- **The Odds API** integration — Pinnacle closing lines, de-vig, ensemble model
- **Vectorized Monte Carlo** — 250k tournament simulations in ~3 seconds (numpy advanced indexing)
- **Walk-forward backtest** — train on prior years, test on current year, rolling calibration
- **Kelly criterion** + CLV tracking — half-Kelly sizing, Sharpe ratio, max drawdown
- **Parlay Kelly** — correlated tournament path bet sizing
- **56 TDD tests** — every betting formula verified before use

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add ODDS_API_KEY

# Run 2026 tournament simulation (mock — no API key needed):
python scripts/run_pipeline.py --year 2026 --mock

# Run with live Pinnacle lines:
python scripts/run_pipeline.py --year 2026

# Build historical results cache:
python scripts/update_tournament_cache.py --all

# Walk-forward backtest (mock):
python scripts/run_backtest.py --start 2016 --end 2025 --mock

# Run tests:
python -m pytest tests/ --timeout=30
```

## Architecture

```
src/
├── api/
│   ├── barttorvik.py       BartTorvik AdjEM scraper (col mapping verified 2026-03-17)
│   ├── espn_ncaab.py       ESPN free API client
│   └── odds_client.py      The Odds API — basketball_ncaab
├── model/
│   ├── win_probability.py  NormCDF ensemble model
│   └── calibration.py      Isotonic regression calibration
├── simulation/
│   └── tournament.py       Vectorized 250k Monte Carlo
├── betting/
│   ├── kelly.py            Kelly criterion + CLV
│   └── parlay.py           Correlated/parlay Kelly
├── backtesting/
│   ├── engine.py           BacktestEngine + TournamentBacktestOrchestrator
│   └── data_loader.py      NcaabDataLoader (file-cached, --mock support)
└── utils/
    └── team_names.py       Canonical name resolver (fails loudly on unknowns)

brackets/2026/
    simulate.py             2026 bracket with 250k simulation
    results_2026.json       Results: Duke 16.8%, Michigan 15.6%, Arizona 14.2%

data/
    team_name_map.csv       BartTorvik ↔ ESPN ↔ canonical name mapping
    tournament_results/     Per-year game CSVs (committed — run update_tournament_cache.py)
    cache/                  BartTorvik JSON cache per year (run cache_torvik_history.py)
    odds/                   Pinnacle lines per year (optional — enables CLV)

scripts/
    run_pipeline.py         Full prediction pipeline (data → signals)
    run_backtest.py         Walk-forward backtest CLI
    update_tournament_cache.py  Populate data/tournament_results/ from shoenot+ESPN
```

## Win Probability Model

```python
# Core formula
z = (AdjEM_a - AdjEM_b) * ((tempo_a + tempo_b) / 200) * 1.07 / sigma
P(a wins) = NormCDF(z)

# sigma = 12.5 for 3pt-heavy matchups (Duke, Michigan, Kansas, Alabama, Purdue...)
# sigma = 11.0 standard

# Ensemble (when Pinnacle line available):
P = 0.65 * P_pinnacle + 0.35 * P_torvik
```

**Theoretical accuracy ceiling:** ~74-75% (binomial shot noise — academic consensus).
Pure BartTorvik: ~71-72%. With Pinnacle closing lines: ~73-74%.

## 2026 Tournament Results (250k simulations)

| Team | Region | Champion % |
|------|--------|-----------|
| Duke | East | 16.8% |
| Michigan | Midwest | 15.6% |
| Arizona | West | 14.2% |
| Florida | South | 8.9% |
| Purdue | West | 8.4% |
| Illinois | South | 8.3% |
| Houston | South | 6.4% |
| Iowa St | Midwest | 5.2% |

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ODDS_API_KEY` | For live lines | The Odds API — basketball_ncaab |
| `ANTHROPIC_API_KEY` | Optional | Future: injury parsing |

## Data Sources

| Source | What | Notes |
|--------|------|-------|
| [BartTorvik](https://barttorvik.com) | AdjOE/DE/T/Barthag | Free, JSON feed |
| [The Odds API](https://the-odds-api.com) | Pinnacle lines | Paid, 500 free/mo |
| [ESPN](https://site.api.espn.com) | Scores, seeds, results | Free |
| [shoenot/march-madness-games-csv](https://github.com/shoenot/march-madness-games-csv) | Historical tournament results 1985-2024 | Free, GitHub |

## License

Private project — EAGLE605. Not for redistribution.
