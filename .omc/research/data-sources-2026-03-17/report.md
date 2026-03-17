# Research Report: NCAAB Data Sources
**Session:** data-sources-2026-03-17
**Date:** 2026-03-17
**Status:** COMPLETE

---

## Finding 1: BartTorvik Column Mapping Was Wrong

**Confidence: HIGH** — verified live against 2016, 2020, 2025 data.

The original `barttorvik.py` column assumptions were all wrong. Correct mapping:

| Col | Field | Notes |
|-----|-------|-------|
| 0 | team name | NOT rank — rows in internal DB order |
| 1 | AdjOE | pts/100 poss offense |
| 2 | AdjDE | pts/100 poss defense |
| 3 | Barthag | power rating |
| 4 | record | "35-4" string |
| 15 | **AdjT** | possessions/40 min (~60-75) — col[12] was DRB, not tempo |
| 18 | 3P% off | three-point % offense |
| 19 | 3P% def | three-point % defense |
| 30 | year | season year int |
| 34 | WAB | wins above bubble |

- **AdjEM is not a column** — compute as `col[1] - col[2]`
- **Schema is stable at 37 columns** across all years 2016-2025
- **Rows NOT sorted by rank** — sort client-side
- **2020 data present** — regular season completed before March 11 cancellation
- **All years 2016-2025 confirmed live** from `trank.php?year=YYYY&json=1`

**Action taken:** `src/api/barttorvik.py` column mapping corrected.

---

## Finding 2: Tournament Results Sources

**Confidence: HIGH** — both primary sources verified with live fetches.

### Recommended: Hybrid approach

**Primary (2016-2024):** `shoenot/march-madness-games-csv` GitHub repo
- URL: `https://raw.githubusercontent.com/shoenot/march-madness-games-csv/main/csv/combined.csv`
- 63 games/year, `round_of` numeric (64/32/16/8/4/2), seeds, scores
- Covers 1985-2024, skips 2020 automatically
- Missing: region column, 2025

**Supplement (2025 + regions):** ESPN free scoreboard API
- `https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates=YYYYMMDD&groups=100&limit=50`
- Returns seeds, scores, region in `notes[0].headline` ("West Region, 1st Round")
- ~8 API calls/year to cover all tournament dates
- Verified working for 2016 and 2025

**Round label mapping:** 64→R64, 32→R32, 16→S16, 8→E8, 4→FF, 2→F

### Sources ranked

| Rank | Source | Years | Free | Action |
|------|--------|-------|------|--------|
| 1 | shoenot/march-madness-games-csv | 1985-2024 (no 2020) | Yes | Use for 2016-2024 |
| 2 | ESPN scoreboard API | 2016-present | Yes | Use for 2025 + regions |
| 3 | danvk/march-madness-data | 1985-2017 | Yes | Cross-reference only |
| 4 | Kaggle nishaanamin | 2008-2025 | Yes (login) | Skip — no game scores |
| 5 | Sports-Reference CBB | All years | Bot-blocked | Avoid |
| 6 | Sportradar | All years | No ($$$) | Skip |

**Implementation target:** `scripts/update_tournament_cache.py` — download shoenot CSV, patch 2025 + regions from ESPN, write to `data/tournament_results/{year}.csv`.
