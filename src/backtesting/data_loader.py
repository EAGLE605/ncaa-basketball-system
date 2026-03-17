"""
Data loader for NCAAB Men's Basketball backtesting.

Loads per-year tournament game results, BartTorvik efficiency data,
and optional historical market lines. All data is file-cached — no
live API calls during backtesting runs.

Cache strategy:
  - Tournament results: data/tournament_results/{year}.csv (committed to git)
  - BartTorvik:         data/cache/torvik_{year}.json (committed to git)
  - Odds:               data/odds/pinnacle_{year}.json (optional, not committed)

Run scripts/update_tournament_cache.py and scripts/cache_torvik_history.py
to populate the caches before running backtests.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Project root
_ROOT = Path(__file__).parent.parent.parent

TOURNAMENT_RESULTS_DIR = _ROOT / "data" / "tournament_results"
TORVIK_CACHE_DIR       = _ROOT / "data" / "cache"
ODDS_DIR               = _ROOT / "data" / "odds"


class NcaabDataLoader:
    """
    Loads data needed for tournament backtesting from local file caches.

    All methods accept mock=True to return minimal fixture data without
    reading real files (for use in test suites).
    """

    def __init__(self, data_dir: Optional[str] = None, force_refresh: bool = False):
        self._root = Path(data_dir) if data_dir else _ROOT
        self.force_refresh = force_refresh
        self._torvik_cache: Dict[int, Dict] = {}

    def load_tournament_results(self, year: int, mock: bool = False) -> List[Dict]:
        """
        Load tournament game results for a year from pre-cached CSV.

        Returns list of dicts:
            {round, region, team_a, team_b, seed_a, seed_b, winner,
             score_a, score_b}

        Round labels: R64 / R32 / S16 / E8 / FF / F
        """
        if mock:
            return _mock_tournament_results(year)

        path = self._root / "data" / "tournament_results" / f"{year}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Tournament results not found: {path}\n"
                f"Run: python scripts/update_tournament_cache.py --year {year}"
            )

        results = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({
                    "round":    row.get("round", "").strip(),
                    "region":   row.get("region", "").strip(),
                    "team_a":   row.get("team_a", "").strip(),
                    "team_b":   row.get("team_b", "").strip(),
                    "seed_a":   int(row.get("seed_a") or 0),
                    "seed_b":   int(row.get("seed_b") or 0),
                    "winner":   row.get("winner", "").strip(),
                    "score_a":  int(row.get("score_a") or 0),
                    "score_b":  int(row.get("score_b") or 0),
                })
        logger.info(f"Loaded {len(results)} tournament games for {year}")
        return results

    def load_torvik_season(self, year: int, mock: bool = False) -> Dict[str, Dict]:
        """
        Load BartTorvik AdjEM data for a season.

        Checks cache at data/cache/torvik_{year}.json first.
        Falls back to live scrape if cache missing and force_refresh=True.
        """
        if mock:
            return _mock_torvik_data()

        if year in self._torvik_cache and not self.force_refresh:
            return self._torvik_cache[year]

        cache_path = self._root / "data" / "cache" / f"torvik_{year}.json"

        if cache_path.exists() and not self.force_refresh:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            self._torvik_cache[year] = data
            logger.info(f"Loaded Torvik {year} from cache ({len(data)} teams)")
            return data

        # Live scrape fallback
        logger.info(f"Torvik {year} not cached — scraping live")
        from src.api.barttorvik import BartTorvik
        bt = BartTorvik()
        data = bt.get_season(year)

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info(f"Cached Torvik {year} to {cache_path}")

        self._torvik_cache[year] = data
        return data

    def load_historical_odds(self, year: int, mock: bool = False) -> Optional[Dict]:
        """
        Load historical Pinnacle market lines for tournament year.
        Returns None if not available (CLV metrics will be skipped).
        """
        if mock:
            return None  # No mock odds — CLV tests use None path

        path = self._root / "data" / "odds" / f"pinnacle_{year}.json"
        if not path.exists():
            logger.debug(f"No historical odds for {year} — CLV will be skipped")
            return None

        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def build_bracket(self, year: int, mock: bool = False) -> Dict[str, List[str]]:
        """
        Build bracket dict {region: [team_name x 16]} in seed order.

        Derives from tournament_results CSV: finds R64 games, groups by region,
        orders teams as [1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]-seed.
        """
        if mock:
            return _mock_bracket()

        results = self.load_tournament_results(year)
        r64 = [r for r in results if r["round"] == "R64"]

        regions: Dict[str, Dict[int, str]] = {}
        for game in r64:
            region = game.get("region", "Unknown")
            if region not in regions:
                regions[region] = {}
            # Map seed -> team
            regions[region][game["seed_a"]] = game["team_a"]
            regions[region][game["seed_b"]] = game["team_b"]

        # Seed order per region slot
        SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        bracket = {}
        for region, seed_map in regions.items():
            bracket[region] = [seed_map.get(s, f"UNKNOWN_{s}") for s in SEED_ORDER]

        return bracket


# ---------------------------------------------------------------------------
# Mock fixtures (--mock mode / test suite)
# ---------------------------------------------------------------------------

def _mock_torvik_data() -> Dict[str, Dict]:
    return {
        "Duke":     {"AdjEM": 36.69, "AdjT": 67.5, "AdjOE": 122.35, "AdjDE": 85.66},
        "Michigan": {"AdjEM": 36.61, "AdjT": 65.2, "AdjOE": 120.27, "AdjDE": 83.66},
        "Arizona":  {"AdjEM": 35.54, "AdjT": 69.1, "AdjOE": 119.12, "AdjDE": 83.58},
        "Florida":  {"AdjEM": 33.82, "AdjT": 68.3, "AdjOE": 115.60, "AdjDE": 81.78},
        "seed16E":  {"AdjEM": -8.0,  "AdjT": 67.5, "AdjOE": 95.0,   "AdjDE": 103.0},
        "seed16W":  {"AdjEM": -8.0,  "AdjT": 67.5, "AdjOE": 95.0,   "AdjDE": 103.0},
        "seed16M":  {"AdjEM": -8.0,  "AdjT": 67.5, "AdjOE": 95.0,   "AdjDE": 103.0},
        "seed16S":  {"AdjEM": -8.0,  "AdjT": 67.5, "AdjOE": 95.0,   "AdjDE": 103.0},
    }


def _mock_tournament_results(year: int) -> List[Dict]:
    """Minimal 4-game fixture — Duke beats seed16E, others chalk."""
    return [
        {"round": "R64", "region": "East", "team_a": "Duke", "team_b": "seed16E",
         "seed_a": 1, "seed_b": 16, "winner": "Duke", "score_a": 93, "score_b": 55},
        {"round": "R64", "region": "West", "team_a": "Arizona", "team_b": "seed16W",
         "seed_a": 1, "seed_b": 16, "winner": "Arizona", "score_a": 88, "score_b": 62},
        {"round": "R64", "region": "Midwest", "team_a": "Michigan", "team_b": "seed16M",
         "seed_a": 1, "seed_b": 16, "winner": "Michigan", "score_a": 85, "score_b": 60},
        {"round": "R64", "region": "South", "team_a": "Florida", "team_b": "seed16S",
         "seed_a": 1, "seed_b": 16, "winner": "Florida", "score_a": 90, "score_b": 58},
    ]


def _mock_bracket() -> Dict[str, List[str]]:
    seed16 = ["seed16", "seed8", "seed9", "seed5", "seed12", "seed4", "seed13",
              "seed6", "seed11", "seed3", "seed14", "seed7", "seed10", "seed2", "seed15"]
    return {
        "East":    ["Duke"]    + [f"s{i}E" for i in seed16],
        "West":    ["Arizona"] + [f"s{i}W" for i in seed16],
        "Midwest": ["Michigan"]+ [f"s{i}M" for i in seed16],
        "South":   ["Florida"] + [f"s{i}S" for i in seed16],
    }
