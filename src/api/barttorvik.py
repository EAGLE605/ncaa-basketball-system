"""
BartTorvik scraper for NCAA Men's Basketball adjusted efficiency metrics.

Scrapes T-Rank (BartTorvik) AdjOE, AdjDE, AdjEM, AdjT, Barthag for any season.
Data URL: https://www.barttorvik.com/trank.php?year=YYYY&json=1

No API key required. Scrapes JSON feed directly.
"""

import logging
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

BASE = "https://www.barttorvik.com"


def _trank_url(year: int) -> str:
    return f"{BASE}/trank.php?year={year}&json=1"


class BartTorvik:
    """
    BartTorvik T-Rank data client.

    Fetches AdjOE, AdjDE, AdjEM (= AdjOE - AdjDE), AdjT, Barthag,
    eFG%, TO%, OR%, FTR (Four Factors) for every D1 team in a season.

    Usage:
        bt = BartTorvik()
        teams = bt.get_season(2026)  # {team_name: stats_dict}
        duke = teams["Duke"]         # {"AdjOE": 122.5, "AdjDE": 89.7, ...}
    """

    # Column positions in BartTorvik's raw JSON array.
    # Verified live against 2016-2025 data (2026-03-17).
    # Schema is STABLE at 37 columns across all years 2016-2025.
    # IMPORTANT:
    #   - col[0] is TEAM NAME (string), NOT rank.
    #   - Rows are served in internal DB order, NOT sorted by AdjEM.
    #     Sort client-side if you need rank order.
    #   - AdjEM is NOT a direct column — compute as AdjOE - AdjDE.
    #   - col[15] is actual AdjT (possessions/40 min, range ~60-75).
    #     The original assumption of col[12]=AdjT was WRONG (col[12]=DRB).
    #   - col[3] = Barthag (power rating = win prob vs avg D1 team).
    _COLS = {
        "team":     0,      # Team name string
        "AdjOE":    1,      # Adjusted Offensive Efficiency (pts/100 poss)
        "AdjDE":    2,      # Adjusted Defensive Efficiency (pts/100 poss allowed)
        "Barthag":  3,      # Power rating (win prob vs avg D1)
        "record":   4,      # Record string e.g. "35-4"
        "wins":     5,      # Wins (int)
        "games":    6,      # Games played (int)
        "eFG_off":  7,      # Effective FG% offense
        "eFG_def":  8,      # Effective FG% defense
        "TO_off":   9,      # Turnover rate offense
        "TO_def":   10,     # Turnover rate defense
        "OR_off":   11,     # Offensive rebound rate
        "OR_def":   12,     # Defensive rebound rate (NOT AdjT — old bug)
        "FTR_off":  13,     # Free throw rate offense
        "FTR_def":  14,     # Free throw rate defense
        "AdjT":     15,     # Adjusted Tempo (possessions/40 min) — CORRECT column
        "2P_off":   16,     # Two-point % offense
        "2P_def":   17,     # Two-point % defense
        "3P_off":   18,     # Three-point % offense
        "3P_def":   19,     # Three-point % defense
        "3PR_off":  20,     # Three-point attempt rate offense
        "3PR_def":  21,     # Three-point attempt rate defense
        # cols 22-29: duplicates/rank fields, not used
        "year":     30,     # Season year (int)
        # cols 31-33: empty strings
        "WAB":      34,     # Wins Above Bubble
        # col 35: unknown metric; col 36: null
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (research)"})
        self._cache: Dict[int, List] = {}

    def _fetch_raw(self, year: int) -> List:
        """Fetch raw JSON array from T-Rank."""
        if year in self._cache:
            return self._cache[year]

        url = _trank_url(year)
        for attempt in range(3):
            try:
                r = self.session.get(url, timeout=20)
                r.raise_for_status()
                data = r.json()
                self._cache[year] = data
                logger.info(f"BartTorvik {year}: {len(data)} teams fetched")
                return data
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"BartTorvik fetch attempt {attempt+1} failed: {e}. Retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"BartTorvik failed to fetch {year} data after 3 attempts")

    def get_season(self, year: int) -> Dict[str, Dict]:
        """
        Get all team stats for a season.

        Args:
            year: Season year (e.g. 2026 = 2025-26 season).

        Returns:
            Dict keyed by team name -> stats dict. Teams are in internal DB
            order (NOT sorted by AdjEM) — sort client-side if needed.
            {
                "team": str,
                "AdjOE": float,   # Adjusted Offensive Efficiency (pts/100 poss)
                "AdjDE": float,   # Adjusted Defensive Efficiency
                "AdjEM": float,   # Computed: AdjOE - AdjDE
                "Barthag": float, # Power rating (win prob vs avg D1)
                "record": str,    # e.g. "35-4"
                "AdjT": float,    # Tempo (possessions/40 min, range ~60-75)
                "eFG_off/def": float, "TO_off/def": float,
                "OR_off/def": float, "3P_off/def": float, "WAB": float,
            }
        """
        raw = self._fetch_raw(year)
        result = {}
        for row in raw:
            try:
                # BartTorvik JSON is array-of-arrays
                if not isinstance(row, (list, tuple)):
                    continue
                def safe_float(val) -> Optional[float]:
                    try:
                        return float(val) if val not in (None, "", "N/A") else None
                    except (TypeError, ValueError):
                        return None

                # Verified column mapping (2016-2025 stable, 37 cols):
                # col[0]=team, col[1]=AdjOE, col[2]=AdjDE, col[3]=Barthag,
                # col[4]=record, col[15]=AdjT (tempo), col[30]=year
                team_name = str(row[0]).strip()
                if not team_name:
                    continue
                adj_oe = safe_float(row[1])
                adj_de = safe_float(row[2])
                adj_em = round(adj_oe - adj_de, 4) if adj_oe and adj_de else None
                adj_t  = safe_float(row[15]) if len(row) > 15 else None

                result[team_name] = {
                    "team":    team_name,
                    "AdjOE":   adj_oe,
                    "AdjDE":   adj_de,
                    "AdjEM":   adj_em,
                    "Barthag": safe_float(row[3]) if len(row) > 3 else None,
                    "record":  str(row[4]).strip() if len(row) > 4 else None,
                    "AdjT":    adj_t,
                    "eFG_off": safe_float(row[7])  if len(row) > 7  else None,
                    "eFG_def": safe_float(row[8])  if len(row) > 8  else None,
                    "TO_off":  safe_float(row[9])  if len(row) > 9  else None,
                    "TO_def":  safe_float(row[10]) if len(row) > 10 else None,
                    "OR_off":  safe_float(row[11]) if len(row) > 11 else None,
                    "OR_def":  safe_float(row[12]) if len(row) > 12 else None,
                    "3P_off":  safe_float(row[18]) if len(row) > 18 else None,
                    "3P_def":  safe_float(row[19]) if len(row) > 19 else None,
                    "WAB":     safe_float(row[34]) if len(row) > 34 else None,
                }
            except (IndexError, TypeError) as e:
                logger.debug(f"BartTorvik row parse error: {e} — row={row[:5]}")
        return result

    def get_team(self, year: int, team: str) -> Optional[Dict]:
        """Get stats for a single team. Returns None if not found."""
        season = self.get_season(year)
        return season.get(team)

    def get_tournament_field(
        self, year: int, teams: List[str]
    ) -> Dict[str, Dict]:
        """
        Get stats for a specific list of teams (tournament field).
        Teams not found in BartTorvik get None.
        """
        season = self.get_season(year)
        return {t: season.get(t) for t in teams}

    def adj_em(self, year: int, team: str) -> Optional[float]:
        """Quick accessor for a single team's AdjEM."""
        t = self.get_team(year, team)
        return t["AdjEM"] if t else None

    def get_historical_seasons(
        self, start_year: int, end_year: int
    ) -> Dict[int, Dict[str, Dict]]:
        """
        Fetch multiple seasons. Useful for building training datasets.

        Args:
            start_year: e.g. 2016
            end_year:   e.g. 2025 (inclusive)

        Returns:
            {year: {team: stats}}
        """
        result = {}
        for year in range(start_year, end_year + 1):
            try:
                result[year] = self.get_season(year)
                time.sleep(0.5)  # polite crawl delay
            except RuntimeError as e:
                logger.error(f"Failed to fetch {year}: {e}")
        return result
