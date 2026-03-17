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

    # Column positions in BartTorvik's raw JSON array
    # (verify each season — they occasionally shift)
    _COLS = {
        "rank":     0,
        "team":     1,
        "conf":     2,
        "record":   3,
        "AdjOE":    4,
        "AdjDE":    5,
        "AdjT":     12,
        "Barthag":  None,   # computed: win% vs avg D1
        "eFG_off":  6,
        "eFG_def":  7,
        "TO_off":   8,      # turnover rate (offense)
        "TO_def":   9,
        "OR_off":   10,     # offensive rebound rate
        "OR_def":   11,
        "P3_off":   None,   # 3pt rate off (column varies)
        "P3_def":   None,
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
            Dict keyed by team name -> stats dict:
            {
                "rank": int,
                "team": str,
                "conf": str,
                "record": str,
                "AdjOE": float,
                "AdjDE": float,
                "AdjEM": float,   # computed: AdjOE - AdjDE
                "AdjT": float,
                "eFG_off": float,
                "eFG_def": float,
                "TO_off": float,
                "TO_def": float,
                "OR_off": float,
                "OR_def": float,
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

                team_name = str(row[1]).strip()
                adj_oe = safe_float(row[4])
                adj_de = safe_float(row[5])
                adj_em = round(adj_oe - adj_de, 4) if adj_oe and adj_de else None
                adj_t  = safe_float(row[12]) if len(row) > 12 else None

                result[team_name] = {
                    "rank":    int(row[0]) if row[0] else None,
                    "team":    team_name,
                    "conf":    str(row[2]).strip() if len(row) > 2 else None,
                    "record":  str(row[3]).strip() if len(row) > 3 else None,
                    "AdjOE":   adj_oe,
                    "AdjDE":   adj_de,
                    "AdjEM":   adj_em,
                    "AdjT":    adj_t,
                    "eFG_off": safe_float(row[6])  if len(row) > 6  else None,
                    "eFG_def": safe_float(row[7])  if len(row) > 7  else None,
                    "TO_off":  safe_float(row[8])  if len(row) > 8  else None,
                    "TO_def":  safe_float(row[9])  if len(row) > 9  else None,
                    "OR_off":  safe_float(row[10]) if len(row) > 10 else None,
                    "OR_def":  safe_float(row[11]) if len(row) > 11 else None,
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
