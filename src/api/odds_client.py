"""
The Odds API client for NCAA Men's Basketball.
Reuses the same key as the NFL betting system (ODDS_API_KEY).
API docs: https://the-odds-api.com/liveapi/guides/v4/

Sport key: basketball_ncaab
Markets:   h2h (moneyline), spreads, totals
Regions:   us (DraftKings, FanDuel, Pinnacle, BetMGM, etc.)
"""

import logging
import time
from typing import Dict, List, Optional

import requests

from src.config.secrets import get as get_secret

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT    = "basketball_ncaab"


class OddsClient:
    """
    The Odds API client for NCAAB.

    Same API key as NFL system — sport key changes to 'basketball_ncaab'.
    Rate limits: 500 req/month (free), 10k/month (basic $5), unlimited (higher tiers).
    Each call returns X-Requests-Remaining in response headers.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_secret("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ODDS_API_KEY not configured. Add to .env or environment variables.\n"
                "Get key at: https://the-odds-api.com/"
            )
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._requests_remaining = None

    def _get(self, endpoint: str, params: dict = None) -> dict | list:
        url    = f"{BASE_URL}/{endpoint}"
        params = {**(params or {}), "apiKey": self.api_key}

        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=15)
                # Track quota
                remaining = r.headers.get("x-requests-remaining")
                if remaining:
                    self._requests_remaining = int(remaining)
                    logger.debug(f"Odds API requests remaining: {remaining}")
                r.raise_for_status()
                return r.json()
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 422:
                    logger.error(f"Odds API 422 — bad params: {params}")
                    raise
                wait = 2 ** attempt
                logger.warning(f"Odds API error (attempt {attempt+1}), retry in {wait}s: {e}")
                time.sleep(wait)
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                logger.warning(f"Odds API request failed (attempt {attempt+1}): {e}")
                time.sleep(wait)

        raise RuntimeError(f"Odds API failed after 3 attempts: {endpoint}")

    # ------------------------------------------------------------------
    # LIVE / UPCOMING ODDS
    # ------------------------------------------------------------------

    def get_tournament_odds(
        self,
        markets: str = "h2h,spreads",
        regions: str = "us",
        odds_format: str = "american",
        bookmakers: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get current NCAAB odds (includes tournament games when live).

        Args:
            markets:     Comma-separated: h2h, spreads, totals
            regions:     us, uk, eu, au
            odds_format: american or decimal
            bookmakers:  Comma-separated list to filter (e.g. "pinnacle,draftkings")
                         None = all available bookmakers

        Returns:
            List of game odds dicts
        """
        params = {
            "regions":     regions,
            "markets":     markets,
            "oddsFormat":  odds_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers

        data = self._get(f"sports/{SPORT}/odds", params)
        logger.info(f"Fetched odds for {len(data)} NCAAB games")
        return data

    def get_pinnacle_lines(self) -> List[Dict]:
        """Convenience: Pinnacle lines only (most accurate closing lines)."""
        return self.get_tournament_odds(bookmakers="pinnacle")

    # ------------------------------------------------------------------
    # HISTORICAL ODDS (requires paid tier)
    # ------------------------------------------------------------------

    def get_historical_odds(
        self,
        date: str,
        markets: str = "h2h,spreads",
        regions: str = "us",
        odds_format: str = "american",
    ) -> List[Dict]:
        """
        Historical odds snapshot at a specific date/time.

        Args:
            date:    ISO 8601 UTC, e.g. "2025-03-21T12:00:00Z"
            markets: h2h, spreads, totals
            regions: us, uk, eu, au
        """
        params = {
            "date":       date,
            "regions":    regions,
            "markets":    markets,
            "oddsFormat": odds_format,
        }
        data = self._get(f"sports/{SPORT}/odds-history", params)
        return data.get("data", [])

    # ------------------------------------------------------------------
    # SCORES / RESULTS
    # ------------------------------------------------------------------

    def get_scores(self, days_from: int = 3) -> List[Dict]:
        """
        Recent and upcoming scores.

        Args:
            days_from: How many days of completed games to include
        """
        params = {"daysFrom": days_from}
        return self._get(f"sports/{SPORT}/scores", params)

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def devig_moneyline(self, ml_home: int, ml_away: int) -> tuple[float, float]:
        """
        Convert American moneylines to de-vigged win probabilities.

        Formula (multiplicative method):
            raw_p = |ml|/(|ml|+100)  if favorite (ml < 0)
            raw_p = 100/(ml+100)     if underdog (ml > 0)
            p_devig = raw_p / (raw_home + raw_away)

        Returns:
            (p_home, p_away) — sum to 1.0
        """
        def raw(ml: int) -> float:
            return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)

        rh, ra = raw(ml_home), raw(ml_away)
        total  = rh + ra
        return rh / total, ra / total

    def extract_pinnacle_probs(self, odds_data: List[Dict]) -> Dict[str, Dict]:
        """
        Extract de-vigged Pinnacle moneyline probabilities from odds data.

        Returns:
            Dict keyed by "home_team vs away_team" ->
                {"home": p_home, "away": p_away, "home_ml": int, "away_ml": int}
        """
        result = {}
        for game in odds_data:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            for bm in game.get("bookmakers", []):
                if bm["key"] != "pinnacle":
                    continue
                for market in bm.get("markets", []):
                    if market["type"] != "h2h":
                        continue
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    ml_home = outcomes.get(home)
                    ml_away = outcomes.get(away)
                    if ml_home and ml_away:
                        ph, pa = self.devig_moneyline(ml_home, ml_away)
                        key = f"{home} vs {away}"
                        result[key] = {
                            "home": home, "away": away,
                            "home_ml": ml_home, "away_ml": ml_away,
                            "p_home": round(ph, 4), "p_away": round(pa, 4),
                        }
        return result

    @property
    def requests_remaining(self) -> Optional[int]:
        return self._requests_remaining
