"""
ESPN API client for NCAA Men's Basketball.
Free, no API key required.

Adapted from EAGLE605/nfl-betting-system espn_client.py
Sport: basketball-college-basketball
"""

import logging
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
SITE_URL = "https://site.api.espn.com/apis"


class ESPNNcaabClient:
    """
    ESPN free API client for NCAA Men's Basketball.

    No authentication required. Rate limit: ~100 req/min.
    Circuit breaker pattern: 3 retries with exponential backoff.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._consecutive_errors = 0
        self._circuit_open = False

    def _get(self, url: str, params: dict = None) -> dict:
        if self._circuit_open:
            raise RuntimeError("ESPN circuit breaker open — too many consecutive errors")

        for attempt in range(3):
            try:
                r = self.session.get(url, params=params or {}, timeout=10)
                r.raise_for_status()
                self._consecutive_errors = 0
                if self._consecutive_errors > 5:
                    self._circuit_open = False
                return r.json()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else 0
                if status == 404:
                    logger.warning(f"ESPN 404: {url}")
                    return {}
                wait = 2 ** attempt
                logger.warning(f"ESPN HTTP error (attempt {attempt+1}), retry in {wait}s: {e}")
                time.sleep(wait)
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                logger.warning(f"ESPN request failed (attempt {attempt+1}): {e}")
                time.sleep(wait)

        self._consecutive_errors += 1
        if self._consecutive_errors >= 5:
            self._circuit_open = True
            logger.error("ESPN circuit breaker opened after 5 consecutive failures")
        raise RuntimeError(f"ESPN API failed after 3 attempts: {url}")

    # ------------------------------------------------------------------
    # SCOREBOARD / SCHEDULE
    # ------------------------------------------------------------------

    def get_scoreboard(self, date: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get today's (or a specific date's) scoreboard.

        Args:
            date:  YYYYMMDD format, e.g. "20260320". None = today.
            limit: Max games to return.

        Returns:
            List of event dicts from ESPN events[] array.
        """
        params = {"limit": limit}
        if date:
            params["dates"] = date
        data = self._get(f"{BASE_URL}/scoreboard", params)
        return data.get("events", [])

    def get_game(self, game_id: str) -> Dict:
        """Full game details including boxscore."""
        return self._get(f"{BASE_URL}/summary", {"event": game_id})

    # ------------------------------------------------------------------
    # TEAMS
    # ------------------------------------------------------------------

    def get_team(self, team_id: str) -> Dict:
        """Team metadata by ESPN team ID."""
        return self._get(f"{BASE_URL}/teams/{team_id}")

    def search_team(self, name: str) -> Optional[Dict]:
        """
        Find a team by approximate name match.
        Returns first result or None.
        """
        data = self._get(f"{BASE_URL}/teams", {"limit": 500})
        teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        name_lower = name.lower()
        for entry in teams:
            team = entry.get("team", {})
            if (
                name_lower in team.get("displayName", "").lower()
                or name_lower in team.get("shortDisplayName", "").lower()
                or name_lower in team.get("abbreviation", "").lower()
            ):
                return team
        return None

    # ------------------------------------------------------------------
    # RANKINGS
    # ------------------------------------------------------------------

    def get_rankings(self) -> List[Dict]:
        """Current AP Top 25 and coaches poll."""
        data = self._get(f"{BASE_URL}/rankings")
        return data.get("rankings", [])

    # ------------------------------------------------------------------
    # TOURNAMENT
    # ------------------------------------------------------------------

    def get_tournament_bracket(self) -> Dict:
        """
        NCAA Tournament bracket structure from ESPN.
        Returns full bracket JSON including seeds, regions, results.
        """
        return self._get(f"{BASE_URL}/tournaments/22", {"limit": 100})

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def parse_game_result(self, event: Dict) -> Optional[Dict]:
        """
        Extract structured result from an ESPN event dict.

        Returns:
            {
                "game_id": str,
                "date": str (ISO),
                "status": "final" | "in_progress" | "scheduled",
                "home_team": str,
                "away_team": str,
                "home_score": int | None,
                "away_score": int | None,
                "neutral_site": bool,
            }
        """
        try:
            comp = event["competitions"][0]
            competitors = {c["homeAway"]: c for c in comp["competitors"]}
            home = competitors.get("home", {})
            away = competitors.get("away", {})
            status = event.get("status", {}).get("type", {}).get("name", "").lower()
            return {
                "game_id":    event.get("id"),
                "date":       event.get("date"),
                "status":     "final" if "final" in status else ("in_progress" if "progress" in status else "scheduled"),
                "home_team":  home.get("team", {}).get("displayName"),
                "away_team":  away.get("team", {}).get("displayName"),
                "home_score": int(home["score"]) if home.get("score") else None,
                "away_score": int(away["score"]) if away.get("score") else None,
                "neutral_site": comp.get("neutralSite", False),
            }
        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"parse_game_result error: {e}")
            return None

    def get_recent_results(self, days: int = 7) -> List[Dict]:
        """
        Get parsed results for the last N days.
        Returns list of parse_game_result dicts (status == 'final' only).
        """
        from datetime import datetime, timedelta
        results = []
        for d in range(days):
            date = (datetime.utcnow() - timedelta(days=d)).strftime("%Y%m%d")
            events = self.get_scoreboard(date=date, limit=200)
            for ev in events:
                parsed = self.parse_game_result(ev)
                if parsed and parsed["status"] == "final":
                    results.append(parsed)
        return results
