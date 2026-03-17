"""
Populate data/tournament_results/{year}.csv from external sources.

Strategy (per research 2026-03-17):
  1. Download shoenot/march-madness-games-csv for 2016-2024
  2. Fetch 2025 via ESPN scoreboard API
  3. Add region column from ESPN for all years
  4. Write per-year CSV files to data/tournament_results/

Usage:
    python scripts/update_tournament_cache.py --year 2025
    python scripts/update_tournament_cache.py --all
    python scripts/update_tournament_cache.py --all --mock
"""

import argparse
import csv
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("cache_builder")

# GitHub raw CSV — covers 1985-2024, skips 2020 automatically
SHOENOT_COMBINED = (
    "https://raw.githubusercontent.com/shoenot/march-madness-games-csv/main/csv/combined.csv"
)
# ESPN scoreboard API
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard?dates={date}&groups=100&limit=50"
)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "tournament_results"

# Tournament date ranges per year (March Madness First Four + R64 through Championship)
TOURNAMENT_DATES: Dict[int, List[str]] = {
    2025: ["20250318", "20250319", "20250320", "20250321", "20250322",
           "20250323", "20250327", "20250328", "20250329", "20250330",
           "20250405", "20250407"],
    2024: ["20240319", "20240320", "20240321", "20240322", "20240323",
           "20240324", "20240328", "20240329", "20240330", "20240331",
           "20240406", "20240408"],
    2023: ["20230315", "20230316", "20230317", "20230318", "20230319",
           "20230323", "20230324", "20230325", "20230326",
           "20230401", "20230403"],
}

# Round-of numeric to label
ROUND_MAP = {64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "FF", 2: "F"}


def download_shoenot() -> List[Dict]:
    """Download shoenot combined CSV and parse into list of dicts."""
    logger.info("Downloading shoenot/march-madness-games-csv combined.csv...")
    r = requests.get(SHOENOT_COMBINED, timeout=30)
    r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text))
    rows = list(reader)
    logger.info(f"  Downloaded {len(rows)} rows")
    return rows


def parse_shoenot_row(row: Dict) -> Optional[Dict]:
    """Convert shoenot row to our canonical format."""
    try:
        year = int(row.get("year", 0))
        round_of = int(row.get("round_of", 0))
        if not year or not round_of:
            return None
        return {
            "round":   ROUND_MAP.get(round_of, str(round_of)),
            "region":  "",   # not in shoenot — will patch from ESPN
            "team_a":  row.get("winning_team_name", "").strip(),
            "team_b":  row.get("losing_team_name", "").strip(),
            "seed_a":  int(row.get("winning_team_seed") or 0),
            "seed_b":  int(row.get("losing_team_seed") or 0),
            "winner":  row.get("winning_team_name", "").strip(),
            "score_a": int(row.get("winning_team_score") or 0),
            "score_b": int(row.get("losing_team_score") or 0),
            "year":    year,
        }
    except (ValueError, TypeError):
        return None


def fetch_espn_games(year: int) -> List[Dict]:
    """Fetch tournament games for a year from ESPN scoreboard API."""
    dates = TOURNAMENT_DATES.get(year, [])
    if not dates:
        logger.warning(f"No date list for {year} — skipping ESPN fetch")
        return []

    games = []
    for date in dates:
        url = ESPN_SCOREBOARD.format(date=date)
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            for event in data.get("events", []):
                game = _parse_espn_event(event, year)
                if game:
                    games.append(game)
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"  ESPN {date}: {e}")

    logger.info(f"ESPN: fetched {len(games)} games for {year}")
    return games


def _parse_espn_event(event: Dict, year: int) -> Optional[Dict]:
    """Parse a single ESPN event into our format."""
    try:
        comp = event["competitions"][0]
        competitors = {c["homeAway"]: c for c in comp["competitors"]}
        home = competitors.get("home", {})
        away = competitors.get("away", {})

        if event.get("status", {}).get("type", {}).get("name", "").lower() not in ("final", "status_final"):
            return None

        # Round/region from notes headline
        notes = comp.get("notes", [])
        headline = notes[0].get("headline", "") if notes else ""
        rnd_label = _parse_round_from_headline(headline)
        region    = _parse_region_from_headline(headline)

        home_seed = home.get("curatedRank", {}).get("current") or 0
        away_seed = away.get("curatedRank", {}).get("current") or 0
        home_score = int(home.get("score", 0) or 0)
        away_score = int(away.get("score", 0) or 0)
        home_name  = home.get("team", {}).get("displayName", "")
        away_name  = away.get("team", {}).get("displayName", "")

        if home_score > away_score:
            team_a, team_b, seed_a, seed_b = home_name, away_name, home_seed, away_seed
            score_a, score_b, winner = home_score, away_score, home_name
        else:
            team_a, team_b, seed_a, seed_b = away_name, home_name, away_seed, home_seed
            score_a, score_b, winner = away_score, home_score, away_name

        return {
            "round": rnd_label, "region": region,
            "team_a": team_a, "team_b": team_b,
            "seed_a": seed_a, "seed_b": seed_b,
            "winner": winner, "score_a": score_a, "score_b": score_b,
            "year": year,
        }
    except (KeyError, IndexError, TypeError):
        return None


def _parse_round_from_headline(headline: str) -> str:
    h = headline.lower()
    if "first round" in h or "1st round" in h:
        return "R64"
    if "second round" in h or "2nd round" in h:
        return "R32"
    if "sweet 16" in h or "regional semifinal" in h:
        return "S16"
    if "elite eight" in h or "regional final" in h:
        return "E8"
    if "final four" in h or "national semifinal" in h:
        return "FF"
    if "national championship" in h or "championship" in h:
        return "F"
    return "UNKNOWN"


def _parse_region_from_headline(headline: str) -> str:
    for region in ("East", "West", "Midwest", "South"):
        if region.lower() in headline.lower():
            return region
    if any(x in headline.lower() for x in ("final four", "national", "championship")):
        return "Final Four"
    return ""


def write_year_csv(year: int, games: List[Dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{year}.csv"
    fieldnames = ["round", "region", "team_a", "team_b", "seed_a", "seed_b",
                  "winner", "score_a", "score_b"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for game in games:
            writer.writerow({k: game.get(k, "") for k in fieldnames})
    logger.info(f"  Wrote {len(games)} games to {path}")


def run_all(mock: bool = False) -> None:
    """Build cache for all years 2016-2025 (skip 2020)."""
    if mock:
        logger.info("[MOCK] Skipping all downloads — writing fixture data")
        for year in range(2016, 2026):
            if year == 2020:
                continue
            from src.backtesting.data_loader import _mock_tournament_results
            games = _mock_tournament_results(year)
            for g in games:
                g["year"] = year
            write_year_csv(year, games)
        return

    # Step 1: Download shoenot for 2016-2024
    all_rows = download_shoenot()
    by_year: Dict[int, List[Dict]] = {}
    for row in all_rows:
        parsed = parse_shoenot_row(row)
        if parsed and 2016 <= parsed["year"] <= 2024 and parsed["year"] != 2020:
            yr = parsed["year"]
            by_year.setdefault(yr, []).append(parsed)

    # Step 2: Write 2016-2024
    for year, games in sorted(by_year.items()):
        logger.info(f"Writing {year}: {len(games)} games")
        write_year_csv(year, games)

    # Step 3: Fetch 2025 from ESPN
    if 2025 in TOURNAMENT_DATES:
        games_2025 = fetch_espn_games(2025)
        if games_2025:
            write_year_csv(2025, games_2025)


def run_year(year: int, mock: bool = False) -> None:
    """Build cache for a single year."""
    if year == 2020:
        logger.info("Skipping 2020 — no tournament (COVID)")
        return
    if mock:
        from src.backtesting.data_loader import _mock_tournament_results
        games = _mock_tournament_results(year)
        for g in games:
            g["year"] = year
        write_year_csv(year, games)
        return
    if year <= 2024:
        all_rows = download_shoenot()
        games = [parse_shoenot_row(r) for r in all_rows
                 if int(r.get("year", 0)) == year]
        games = [g for g in games if g]
    else:
        games = fetch_espn_games(year)
    if games:
        write_year_csv(year, games)
    else:
        logger.error(f"No games found for {year}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--all",  action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.all:
        run_all(mock=args.mock)
    elif args.year:
        run_year(args.year, mock=args.mock)
    else:
        parser.print_help()
