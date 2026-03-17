"""
Pull live Pinnacle moneylines for NCAA Men's Basketball tournament games
and write brackets/2026/market_lines.json for use by simulate.py.

simulate.py auto-loads market_lines.json on startup if it exists,
enabling the 65% Pinnacle / 35% BartTorvik ensemble model.

Usage:
    python scripts/pull_tournament_lines.py            # pull live lines
    python scripts/pull_tournament_lines.py --dry-run  # print without writing
    python scripts/pull_tournament_lines.py --mock     # fixture data (no API)

Output:
    brackets/2026/market_lines.json
    {"Team A vs Team B": 0.732, ...}  <- P(Team A wins), de-vigged

Requires:
    ODDS_API_KEY in .env or environment variable
    Add to .env:  ODDS_API_KEY=your_key_here
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("pull_lines")

OUT_PATH = Path(__file__).parent.parent / "brackets" / "2026" / "market_lines.json"

# Teams in the 2026 bracket (matches simulate.py keys exactly)
BRACKET_TEAMS = {
    # East
    "Duke", "Siena", "Ohio St", "TCU", "St Johns", "Northern Iowa",
    "Kansas", "Cal Baptist", "Louisville", "South Florida", "Michigan St",
    "North Dakota St", "UCLA", "UCF", "UConn", "Furman",
    # West
    "Arizona", "LIU", "Villanova", "Utah St", "Wisconsin", "High Point",
    "Arkansas", "Hawaii", "BYU", "NCST/SMU", "Gonzaga", "Kennesaw St",
    "Miami FL", "Missouri", "Purdue", "Queens",
    # Midwest
    "Michigan", "HWD/LEH", "Georgia", "Saint Louis", "Texas Tech", "Akron",
    "Alabama", "Hofstra", "Tennessee", "Miami OH", "Virginia", "Wright St",
    "Kentucky", "Santa Clara", "Iowa St", "Tennessee St",
    # South
    "Florida", "Prairie View", "Clemson", "Iowa", "Vanderbilt", "McNeese",
    "Nebraska", "Troy", "North Carolina", "VCU", "Illinois", "Penn",
    "Saint Marys", "Texas AM", "Houston", "Idaho",
}

# ESPN display name -> simulate.py canonical name
# Handles mismatches between what the API returns and our internal names
ESPN_TO_CANONICAL = {
    "Connecticut":               "UConn",
    "Connecticut Huskies":       "UConn",
    "NC State":                  "NCST/SMU",   # First Four winner slot
    "Saint John's":              "St Johns",
    "Saint John's (NY)":         "St Johns",
    "St. John's":                "St Johns",
    "St. Mary's":                "Saint Marys",
    "North Carolina State":      "NCST/SMU",
    "Miami (FL)":                "Miami FL",
    "Miami (Ohio)":              "Miami OH",
    "N.C. State":                "NCST/SMU",
    "Tennessee State":           "Tennessee St",
    "North Dakota State":        "North Dakota St",
    "Iowa State":                "Iowa St",
    "Michigan State":            "Michigan St",
    "California Baptist":        "Cal Baptist",
    "Howard":                    "HWD/LEH",    # First Four winner slot
    "Lehigh":                    "HWD/LEH",
    "Prairie View A&M":          "Prairie View",
    "Saint Louis":               "Saint Louis",
    "Santa Clara":               "Santa Clara",
    "Wright State":              "Wright St",
    "Kennesaw State":            "Kennesaw St",
}


def normalize(name: str) -> str:
    """Map API team name to simulate.py canonical name."""
    # Direct bracket match first (fastest path)
    if name in BRACKET_TEAMS:
        return name
    # Explicit override dict for First Four composites and known edge cases
    if name in ESPN_TO_CANONICAL:
        return ESPN_TO_CANONICAL[name]
    # Fall back to CSV resolver (covers all espn_name entries)
    from src.utils.team_names import resolve, UnresolvableTeamError
    try:
        canonical = resolve(name)
        if canonical in BRACKET_TEAMS:
            return canonical
    except UnresolvableTeamError:
        pass
    # Not a bracket team — return as-is so caller can detect and skip
    return name


def devig(ml_a: int, ml_b: int) -> float:
    """
    De-vig two moneylines. Returns P(team_a wins), de-vigged.
    Multiplicative method: divide each raw implied prob by their sum.
    """
    raw_a = abs(ml_a) / (abs(ml_a) + 100) if ml_a < 0 else 100 / (ml_a + 100)
    raw_b = abs(ml_b) / (abs(ml_b) + 100) if ml_b < 0 else 100 / (ml_b + 100)
    total = raw_a + raw_b
    return round(raw_a / total, 5)


def pull_live_lines() -> dict:
    """
    Fetch current NCAAB tournament moneylines from The Odds API.
    Returns dict: "Team A vs Team B" -> P(Team A wins)
    """
    from src.api.odds_client import OddsClient

    client = OddsClient()
    logger.info("Fetching live NCAAB odds from The Odds API (Pinnacle)...")

    try:
        odds_list = client.get_tournament_odds(odds_format="decimal", regions="eu")
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        return {}

    lines = {}
    for game in odds_list:
        home = normalize(game.get("home_team", ""))
        away = normalize(game.get("away_team", ""))
        if not home or not away:
            continue
        # Skip non-bracket teams (conference tournaments, NIT, etc.)
        if home not in BRACKET_TEAMS or away not in BRACKET_TEAMS:
            logger.debug(f"Skipping non-bracket game: {home} vs {away}")
            continue

        # Extract Pinnacle moneyline
        bookmakers = game.get("bookmakers", [])
        pinnacle = next((b for b in bookmakers if b["key"] == "pinnacle"), None)
        if not pinnacle:
            logger.debug(f"No Pinnacle line for {home} vs {away}")
            continue

        h2h = next((m for m in pinnacle.get("markets", []) if m["key"] == "h2h"), None)
        if not h2h:
            continue

        outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
        home_price = outcomes.get(game["home_team"]) or outcomes.get(home)
        away_price = outcomes.get(game["away_team"]) or outcomes.get(away)

        if home_price is None or away_price is None:
            logger.warning(f"Could not extract prices for {home} vs {away}: {outcomes}")
            continue

        # The Odds API returns decimal odds; convert to American for de-vig
        def dec_to_american(dec: float) -> int:
            if dec >= 2.0:
                return int((dec - 1) * 100)
            return int(-100 / (dec - 1))

        ml_home = dec_to_american(home_price)
        ml_away = dec_to_american(away_price)
        p_home = devig(ml_home, ml_away)

        key = f"{home} vs {away}"
        lines[key] = p_home
        logger.info(f"  {home} ({ml_home:+d}) vs {away} ({ml_away:+d}) -> P({home})={p_home:.3f}")

    logger.info(f"Fetched {len(lines)} matchups with Pinnacle lines")
    return lines


def mock_lines() -> dict:
    """Fixture market lines for testing (no API call)."""
    return {
        "Duke vs Siena":              0.985,
        "UConn vs Furman":            0.978,
        "Michigan vs HWD/LEH":        0.990,
        "Arizona vs LIU":             0.988,
        "Florida vs Prairie View":    0.992,
    }


def main():
    parser = argparse.ArgumentParser(description="Pull Pinnacle tournament lines for simulate.py")
    parser.add_argument("--dry-run", action="store_true", help="Print lines without writing file")
    parser.add_argument("--mock",    action="store_true", help="Use fixture data (no API)")
    args = parser.parse_args()

    if args.mock:
        lines = mock_lines()
        logger.info(f"[MOCK] Using {len(lines)} fixture lines")
    else:
        lines = pull_live_lines()

    if not lines:
        logger.error("No lines fetched — nothing to write")
        sys.exit(1)

    print(f"\n{'Matchup':45s}  P(Home wins)")
    print("-" * 60)
    for matchup, prob in sorted(lines.items()):
        print(f"  {matchup:43s}  {prob:.4f}")

    if args.dry_run:
        logger.info("\n--dry-run: file not written")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=2)
    logger.info(f"\nWrote {len(lines)} lines to {OUT_PATH}")
    logger.info("Run simulate.py to use these lines (auto-loaded on startup).")


if __name__ == "__main__":
    main()
