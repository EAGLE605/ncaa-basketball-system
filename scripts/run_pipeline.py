"""
NCAAB Men's Basketball Prediction Pipeline Runner.

Executes the full prediction pipeline:
  Stage 1 (parallel): BartTorvik AdjEM + Pinnacle market lines
  Stage 2: Injury adjustments
  Stage 3: Win probability ensemble (65% market / 35% torvik)
  Stage 4: 250k Monte Carlo simulation
  Stage 5: Half-Kelly bet sizing
  Stage 6: Output (results JSON + console signal table)

Usage:
    python scripts/run_pipeline.py --year 2026 --mock
    python scripts/run_pipeline.py --year 2026
    python scripts/run_pipeline.py --year 2026 --output-dir brackets/2026/
"""

import argparse
import concurrent.futures
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.barttorvik import BartTorvik
from src.api.odds_client import OddsClient
from src.model.win_probability import build_wp_table
from src.simulation.tournament import simulate_tournament, top_n
from src.betting.kelly import kelly_fraction, edge

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Injury adjustments (update each tournament)
# ---------------------------------------------------------------------------
INJURY_ADJUSTMENTS: Dict[str, float] = {
    "Alabama":    -1.20,   # Holloway out
    "Duke":       -0.66,   # Foster limited
    "Texas Tech": -0.70,   # Toppin partial
    "Louisville": -0.50,   # Brown out
}

# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def stage_fetch_torvik(year: int, mock: bool) -> Dict[str, Dict]:
    """Stage 1a: Load BartTorvik AdjEM for tournament field."""
    if mock:
        logger.info("[MOCK] BartTorvik: returning fixture data")
        return _mock_torvik_data()

    bt = BartTorvik()
    logger.info(f"Fetching BartTorvik {year} season data...")
    return bt.get_season(year)


def stage_fetch_odds(mock: bool) -> Optional[Dict]:
    """Stage 1b: Fetch Pinnacle market lines from The Odds API."""
    if mock:
        logger.info("[MOCK] Odds API: skipping (no mock lines)")
        return None
    try:
        client = OddsClient()
        logger.info("Fetching Pinnacle tournament lines...")
        raw = client.get_pinnacle_lines()
        probs = client.extract_pinnacle_probs(raw)
        logger.info(f"Got Pinnacle lines for {len(probs)} games")
        return probs
    except RuntimeError as e:
        logger.warning(f"Odds API unavailable, using pure BartTorvik: {e}")
        return None


def stage_injury_adjustments(teams: Dict[str, Dict]) -> Dict[str, Dict]:
    """Stage 2: Apply injury AdjEM deltas."""
    adjusted = {}
    for team, stats in teams.items():
        if stats is None:
            adjusted[team] = stats
            continue
        delta = INJURY_ADJUSTMENTS.get(team, 0.0)
        if delta != 0.0:
            new_stats = dict(stats)
            new_stats["AdjEM"] = (stats.get("AdjEM") or 0.0) + delta
            new_stats["_injury_delta"] = delta
            adjusted[team] = new_stats
            logger.info(f"  {team}: AdjEM adjusted by {delta:+.2f}")
        else:
            adjusted[team] = stats
    return adjusted


def stage_win_probability(
    teams: Dict[str, Dict],
    market_lines: Optional[Dict],
) -> Dict[str, Dict[str, float]]:
    """Stage 3: Build WP table (ensemble or pure torvik)."""
    mode = "ensemble (65% market + 35% torvik)" if market_lines else "pure BartTorvik"
    logger.info(f"Building win probability table: {mode}")
    return build_wp_table(teams, market_lines)


def stage_monte_carlo(
    bracket: Dict[str, List[str]],
    wp_table: Dict[str, Dict[str, float]],
    n_sims: int = 250_000,
) -> Dict[str, Dict[str, float]]:
    """Stage 4: Vectorized Monte Carlo tournament simulation."""
    logger.info(f"Running {n_sims:,} tournament simulations...")
    t0 = time.time()
    results = simulate_tournament(bracket, wp_table, n_sims=n_sims)
    elapsed = time.time() - t0
    logger.info(f"Simulation complete in {elapsed:.1f}s")
    return results


def stage_kelly_sizing(
    sim_results: Dict[str, Dict[str, float]],
    market_lines: Optional[Dict],
    kelly_frac: float = 0.5,
    min_edge_pct: float = 0.02,
    max_bet_pct: float = 0.05,
) -> List[Dict]:
    """Stage 5: Compute half-Kelly bet sizes for available lines."""
    if not market_lines:
        logger.info("No market lines — skipping Kelly sizing")
        return []

    bets = []
    for matchup, line_data in market_lines.items():
        home = line_data.get("home")
        away = line_data.get("away")
        home_res = sim_results.get(home, {})
        away_res  = sim_results.get(away, {})

        # Use R64 probability as our model probability for this game
        p_home_model = home_res.get("R64", 0.5)
        p_away_model = away_res.get("R64", 0.5)

        ml_home = line_data.get("home_ml")
        ml_away = line_data.get("away_ml")
        if not ml_home or not ml_away:
            continue

        edge_home = edge(p_home_model, ml_home)
        edge_away = edge(p_away_model, ml_away)

        for team, p_model, ml, e in [
            (home, p_home_model, ml_home, edge_home),
            (away, p_away_model, ml_away, edge_away),
        ]:
            if e >= min_edge_pct:
                f = kelly_fraction(p_model, ml, fraction=kelly_frac, max_bet=max_bet_pct)
                if f > 0:
                    bets.append({
                        "matchup": matchup,
                        "team": team,
                        "p_model": round(p_model, 4),
                        "ml": ml,
                        "edge": round(e, 4),
                        "kelly_fraction": round(f, 4),
                    })

    bets.sort(key=lambda x: x["edge"], reverse=True)
    return bets


def stage_output(
    sim_results: Dict[str, Dict[str, float]],
    bets: List[Dict],
    year: int,
    output_dir: str = "brackets/2026/",
) -> str:
    """Stage 6: Write results and print signal table."""
    output = {
        "year": year,
        "model": "BartTorvik + Pinnacle ensemble",
        "n_sims": 250_000,
        "champion_probs": {
            t: round(sim_results[t]["champion"], 4)
            for t in sorted(sim_results, key=lambda x: sim_results[x]["champion"], reverse=True)
            if sim_results[t]["champion"] > 0.001
        },
        "bets": bets,
    }

    path = Path(output_dir) / f"results_{year}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results written to {path}")

    # Print top 10 champion probabilities
    print("\n=== NCAAB Men's Tournament Champion Probabilities ===")
    for team, prob in list(output["champion_probs"].items())[:10]:
        bar = "#" * int(prob * 200)
        print(f"  {team:<20} {prob:>6.1%}  {bar}")

    # Print betting signals
    if bets:
        print(f"\n=== Betting Signals ({len(bets)} bets with edge > 2%) ===")
        for bet in bets[:10]:
            print(
                f"  {bet['team']:<20} ML={bet['ml']:>+5d}  "
                f"model={bet['p_model']:.1%}  edge={bet['edge']:+.1%}  "
                f"Kelly={bet['kelly_fraction']:.1%}"
            )
    else:
        print("\nNo betting signals (no market lines or edge below threshold)")

    return str(path)


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------

def _mock_torvik_data() -> Dict[str, Dict]:
    """Minimal mock for --mock mode. Mirrors 2026 tournament top seeds."""
    return {
        "Duke":      {"AdjEM": 36.69, "AdjT": 67.5},
        "Michigan":  {"AdjEM": 36.61, "AdjT": 65.2},
        "Arizona":   {"AdjEM": 35.54, "AdjT": 69.1},
        "Florida":   {"AdjEM": 33.82, "AdjT": 68.3},
        "Illinois":  {"AdjEM": 33.73, "AdjT": 66.8},
        "Purdue":    {"AdjEM": 33.06, "AdjT": 63.5},
        "Houston":   {"AdjEM": 32.95, "AdjT": 64.2},
        "Iowa St":   {"AdjEM": 31.14, "AdjT": 67.1},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    year: int,
    bracket: Dict[str, List[str]],
    mock: bool = False,
    n_sims: int = 250_000,
    output_dir: str = "brackets/2026/",
) -> Dict:
    """
    Run the full prediction pipeline.

    Args:
        year:       Tournament year (e.g. 2026)
        bracket:    {region: [team_name x 16]} in seed order — must be provided
        mock:       True = skip all API calls, use fixture data
        n_sims:     Monte Carlo iterations
        output_dir: Where to write results JSON

    Returns:
        Pipeline output dict (same as results JSON)
    """
    logger.info(f"=== NCAAB Men's Prediction Pipeline — {year} ===")

    # Stage 1 (parallel): BartTorvik + market lines
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_torvik = pool.submit(stage_fetch_torvik, year, mock)
        f_odds   = pool.submit(stage_fetch_odds, mock)
        teams_raw     = f_torvik.result()
        market_lines  = f_odds.result()

    # Filter to bracket teams only
    all_bracket_teams = [t for region in bracket.values() for t in region]
    teams_bracket = {t: teams_raw.get(t) for t in all_bracket_teams}
    missing = [t for t, v in teams_bracket.items() if v is None]
    if missing:
        logger.warning(f"Missing Torvik data for {len(missing)} teams: {missing[:5]}...")

    # Stage 2: Injury adjustments
    teams_adjusted = stage_injury_adjustments(teams_bracket)

    # Stage 3: Win probability
    wp_table = stage_win_probability(teams_adjusted, market_lines)

    # Stage 4: Monte Carlo
    sim_results = stage_monte_carlo(bracket, wp_table, n_sims=n_sims)

    # Stage 5: Kelly sizing
    bets = stage_kelly_sizing(sim_results, market_lines)

    # Stage 6: Output
    stage_output(sim_results, bets, year, output_dir)

    return {"sim_results": sim_results, "bets": bets}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCAAB Men's prediction pipeline")
    parser.add_argument("--year",       type=int, default=2026)
    parser.add_argument("--mock",       action="store_true")
    parser.add_argument("--n-sims",     type=int, default=250_000)
    parser.add_argument("--output-dir", default="brackets/2026/")
    args = parser.parse_args()

    # Import bracket from simulate.py
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "brackets" / str(args.year)))
        from simulate import BRACKET
    except ImportError:
        logger.error("Could not import BRACKET from brackets/{year}/simulate.py. Run with existing bracket.")
        sys.exit(1)

    run(
        year=args.year,
        bracket=BRACKET,
        mock=args.mock,
        n_sims=args.n_sims,
        output_dir=args.output_dir,
    )
