"""
2026 NCAA March Madness Pipeline Orchestrator.

Thin wrapper that:
  1. Refreshes brackets/2026/market_lines.json via pull_tournament_lines.py functions
  2. Runs brackets/2026/simulate.py (direct import, not subprocess)
  3. Loads results_2026.json and prints top-10 champion odds
  4. Shows Kelly bet sizing for any games with market lines

Usage:
    python scripts/run_pipeline.py             # pull live lines + run simulation
    python scripts/run_pipeline.py --mock      # use fixture lines (no API call)
    python scripts/run_pipeline.py --skip-lines  # skip API, use existing market_lines.json
"""

import argparse
import json
import sys
from pathlib import Path

# Project root on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Add brackets/2026 so we can import simulate directly
BRACKETS_2026 = ROOT / "brackets" / "2026"
sys.path.insert(0, str(BRACKETS_2026))

MARKET_LINES_PATH = BRACKETS_2026 / "market_lines.json"
RESULTS_PATH      = BRACKETS_2026 / "results_2026.json"


# ---------------------------------------------------------------------------
# Step 1: Refresh market lines
# ---------------------------------------------------------------------------

def refresh_market_lines(mock: bool) -> dict:
    """Pull lines (or use mock fixture) and write market_lines.json."""
    from scripts.pull_tournament_lines import pull_live_lines, mock_lines

    if mock:
        lines = mock_lines()
        print(f"[MOCK] Using {len(lines)} fixture market lines")
    else:
        lines = pull_live_lines()
        if not lines:
            print("WARNING: No lines fetched from API. market_lines.json not updated.")
            return {}

    MARKET_LINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MARKET_LINES_PATH, "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=2)
    print(f"Wrote {len(lines)} lines to {MARKET_LINES_PATH}")
    return lines


# ---------------------------------------------------------------------------
# Step 2: Run simulation
# ---------------------------------------------------------------------------

def run_simulation() -> None:
    """Import and execute brackets/2026/simulate.py run logic."""
    # Import the module (sys.path already includes brackets/2026)
    import simulate as sim_module

    print("\n--- Running 2026 simulation ---")
    adv     = sim_module.run_simulation()
    bracket = sim_module.generate_bracket(adv)
    sim_module.print_results(adv, bracket)
    projections = sim_module.compute_projected_totals(bracket)

    out_path = str(BRACKETS_2026 / "results_2026.json")
    sim_module.save_json(adv, bracket, out_path, projections)


# ---------------------------------------------------------------------------
# Step 3: Load results and display top-10 champion odds
# ---------------------------------------------------------------------------

def display_results() -> dict:
    """Load results_2026.json and print top-10 champion odds."""
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Did the simulation write results?")
        return {}

    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    champ_odds = results.get("championship_odds", {})
    print("\n=== TOP 10 CHAMPION ODDS ===")
    for i, (team, pct_str) in enumerate(champ_odds.items()):
        if i >= 10:
            break
        pct = float(pct_str.rstrip("%"))
        bar = "#" * int(pct * 2)
        print(f"  {team:<22} {pct_str:>7}  {bar}")

    return results


# ---------------------------------------------------------------------------
# Step 4: Kelly bet sizing for matchups with market lines
# ---------------------------------------------------------------------------

def show_kelly_bets(results: dict, market_lines: dict) -> None:
    """Print Kelly bet sizing for any R64 matchups with market lines."""
    if not market_lines:
        print("\nNo market lines available -- skipping Kelly sizing.")
        return

    from src.betting.kelly import kelly_fraction, edge

    full_adv = results.get("full_advancement", {})

    # market_lines.json format: "Team A vs Team B" -> P(Team A wins) de-vigged
    # We need American moneylines to compute Kelly; derive implied ML from de-vigged prob.
    # Since we only have the de-vigged probability (not raw moneylines), we estimate
    # a fair-vig ML for display purposes using -110 standard vig.
    def prob_to_ml(p: float) -> int:
        """Convert probability to American moneyline (fair, no vig)."""
        if p <= 0 or p >= 1:
            return 0
        if p >= 0.5:
            return -int(round(p / (1 - p) * 100))
        else:
            return int(round((1 - p) / p * 100))

    bets = []
    for matchup, p_market_home in market_lines.items():
        parts = matchup.split(" vs ")
        if len(parts) != 2:
            continue
        home, away = parts[0].strip(), parts[1].strip()

        # Get model championship probability as model signal
        # Use R64 advancement rate as the relevant round probability
        home_adv = full_adv.get(home, {})
        away_adv = full_adv.get(away, {})

        # full_advancement stores strings like "97.3%" -- convert to float
        def pct_to_float(s):
            try:
                return float(str(s).rstrip("%")) / 100.0
            except (ValueError, AttributeError):
                return 0.0

        p_home_model = pct_to_float(home_adv.get("R64", "0%"))
        p_away_model = pct_to_float(away_adv.get("R64", "0%"))

        # Market-implied moneylines (reconstructed from de-vigged prob)
        ml_home = prob_to_ml(p_market_home)
        ml_away = prob_to_ml(1.0 - p_market_home)

        if ml_home == 0 or ml_away == 0:
            continue

        for team, p_model, ml in [
            (home, p_home_model, ml_home),
            (away, p_away_model, ml_away),
        ]:
            e = edge(p_model, ml)
            if e >= 0.02:
                f = kelly_fraction(p_model, ml, fraction=0.5, max_bet=0.05)
                if f > 0:
                    bets.append({
                        "matchup": matchup,
                        "team":    team,
                        "p_model": p_model,
                        "p_mkt":   p_market_home if team == home else 1.0 - p_market_home,
                        "ml":      ml,
                        "edge":    e,
                        "kelly":   f,
                    })

    bets.sort(key=lambda x: x["edge"], reverse=True)

    if not bets:
        print("\nNo Kelly bets with edge >= 2% found.")
        return

    print(f"\n=== KELLY BET SIGNALS ({len(bets)} bets, edge >= 2%) ===")
    print(f"  {'Team':<22} {'Matchup':<35} {'ML':>6}  {'Model':>7}  {'Mkt':>7}  {'Edge':>6}  {'Kelly':>6}")
    print("  " + "-" * 95)
    for b in bets[:10]:
        short_matchup = b["matchup"][:33]
        print(
            f"  {b['team']:<22} {short_matchup:<35} {b['ml']:>+6d}"
            f"  {b['p_model']:>6.1%}  {b['p_mkt']:>6.1%}"
            f"  {b['edge']:>+6.1%}  {b['kelly']:>5.1%}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="2026 NCAA tournament pipeline orchestrator")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use fixture market lines (no API call)",
    )
    parser.add_argument(
        "--skip-lines",
        action="store_true",
        help="Skip market lines refresh, use existing market_lines.json",
    )
    args = parser.parse_args()

    print("=== 2026 NCAA Tournament Pipeline ===\n")

    # Step 1: Market lines
    if args.skip_lines:
        if MARKET_LINES_PATH.exists():
            with open(MARKET_LINES_PATH, encoding="utf-8") as f:
                market_lines = json.load(f)
            print(f"Using existing market_lines.json ({len(market_lines)} lines)")
        else:
            print(f"WARNING: {MARKET_LINES_PATH} not found and --skip-lines set. Running without lines.")
            market_lines = {}
    else:
        market_lines = refresh_market_lines(mock=args.mock)

    # Step 2: Simulation
    run_simulation()

    # Step 3: Results display
    results = display_results()

    # Step 4: Kelly sizing
    show_kelly_bets(results, market_lines)

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
