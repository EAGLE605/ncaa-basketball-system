"""
Score a filled bracket against actual tournament results and
compare to our simulation's prediction accuracy.

Usage:
    # Score against live ESPN results (games completed so far):
    python scripts/score_bracket.py --bracket brackets/2026/results_2026.json

    # Score against a picks file:
    python scripts/score_bracket.py --picks my_picks.json

    # Mock (unit test friendly):
    python scripts/score_bracket.py --bracket brackets/2026/results_2026.json --mock

Scoring:
    Standard ESPN/CBS format:
        R64 = 10 pts, R32 = 20 pts, S16 = 40 pts,
        E8 = 80 pts, FF = 160 pts, Champion = 320 pts
    Plus: upset bonus (+10 per upset correctly picked)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("score_bracket")

ROUND_POINTS = {
    "R64": 10,
    "R32": 20,
    "S16": 40,
    "E8":  80,
    "FF":  160,
    "F":   320,
}

ROUND_LABELS = ["R64", "R32", "S16", "E8", "FF", "F"]

# Canonical round labels used in our data vs ESPN round-of-N labels
ROUND_OF_N_TO_LABEL = {
    64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "FF", 2: "F",
}


def load_simulation_results(path: str) -> dict:
    """Load results_2026.json simulation output."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_actual_results(year: int, mock: bool = False) -> List[Dict]:
    """
    Fetch completed tournament games for scoring.
    Returns list of dicts: {round, team_a, team_b, winner, seed_a, seed_b}
    """
    if mock:
        return _mock_results()

    # Try local cache first
    cache_path = Path(__file__).parent.parent / "data" / "tournament_results" / f"{year}.csv"
    if cache_path.exists():
        import csv
        results = []
        with open(cache_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                results.append({
                    "round":  row["round"],
                    "team_a": row["team_a"],
                    "team_b": row["team_b"],
                    "winner": row["winner"],
                    "seed_a": int(row.get("seed_a") or 0),
                    "seed_b": int(row.get("seed_b") or 0),
                })
        if results:
            logger.info(f"Loaded {len(results)} completed games from {cache_path}")
            return results

    # Fall back to ESPN live scoreboard
    logger.info("No local cache — fetching from ESPN API...")
    from scripts.update_tournament_cache import fetch_espn_games
    return fetch_espn_games(year)


def _mock_results() -> List[Dict]:
    """Minimal fixture results for testing."""
    return [
        {"round": "R64", "team_a": "Duke",     "team_b": "Siena",   "winner": "Duke",     "seed_a": 1, "seed_b": 16},
        {"round": "R64", "team_a": "Ohio St",  "team_b": "TCU",     "winner": "Ohio St",  "seed_a": 8, "seed_b": 9},
        {"round": "R64", "team_a": "Michigan", "team_b": "HWD/LEH", "winner": "Michigan", "seed_a": 1, "seed_b": 16},
        {"round": "R64", "team_a": "Iowa St",  "team_b": "Tennessee St", "winner": "Iowa St", "seed_a": 2, "seed_b": 15},
        {"round": "R64", "team_a": "Florida",  "team_b": "Prairie View", "winner": "Florida", "seed_a": 1, "seed_b": 16},
    ]


def normalize_team(name: str) -> str:
    """Resolve ESPN display names to our canonical names for comparison."""
    from src.utils.team_names import resolve, UnresolvableTeamError
    try:
        return resolve(name, fuzzy=True)
    except UnresolvableTeamError:
        return name


def score_bracket(sim_results: dict, actual_results: List[Dict]) -> dict:
    """
    Score simulation bracket picks against actual completed results.

    Returns:
        {
          "total_points": int,
          "max_possible": int,
          "by_round": {"R64": {"correct": n, "possible": n, "points": n}, ...},
          "correct_picks": [...],
          "missed_picks": [...],
          "upset_bonus": int,
        }
    """
    # Build set of actual winners by round
    actual_by_round: Dict[str, Dict[str, str]] = {}  # round -> {team_a|team_b -> winner}
    for game in actual_results:
        rnd = game["round"]
        if rnd not in actual_by_round:
            actual_by_round[rnd] = {}
        ta = normalize_team(game["team_a"])
        tb = normalize_team(game["team_b"])
        winner = normalize_team(game["winner"])
        actual_by_round[rnd][frozenset([ta, tb])] = winner

    # Extract our picks from simulation results
    picks_by_round = {}
    r64_raw = sim_results.get("round_of_64", {})
    r32_raw = sim_results.get("round_of_32", {})
    s16_raw = sim_results.get("sweet_sixteen", {})
    e8_raw  = sim_results.get("elite_eight", {})    # region -> team
    ff_raw  = sim_results.get("final_four", [])
    champ   = sim_results.get("champion", "")

    # Flatten by-region to flat list of picked winners
    def flatten(by_region):
        result = []
        for teams in by_region.values():
            if isinstance(teams, list):
                result.extend(teams)
            else:
                result.append(teams)
        return result

    picks_by_round["R64"] = flatten(r64_raw)
    picks_by_round["R32"] = flatten(r32_raw)
    picks_by_round["S16"] = flatten(s16_raw)
    picks_by_round["E8"]  = list(e8_raw.values()) if isinstance(e8_raw, dict) else e8_raw
    picks_by_round["FF"]  = ff_raw
    picks_by_round["F"]   = [champ]

    scoring = {}
    correct_picks = []
    missed_picks  = []
    upset_bonus   = 0
    total_pts     = 0
    max_possible  = 0

    # Get seeds for upset detection
    adv = sim_results.get("full_advancement", {})
    def get_seed(team):
        return adv.get(team, {}).get("seed", 0)

    for rnd in ROUND_LABELS:
        actual_games = actual_by_round.get(rnd, {})
        picks = picks_by_round.get(rnd, [])
        pts_per = ROUND_POINTS[rnd]
        correct = 0
        possible = len(picks)

        for pick in picks:
            pick_norm = normalize_team(pick)
            # Find the game in actual results that contains this pick
            for matchup, winner in actual_games.items():
                if pick_norm in matchup:
                    if normalize_team(winner) == pick_norm:
                        correct += 1
                        pts = pts_per
                        # Upset bonus
                        loser = next(iter(matchup - {pick_norm}), "")
                        if loser and get_seed(pick_norm) > get_seed(loser):
                            pts += 10
                            upset_bonus += 10
                        total_pts += pts
                        correct_picks.append(f"{rnd}: {pick_norm} ({pts}pts)")
                    else:
                        missed_picks.append(f"{rnd}: picked {pick_norm}, actual winner={winner}")
                    break

        max_possible += possible * pts_per
        scoring[rnd] = {
            "correct":  correct,
            "possible": possible,
            "points":   correct * pts_per,
            "pct":      round(correct / possible, 3) if possible else 0.0,
        }

    return {
        "total_points":  total_pts,
        "max_possible":  max_possible,
        "upset_bonus":   upset_bonus,
        "by_round":      scoring,
        "correct_picks": correct_picks,
        "missed_picks":  missed_picks,
    }


def compare_to_simulation(sim_results: dict, actual_results: List[Dict]) -> dict:
    """Compare simulation accuracy metrics to actual results."""
    adv = sim_results.get("full_advancement", {})
    correct = 0
    total   = 0
    by_round: Dict[str, Dict] = {}

    actual_by_round: Dict[str, Dict] = {}
    for game in actual_results:
        rnd = game["round"]
        ta = normalize_team(game["team_a"])
        tb = normalize_team(game["team_b"])
        winner = normalize_team(game["winner"])
        actual_by_round.setdefault(rnd, {})[frozenset([ta, tb])] = winner

    for rnd, games in actual_by_round.items():
        rnd_correct = 0
        for matchup, winner in games.items():
            teams = list(matchup)
            if len(teams) < 2:
                continue
            ta, tb = teams[0], teams[1]
            # Our simulation's predicted winner = higher advancement count at this round
            rnd_idx = ROUND_LABELS.index(rnd) if rnd in ROUND_LABELS else None
            if rnd_idx is None:
                continue
            adv_a = adv.get(ta, {}).get(rnd, "0%")
            adv_b = adv.get(tb, {}).get(rnd, "0%")
            pred_winner = ta if float(str(adv_a).rstrip("%")) >= float(str(adv_b).rstrip("%")) else tb
            if pred_winner == normalize_team(winner):
                rnd_correct += 1
                correct += 1
            total += 1

        by_round[rnd] = {
            "correct":  rnd_correct,
            "total":    len(games),
            "accuracy": round(rnd_correct / len(games), 3) if games else 0.0,
        }

    return {
        "overall_accuracy": round(correct / total, 3) if total else 0.0,
        "correct":  correct,
        "total":    total,
        "by_round": by_round,
    }


def main():
    parser = argparse.ArgumentParser(description="Score 2026 NCAA bracket against actual results")
    parser.add_argument("--bracket", default="brackets/2026/results_2026.json",
                        help="Simulation results JSON to score")
    parser.add_argument("--year",   type=int, default=2026)
    parser.add_argument("--mock",   action="store_true")
    args = parser.parse_args()

    sim_path = Path(args.bracket)
    if not sim_path.exists():
        logger.error(f"Bracket file not found: {sim_path}")
        sys.exit(1)

    sim_results   = load_simulation_results(str(sim_path))
    actual        = fetch_actual_results(args.year, mock=args.mock)

    if not actual:
        logger.warning("No completed games yet — nothing to score")
        print("\nTournament hasn't started or no results cached.")
        print("Run: python scripts/update_tournament_cache.py --year 2026")
        return

    score = score_bracket(sim_results, actual)
    sim_acc = compare_to_simulation(sim_results, actual)

    # Print results
    print("\n" + "=" * 65)
    print(f"BRACKET SCORE vs ACTUAL RESULTS  ({len(actual)} games complete)")
    print("=" * 65)
    print(f"\nTotal Points:    {score['total_points']} / {score['max_possible']} possible")
    print(f"Upset Bonus:     {score['upset_bonus']} pts")
    print(f"\n{'Round':6s}  {'Correct':>8s}  {'Possible':>9s}  {'Points':>7s}  {'Acc':>6s}")
    print("-" * 45)
    for rnd in ROUND_LABELS:
        if rnd in score["by_round"]:
            r = score["by_round"][rnd]
            print(f"  {rnd:4s}  {r['correct']:>8d}  {r['possible']:>9d}  {r['points']:>7d}  {r['pct']:>5.1%}")

    print(f"\n{'='*65}")
    print(f"SIMULATION ACCURACY vs ACTUAL RESULTS")
    print(f"{'='*65}")
    print(f"Overall: {sim_acc['overall_accuracy']:.1%}  ({sim_acc['correct']}/{sim_acc['total']} games)")
    print(f"\n{'Round':6s}  {'Correct':>8s}  {'Total':>7s}  {'Accuracy':>9s}")
    print("-" * 40)
    for rnd, r in sim_acc["by_round"].items():
        print(f"  {rnd:4s}  {r['correct']:>8d}  {r['total']:>7d}  {r['accuracy']:>8.1%}")

    if score["missed_picks"]:
        print(f"\n{'='*65}")
        print("MISSED PICKS:")
        for m in score["missed_picks"]:
            print(f"  {m}")


if __name__ == "__main__":
    main()
