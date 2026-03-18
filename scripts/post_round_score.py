"""
Post-round scoring automation for the 2026 NCAA tournament.

Runs after each tournament round to fetch the latest results, score the bracket,
and save a timestamped record of the output.

Usage:
    python scripts/post_round_score.py --round R64
    python scripts/post_round_score.py --round R32 --mock
    python scripts/post_round_score.py --update-cache
    python scripts/post_round_score.py --update-cache --round R64

Valid round values: R64, R32, S16, E8, FF, F

Output:
    .omc/round_scores/{round}_{YYYYMMDD_HHMMSS}.txt

Requires:
    brackets/2026/results_2026.json  (run simulate.py first)
    data/tournament_results/2026.csv (updated by --update-cache or update_tournament_cache.py)
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("post_round_score")

ROOT         = Path(__file__).parent.parent
BRACKET_PATH = ROOT / "brackets" / "2026" / "results_2026.json"
SCORES_DIR   = ROOT / ".omc" / "round_scores"
YEAR         = 2026

VALID_ROUNDS = {"R64", "R32", "S16", "E8", "FF", "F"}

ROUND_NAMES = {
    "R64": "Round of 64",
    "R32": "Round of 32",
    "S16": "Sweet 16",
    "E8":  "Elite Eight",
    "FF":  "Final Four",
    "F":   "National Championship",
}


def update_cache(mock: bool = False) -> bool:
    """
    Call update_tournament_cache.py --year 2026 to refresh ESPN results.
    Returns True on success.
    """
    logger.info(f"Updating tournament cache for {YEAR}...")
    cmd = [sys.executable, str(ROOT / "scripts" / "update_tournament_cache.py"),
           "--year", str(YEAR)]
    if mock:
        cmd.append("--mock")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"Cache update failed (exit {result.returncode}):\n{result.stderr}")
            return False
        logger.info("Cache updated successfully.")
        if result.stdout:
            logger.info(result.stdout.strip())
        return True
    except subprocess.TimeoutExpired:
        logger.error("Cache update timed out after 60 seconds")
        return False
    except Exception as e:
        logger.error(f"Cache update error: {e}")
        return False


def run_score_bracket(mock: bool = False) -> str:
    """
    Call score_bracket.py and capture its output as a string.
    Returns the captured output, or empty string on failure.
    """
    if not BRACKET_PATH.exists():
        logger.error(f"Bracket file not found: {BRACKET_PATH}")
        logger.error("Run simulate.py first to generate results_2026.json")
        return ""

    logger.info("Scoring bracket against actual results...")
    cmd = [sys.executable, str(ROOT / "scripts" / "score_bracket.py"),
           "--bracket", str(BRACKET_PATH),
           "--year", str(YEAR)]
    if mock:
        cmd.append("--mock")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        combined = result.stdout
        if result.stderr:
            # score_bracket.py logs to stderr via logging; include at debug level
            logger.debug(f"score_bracket stderr:\n{result.stderr}")
        if result.returncode != 0:
            logger.error(f"score_bracket.py failed (exit {result.returncode})")
            if result.stderr:
                logger.error(result.stderr.strip())
        return combined
    except subprocess.TimeoutExpired:
        logger.error("score_bracket.py timed out after 60 seconds")
        return ""
    except Exception as e:
        logger.error(f"score_bracket.py error: {e}")
        return ""


def parse_score_output(raw: str) -> dict:
    """
    Parse score_bracket.py stdout to extract key metrics for the summary.
    Returns a dict with total_points, max_possible, upset_bonus, by_round,
    sim_accuracy, and missed_picks.
    """
    result = {
        "total_points": None,
        "max_possible": None,
        "upset_bonus":  None,
        "games_complete": None,
        "by_round":     {},
        "sim_accuracy": None,
        "missed_picks": [],
    }

    lines = raw.splitlines()
    in_round_table    = False
    in_sim_table      = False
    in_missed         = False

    for line in lines:
        stripped = line.strip()

        # Header line with game count
        if "BRACKET SCORE vs ACTUAL RESULTS" in stripped:
            # e.g. "BRACKET SCORE vs ACTUAL RESULTS  (32 games complete)"
            import re
            m = re.search(r"\((\d+) games complete\)", stripped)
            if m:
                result["games_complete"] = int(m.group(1))
            in_round_table = False
            in_sim_table   = False
            in_missed      = False

        elif stripped.startswith("Total Points:"):
            # "Total Points:    120 / 320 possible"
            import re
            m = re.search(r"(\d+)\s*/\s*(\d+)", stripped)
            if m:
                result["total_points"] = int(m.group(1))
                result["max_possible"] = int(m.group(2))

        elif stripped.startswith("Upset Bonus:"):
            import re
            m = re.search(r"(\d+)", stripped)
            if m:
                result["upset_bonus"] = int(m.group(1))

        elif stripped.startswith("Round") and "Correct" in stripped and "Possible" in stripped:
            in_round_table = True
            in_sim_table   = False
            in_missed      = False

        elif stripped.startswith("SIMULATION ACCURACY"):
            in_round_table = False
            in_sim_table   = True
            in_missed      = False

        elif stripped.startswith("MISSED PICKS"):
            in_round_table = False
            in_sim_table   = False
            in_missed      = True

        elif in_round_table and stripped and not stripped.startswith("-"):
            # "R64       28        32     280  87.5%"
            import re
            m = re.match(r"(R64|R32|S16|E8|FF|F)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+%)", stripped)
            if m:
                rnd = m.group(1)
                result["by_round"][rnd] = {
                    "correct":  int(m.group(2)),
                    "possible": int(m.group(3)),
                    "points":   int(m.group(4)),
                    "pct":      m.group(5),
                }

        elif in_sim_table and stripped.startswith("Overall:"):
            # "Overall: 78.1%  (25/32 games)"
            result["sim_accuracy"] = stripped

        elif in_missed and stripped and not stripped.startswith("="):
            result["missed_picks"].append(stripped)

    return result


def format_summary(rnd: str, parsed: dict, raw_output: str) -> str:
    """Build a clean human-readable summary string."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rnd_label = ROUND_NAMES.get(rnd, rnd) if rnd else "All rounds"
    games_str = f" ({parsed['games_complete']} games complete)" if parsed["games_complete"] else ""

    lines = [
        "=" * 65,
        f"POST-ROUND SCORE REPORT -- {rnd_label}{games_str}",
        f"Generated: {ts}",
        "=" * 65,
    ]

    if parsed["total_points"] is not None:
        lines += [
            "",
            f"Bracket Score:  {parsed['total_points']} / {parsed['max_possible']} pts possible",
        ]
        if parsed["upset_bonus"] is not None:
            lines.append(f"Upset Bonus:    {parsed['upset_bonus']} pts")

    if parsed["by_round"]:
        lines += [
            "",
            f"{'Round':6s}  {'Correct':>8s}  {'Possible':>9s}  {'Points':>7s}  {'Acc':>6s}",
            "-" * 45,
        ]
        for r in ("R64", "R32", "S16", "E8", "FF", "F"):
            if r in parsed["by_round"]:
                d = parsed["by_round"][r]
                lines.append(
                    f"  {r:4s}  {d['correct']:>8d}  {d['possible']:>9d}"
                    f"  {d['points']:>7d}  {d['pct']:>6s}"
                )

    if parsed["sim_accuracy"]:
        lines += [
            "",
            f"Simulation accuracy: {parsed['sim_accuracy']}",
        ]

    upsets_in_missed = [m for m in parsed["missed_picks"] if "upset" in m.lower()]
    regular_missed   = [m for m in parsed["missed_picks"] if m not in upsets_in_missed]

    if regular_missed:
        lines += ["", "Missed picks:"]
        for m in regular_missed[:20]:
            lines.append(f"  {m}")
        if len(regular_missed) > 20:
            lines.append(f"  ... and {len(regular_missed) - 20} more")

    lines += [
        "",
        "=" * 65,
        "FULL SCORE OUTPUT",
        "=" * 65,
        raw_output,
    ]

    return "\n".join(lines)


def save_output(rnd: str, text: str) -> Path:
    """Save report to .omc/round_scores/{round}_{datetime}.txt."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{rnd}_{ts}.txt" if rnd else f"ALL_{ts}.txt"
    path  = SCORES_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Post-round scoring: fetch results, score bracket, save history"
    )
    parser.add_argument(
        "--round", metavar="RND",
        help=f"Tournament round label: {', '.join(sorted(VALID_ROUNDS))}",
    )
    parser.add_argument(
        "--update-cache", action="store_true",
        help="Fetch latest ESPN results before scoring",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use fixture data (no live API calls)",
    )
    args = parser.parse_args()

    rnd = args.round
    if rnd and rnd.upper() not in VALID_ROUNDS:
        logger.error(f"Invalid round '{rnd}'. Valid values: {', '.join(sorted(VALID_ROUNDS))}")
        sys.exit(1)
    if rnd:
        rnd = rnd.upper()

    # Step 1: optionally refresh the ESPN cache
    if args.update_cache:
        ok = update_cache(mock=args.mock)
        if not ok:
            logger.warning("Cache update failed — scoring against last cached results")

    # Step 2: run score_bracket.py, capture output
    raw = run_score_bracket(mock=args.mock)
    if not raw:
        logger.error("No output from score_bracket.py — nothing to save")
        sys.exit(1)

    # Step 3: print to stdout
    print(raw, end="")

    # Step 4: build summary and save to file
    parsed  = parse_score_output(raw)
    summary = format_summary(rnd, parsed, raw)
    out_path = save_output(rnd or "ALL", summary)
    logger.info(f"Report saved to {out_path}")

    # Step 5: quick digest to stdout
    if parsed["total_points"] is not None:
        print()
        print(f"[Summary] {ROUND_NAMES.get(rnd, rnd or 'All')}: "
              f"{parsed['total_points']}/{parsed['max_possible']} pts  "
              + (f"| upset bonus {parsed['upset_bonus']} pts  " if parsed["upset_bonus"] else "")
              + (f"| sim acc: {parsed['sim_accuracy']}" if parsed["sim_accuracy"] else ""))
    print(f"[Saved]   {out_path}")


if __name__ == "__main__":
    main()
