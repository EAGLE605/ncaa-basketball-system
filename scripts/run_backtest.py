"""
CLI entry point for walk-forward backtesting.

Usage:
    python scripts/run_backtest.py --mock                    # fixture data
    python scripts/run_backtest.py --start 2016 --end 2025   # real data
    python scripts/run_backtest.py --year 2024               # single year
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.engine import TournamentBacktestOrchestrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backtest")


def main():
    parser = argparse.ArgumentParser(description="NCAAB Men's walk-forward backtest")
    parser.add_argument("--start", type=int, default=2016)
    parser.add_argument("--end",   type=int, default=2025)
    parser.add_argument("--year",  type=int, help="Run single year only")
    parser.add_argument("--mock",  action="store_true")
    parser.add_argument("--n-sims", type=int, default=10_000)
    parser.add_argument("--output", default=".omc/backtest_results.json")
    args = parser.parse_args()

    orch = TournamentBacktestOrchestrator(n_sims=args.n_sims)

    if args.year:
        result = orch.run_year(args.year, prior_games=[], mock=args.mock)
        print(f"\nYear {args.year}:")
        print(f"  Accuracy: {result.accuracy_by_round}")
        print(f"  Brier:    {result.brier_score:.4f}")
        print(f"  ROI:      {result.roi_pct:+.1f}%")
        return

    wf = orch.run_walk_forward(start=args.start, end=args.end, mock=args.mock)

    # Print summary
    print(f"\n=== Walk-Forward Results {args.start}–{args.end} ===")
    print(f"Years completed: {len(wf.years)}  |  Errors: {len(wf.year_errors)}")
    if wf.year_errors:
        for yr, err in wf.year_errors.items():
            print(f"  {yr} FAILED: {err}")
    print(f"\nEnsemble Accuracy:")
    for rnd, acc in wf.ensemble_accuracy.items():
        print(f"  {rnd:8s}: {acc:.1%}")
    print(f"\nTotal ROI:   {wf.total_roi_pct:+.1f}%")
    print(f"Total Sharpe: {wf.total_sharpe:.3f}")
    if wf.clv_summary:
        print(f"Mean CLV:    {wf.clv_summary['mean']:+.3f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "ensemble_accuracy": wf.ensemble_accuracy,
        "total_roi_pct": wf.total_roi_pct,
        "total_sharpe": wf.total_sharpe,
        "clv_summary": wf.clv_summary,
        "year_errors": wf.year_errors,
        "per_year": [
            {
                "year": y.year,
                "accuracy": y.accuracy_by_round,
                "brier": y.brier_score,
                "roi_pct": y.roi_pct,
                "calibration_skipped": y.calibration_skipped,
            }
            for y in wf.years
        ],
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
