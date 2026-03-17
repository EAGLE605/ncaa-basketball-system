"""
Download and cache BartTorvik AdjEM data for all seasons 2016-2025.

Saves per-year JSON to data/cache/torvik_{year}.json.
Uses the verified column mapping from CLAUDE.md (2026-03-17).

Usage:
    python scripts/cache_torvik_history.py            # all years
    python scripts/cache_torvik_history.py --year 2024  # single year
    python scripts/cache_torvik_history.py --force    # re-download existing
    python scripts/cache_torvik_history.py --mock     # write fixture data
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("torvik_cache")

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
YEARS = list(range(2016, 2026))  # 2016-2025 inclusive


def cache_year(year: int, force: bool = False) -> bool:
    """
    Download and cache BartTorvik data for a single season.

    Returns True on success, False on failure.
    """
    from src.api.barttorvik import BartTorvik

    out_path = CACHE_DIR / f"torvik_{year}.json"
    if out_path.exists() and not force:
        logger.info(f"  {year}: already cached at {out_path} (use --force to re-download)")
        return True

    logger.info(f"  {year}: downloading from BartTorvik...")
    try:
        client = BartTorvik()
        data = client.get_season(year)
        if not data:
            logger.error(f"  {year}: empty response from BartTorvik")
            return False

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"  {year}: cached {len(data)} teams -> {out_path}")
        return True

    except Exception as e:
        logger.error(f"  {year}: download failed — {e}")
        return False


def cache_year_mock(year: int) -> bool:
    """Write fixture data for a single year (no network)."""
    from src.backtesting.data_loader import NcaabDataLoader

    out_path = CACHE_DIR / f"torvik_{year}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    loader = NcaabDataLoader()
    data = loader.load_torvik_season(year, mock=True)
    if not data:
        logger.warning(f"  {year}: mock returned empty data")
        return False

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"  {year}: wrote mock fixture ({len(data)} teams) -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Cache BartTorvik AdjEM history 2016-2025")
    parser.add_argument("--year",  type=int, help="Cache a single year only")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--mock",  action="store_true", help="Write fixture data (no network)")
    args = parser.parse_args()

    years = [args.year] if args.year else YEARS
    logger.info(f"Caching BartTorvik data for years: {years}")

    ok = 0
    fail = 0
    for year in years:
        if args.mock:
            success = cache_year_mock(year)
        else:
            success = cache_year(year, force=args.force)
            if success and year != years[-1]:
                time.sleep(0.5)  # be polite to BartTorvik

        if success:
            ok += 1
        else:
            fail += 1

    logger.info(f"\nDone: {ok} succeeded, {fail} failed")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
