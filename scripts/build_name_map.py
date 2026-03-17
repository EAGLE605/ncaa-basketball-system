"""
Build/update data/team_name_map.csv from live BartTorvik and ESPN data.

Fetches current season team names from both sources and adds any unknown
teams to the mapping file with canonical = torvik_name as the default.

Usage:
    python scripts/build_name_map.py                  # update from live APIs
    python scripts/build_name_map.py --year 2025      # specific season
    python scripts/build_name_map.py --check          # report unmapped names only
    python scripts/build_name_map.py --mock           # use fixture data

Output:
    data/team_name_map.csv (appends new rows, never overwrites existing)
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("build_name_map")

MAP_PATH = Path(__file__).parent.parent / "data" / "team_name_map.csv"
FIELDNAMES = ["canonical", "torvik_name", "espn_name", "alt_names"]


def load_existing_map() -> Dict[str, Dict]:
    """Load existing map file. Returns dict keyed by canonical name."""
    if not MAP_PATH.exists():
        logger.info(f"No existing map at {MAP_PATH} — will create fresh")
        return {}

    rows = {}
    with open(MAP_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical = row.get("canonical", "").strip()
            if canonical:
                rows[canonical] = {k: row.get(k, "") for k in FIELDNAMES}
    logger.info(f"Loaded {len(rows)} existing mappings from {MAP_PATH}")
    return rows


def get_all_known_names(existing: Dict[str, Dict]) -> Set[str]:
    """Return all names (canonical + torvik + espn + alts) that are already mapped."""
    known = set()
    for row in existing.values():
        for col in ("canonical", "torvik_name", "espn_name", "alt_names"):
            val = row.get(col, "").strip()
            for part in val.split("|"):
                part = part.strip()
                if part:
                    known.add(part.lower())
    return known


def fetch_torvik_names(year: int, mock: bool) -> List[str]:
    """Return list of team names from BartTorvik for a given season."""
    from src.backtesting.data_loader import NcaabDataLoader

    loader = NcaabDataLoader()
    try:
        data = loader.load_torvik_season(year, mock=mock)
        return [t["team"] for t in data if t.get("team")]
    except Exception as e:
        logger.warning(f"BartTorvik {year}: {e}")
        return []


def fetch_espn_names(year: int) -> List[str]:
    """Return list of team names from ESPN rankings for a given season."""
    from src.api.espn_ncaab import ESPNNcaab

    client = ESPNNcaab()
    try:
        rankings = client.get_rankings()
        return [t.get("displayName", "") for t in rankings if t.get("displayName")]
    except Exception as e:
        logger.warning(f"ESPN rankings: {e}")
        return []


def write_map(rows: Dict[str, Dict]) -> None:
    """Write all rows to team_name_map.csv, sorted by canonical name."""
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MAP_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for canonical in sorted(rows.keys()):
            writer.writerow(rows[canonical])
    logger.info(f"Wrote {len(rows)} rows to {MAP_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Build/update data/team_name_map.csv")
    parser.add_argument("--year",  type=int, default=2025)
    parser.add_argument("--check", action="store_true", help="Report unmapped only, no writes")
    parser.add_argument("--mock",  action="store_true")
    args = parser.parse_args()

    existing = load_existing_map()
    known_lower = get_all_known_names(existing)

    # Fetch names from both sources
    torvik_names = fetch_torvik_names(args.year, mock=args.mock)
    espn_names   = fetch_espn_names(args.year) if not args.mock else []

    all_source_names = set(torvik_names + espn_names)
    unmapped = [n for n in all_source_names if n.lower() not in known_lower and n]

    if not unmapped:
        logger.info("All fetched team names are already in the map.")
        return

    logger.info(f"\nFound {len(unmapped)} unmapped team names:")
    for name in sorted(unmapped):
        logger.info(f"  {name}")

    if args.check:
        logger.info("\n--check mode: no writes performed")
        return

    # Add new rows with canonical = torvik_name as default
    new_rows = 0
    for name in unmapped:
        canonical = name  # human should review and clean these up
        existing[canonical] = {
            "canonical":  canonical,
            "torvik_name": name if name in torvik_names else "",
            "espn_name":   name if name in espn_names else "",
            "alt_names":   "",
        }
        new_rows += 1

    write_map(existing)
    logger.info(
        f"\nAdded {new_rows} new rows. Review {MAP_PATH} and update canonical names as needed."
    )


if __name__ == "__main__":
    main()
