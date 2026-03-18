"""
Update brackets/2026/simulate.py with First Four game winners.

First Four matchups:
  Midwest #16: Howard vs Lehigh   -> placeholder "HWD/LEH"
  West   #11: NC State vs SMU    -> placeholder "NCST/SMU"

Usage:
    python scripts/update_first_four.py --midwest-winner Howard
    python scripts/update_first_four.py --west-winner "NC State"
    python scripts/update_first_four.py --midwest-winner Lehigh --west-winner SMU
    python scripts/update_first_four.py --midwest-winner Howard --west-winner "NC State" --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

SIMULATE_PATH = Path(__file__).parent.parent / "brackets" / "2026" / "simulate.py"

# ---------------------------------------------------------------------------
# Known team data for First Four participants
# Use actual winner's real AdjEM from BartTorvik (or hardcoded estimate).
# ---------------------------------------------------------------------------
TEAM_DATA = {
    # Midwest #16 First Four
    "Howard":   {"em": -8.0,  "ao": 103.0, "ad": 111.0, "tempo": 67, "seed": 16, "region": "Midwest", "3pt": False},
    "Lehigh":   {"em": -7.0,  "ao": 103.5, "ad": 110.5, "tempo": 68, "seed": 16, "region": "Midwest", "3pt": False},
    # West #11 First Four
    "NC State": {"em": 14.69, "ao": 114.8, "ad": 100.1, "tempo": 70, "seed": 11, "region": "West",    "3pt": False},
    "SMU":      {"em": 14.00, "ao": 114.0, "ad": 100.0, "tempo": 71, "seed": 11, "region": "West",    "3pt": True},
}

# Mapping from CLI winner name variants to canonical form and slot info
MIDWEST_CANDIDATES = {
    "howard":   "Howard",
    "lehigh":   "Lehigh",
}
WEST_CANDIDATES = {
    "nc state": "NC State",
    "ncstate":  "NC State",
    "ncst":     "NC State",
    "smu":      "SMU",
}

# Composite placeholders currently in simulate.py
MIDWEST_PLACEHOLDER = "HWD/LEH"
WEST_PLACEHOLDER    = "NCST/SMU"


def resolve_winner(raw: str, candidates: dict, slot_label: str) -> str:
    """Normalize winner name and validate against known candidates."""
    key = raw.strip().lower()
    if key not in candidates:
        valid = ", ".join(candidates.values())
        print(f"ERROR: Unknown {slot_label} winner '{raw}'. Valid options: {valid}")
        sys.exit(1)
    return candidates[key]


def build_teams_raw_line(winner: str) -> str:
    """
    Build the _TEAMS_RAW entry line for the winner, replacing the composite placeholder.
    The format must match the existing style in simulate.py exactly.
    """
    d = TEAM_DATA[winner]
    three_pt_str = "True" if d["3pt"] else "False"
    # Use explicit ao/ad (not _est_ao/_est_ad) since we have real data
    line = (
        f'    "{winner}":'
        f'            {{"em": {d["em"]:.2f}, "ao": {d["ao"]:.1f},'
        f' "ad": {d["ad"]:.1f}, "tempo": {d["tempo"]},'
        f' "seed": {d["seed"]},  "region": "{d["region"]}", "3pt": {three_pt_str}}},'
    )
    return line


def make_changes(content: str, midwest_winner: str | None, west_winner: str | None) -> tuple[str, list[str]]:
    """
    Apply First Four winner substitutions to simulate.py content.
    Returns (new_content, list_of_change_descriptions).
    """
    changes = []
    new_content = content

    def replace_all_occurrences(text: str, old: str, new: str, label: str) -> tuple[str, int]:
        count = text.count(old)
        if count == 0:
            print(f"  WARNING: '{old}' not found in simulate.py -- already updated?")
            return text, 0
        return text.replace(old, new), count

    if midwest_winner:
        placeholder = MIDWEST_PLACEHOLDER
        winner = midwest_winner

        # 1. Replace the _TEAMS_RAW entry (the composite line with "HWD/LEH")
        # Pattern: the full line starting with "HWD/LEH" key in _TEAMS_RAW
        teams_raw_pattern = re.compile(
            r'    "HWD/LEH":\s+\{[^}]+\},?\s*\n'
        )
        new_teams_line = build_teams_raw_line(winner) + "\n"
        if teams_raw_pattern.search(new_content):
            new_content = teams_raw_pattern.sub(new_teams_line, new_content)
            changes.append(f"_TEAMS_RAW: replaced 'HWD/LEH' entry with '{winner}' data")
        else:
            print(f"  WARNING: Could not find HWD/LEH _TEAMS_RAW entry via regex -- trying literal replace")
            new_content, n = replace_all_occurrences(new_content, f'"HWD/LEH":', f'"{winner}":', "_TEAMS_RAW key")
            if n:
                changes.append(f"_TEAMS_RAW key: 'HWD/LEH' -> '{winner}' ({n} occurrences)")

        # 2. Replace all remaining string occurrences of "HWD/LEH" (BRACKET tuples, comments, market_lines lookups)
        remaining = new_content.count(f'"{placeholder}"') + new_content.count(f"'{placeholder}'")
        if remaining > 0:
            new_content = new_content.replace(f'"{placeholder}"', f'"{winner}"')
            new_content = new_content.replace(f"'{placeholder}'", f"'{winner}'")
            changes.append(f"BRACKET/comments: replaced {remaining} remaining occurrences of '{placeholder}' -> '{winner}'")

        # 3. Update INJURY_ADJ if winner has known injuries (none currently for First Four teams)
        # No-op for now; add entries here if injuries are known at game time.

    if west_winner:
        placeholder = WEST_PLACEHOLDER
        winner = west_winner

        # 1. Replace the _TEAMS_RAW entry for NCST/SMU
        teams_raw_pattern = re.compile(
            r'    "NCST/SMU":\s+\{[^}]+\},?\s*\n'
        )
        new_teams_line = build_teams_raw_line(winner) + "\n"
        if teams_raw_pattern.search(new_content):
            new_content = teams_raw_pattern.sub(new_teams_line, new_content)
            changes.append(f"_TEAMS_RAW: replaced 'NCST/SMU' entry with '{winner}' data")
        else:
            print(f"  WARNING: Could not find NCST/SMU _TEAMS_RAW entry via regex -- trying literal replace")
            new_content, n = replace_all_occurrences(new_content, f'"NCST/SMU":', f'"{winner}":', "_TEAMS_RAW key")
            if n:
                changes.append(f"_TEAMS_RAW key: 'NCST/SMU' -> '{winner}' ({n} occurrences)")

        # 2. Replace all remaining string occurrences of "NCST/SMU"
        remaining = new_content.count(f'"{placeholder}"') + new_content.count(f"'{placeholder}'")
        if remaining > 0:
            new_content = new_content.replace(f'"{placeholder}"', f'"{winner}"')
            new_content = new_content.replace(f"'{placeholder}'", f"'{winner}'")
            changes.append(f"BRACKET/comments: replaced {remaining} remaining occurrences of '{placeholder}' -> '{winner}'")

    return new_content, changes


def main():
    parser = argparse.ArgumentParser(description="Update simulate.py with First Four winners")
    parser.add_argument(
        "--midwest-winner",
        metavar="TEAM",
        help="Winner of the Midwest First Four (Howard or Lehigh)",
    )
    parser.add_argument(
        "--west-winner",
        metavar="TEAM",
        help="Winner of the West First Four (NC State or SMU)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without writing to simulate.py",
    )
    args = parser.parse_args()

    if not args.midwest_winner and not args.west_winner:
        parser.print_help()
        print("\nERROR: Provide at least one of --midwest-winner or --west-winner.")
        sys.exit(1)

    if not SIMULATE_PATH.exists():
        print(f"ERROR: {SIMULATE_PATH} not found.")
        sys.exit(1)

    # Resolve and validate winner names
    midwest_winner = None
    west_winner    = None

    if args.midwest_winner:
        midwest_winner = resolve_winner(args.midwest_winner, MIDWEST_CANDIDATES, "Midwest")
    if args.west_winner:
        west_winner = resolve_winner(args.west_winner, WEST_CANDIDATES, "West")

    # Read current content
    content = SIMULATE_PATH.read_text(encoding="utf-8")

    # Apply changes
    new_content, changes = make_changes(content, midwest_winner, west_winner)

    if not changes:
        print("No changes to apply -- placeholders may already have been replaced.")
        sys.exit(0)

    # Summary
    print(f"simulate.py: {SIMULATE_PATH}")
    print(f"\nChanges to apply ({len(changes)}):")
    for c in changes:
        print(f"  - {c}")

    if args.dry_run:
        print("\n--dry-run: no files written.")

        # Show a diff-style preview of changed lines
        old_lines = content.splitlines()
        new_lines = new_content.splitlines()
        print("\nPreview (changed lines only):")
        for i, (old, new) in enumerate(zip(old_lines, new_lines)):
            if old != new:
                print(f"  line {i+1}:")
                print(f"    - {old}")
                print(f"    + {new}")
        # Lines that exist in new but not old (extra lines from expansion)
        if len(new_lines) != len(old_lines):
            print(f"  (line count changed: {len(old_lines)} -> {len(new_lines)})")
        return

    # Write updated file
    SIMULATE_PATH.write_text(new_content, encoding="utf-8")
    print(f"\nWrote updated simulate.py ({len(changes)} changes applied).")

    if midwest_winner:
        d = TEAM_DATA[midwest_winner]
        print(f"\nMidwest winner: {midwest_winner}  (em={d['em']:+.2f}, seed={d['seed']})")
    if west_winner:
        d = TEAM_DATA[west_winner]
        print(f"West winner:    {west_winner}  (em={d['em']:+.2f}, seed={d['seed']})")

    print("\nNext step: run scripts/pull_tournament_lines.py to refresh market lines, then run_pipeline.py.")


if __name__ == "__main__":
    main()
