"""
Canonical team name resolver for NCAAB Men's Basketball system.

Every data source uses different team name strings:
  - BartTorvik: "St Johns", "UConn", "NCST/SMU" (First Four composite)
  - ESPN: "St. John's (NY)", "Connecticut Huskies"
  - Brackets: "Duke", "Michigan" (clean canonical form)

This module provides a single canonical name for each program so all
cross-source joins work without silent mismatches.

Usage:
    from src.utils.team_names import resolve, NameResolver

    canonical = resolve("St Johns")      # -> "St. John's"
    canonical = resolve("UConn")         # -> "Connecticut"

UnresolvableTeamError is raised (not silently returned None) on unknown names.
The resolver also supports fuzzy fallback with a confidence threshold.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Path to canonical name mapping CSV (committed to git)
_MAP_PATH = Path(__file__).parent.parent.parent / "data" / "team_name_map.csv"

# In-memory cache: any_name_lowercase -> canonical_name
_CACHE: Dict[str, str] = {}
_LOADED = False


class UnresolvableTeamError(ValueError):
    """Raised when a team name cannot be resolved to a canonical form."""
    pass


def _load_map() -> None:
    """Load team_name_map.csv into _CACHE. Called once on first use."""
    global _LOADED
    if _LOADED:
        return

    if not _MAP_PATH.exists():
        logger.warning(f"team_name_map.csv not found at {_MAP_PATH} — resolver will use identity fallback")
        _LOADED = True
        return

    with open(_MAP_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical = row["canonical"].strip()
            if not canonical:
                continue
            # Map every known alias -> canonical (case-insensitive key)
            for col in ("canonical", "torvik_name", "espn_name", "alt_names"):
                val = row.get(col, "").strip()
                if val:
                    # alt_names may be pipe-separated
                    for alias in val.split("|"):
                        alias = alias.strip()
                        if alias:
                            _CACHE[alias.lower()] = canonical

    logger.info(f"team_names: loaded {len(_CACHE)} aliases from {_MAP_PATH}")
    _LOADED = True


def resolve(name: str, fuzzy: bool = False, fuzzy_threshold: int = 97) -> str:
    """
    Resolve any team name variant to its canonical form.

    Fuzzy matching is OFF by default (fuzzy=False).

    NCAA team names have dangerous near-duplicates:
      "Iowa" vs "Iowa St", "Tennessee" vs "Tennessee St",
      "Michigan" vs "Michigan St", "Ohio" vs "Miami OH", etc.
    Fuzzy matching at any reasonable threshold produces false positives
    for these pairs. All known aliases must be in team_name_map.csv.

    To add a new alias: append a row to data/team_name_map.csv and
    call reload(). Do NOT rely on fuzzy to paper over missing entries.

    Args:
        name:             Raw team name from any source
        fuzzy:            Fall back to fuzzy matching (default False — dangerous)
        fuzzy_threshold:  Minimum rapidfuzz ratio score; 97+ required to avoid
                          "Iowa"/"Iowa St" class collisions (default 97)

    Returns:
        Canonical team name string

    Raises:
        UnresolvableTeamError: If name cannot be resolved
    """
    _load_map()

    # Exact match (case-insensitive)
    key = name.strip().lower()
    if key in _CACHE:
        return _CACHE[key]

    # Fuzzy fallback — disabled by default, threshold ≥97 required
    if fuzzy and _CACHE:
        try:
            from rapidfuzz import process as fuzz_process
            candidates = list(_CACHE.keys())
            result = fuzz_process.extractOne(key, candidates, score_cutoff=fuzzy_threshold)
            if result:
                matched_key = result[0]
                canonical = _CACHE[matched_key]
                logger.debug(f"team_names: fuzzy '{name}' -> '{canonical}' (via '{matched_key}', score={result[1]:.0f})")
                return canonical
        except ImportError:
            pass  # rapidfuzz not installed, skip fuzzy

    # Identity fallback with warning (if map not loaded or empty)
    if not _CACHE:
        logger.debug(f"team_names: no map loaded, returning '{name}' as-is")
        return name.strip()

    raise UnresolvableTeamError(
        f"Cannot resolve team name '{name}' — add it to data/team_name_map.csv. "
        f"Columns: canonical, torvik_name, espn_name, alt_names"
    )


def resolve_many(names: list, **kwargs) -> list:
    """Resolve a list of team names. Raises on first unresolvable name."""
    return [resolve(n, **kwargs) for n in names]


def is_resolvable(name: str) -> bool:
    """Return True if name can be resolved without raising."""
    try:
        resolve(name)
        return True
    except UnresolvableTeamError:
        return False


def add_alias(alias: str, canonical: str) -> None:
    """
    Register a temporary in-memory alias (not persisted to CSV).
    Useful for First Four composite names like 'NCST/SMU'.
    """
    _load_map()
    _CACHE[alias.strip().lower()] = canonical
    logger.debug(f"team_names: added runtime alias '{alias}' -> '{canonical}'")


def reload() -> None:
    """Force reload of team_name_map.csv (clears cache first)."""
    global _LOADED
    _CACHE.clear()
    _LOADED = False
    _load_map()
