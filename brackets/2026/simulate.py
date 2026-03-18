#!/usr/bin/env python3
"""
2026 NCAA March Madness Monte Carlo Bracket Simulator -- ENHANCED
Real Data: BartTorvik AdjEM + AdjOE/AdjDE + injury adjustments
Model: NormCDF(AdjEM_diff * tempo_factor * TOURNEY_MULT / sigma)
sigma=11.0 baseline | 12.5 for 3pt-heavy matchups
250,000 simulations per game

Enhancements vs base version:
  - Luck regression: 30% regression toward true talent (BartTorvik Luck column)
  - Coaching tempo: 60/40 blend toward slower team (controls pace ~60% of games)
  - Coaching style: zone defense (-0.8 AdjEM), press offense (-0.5 opponent)
  - Projected totals: AdjOE/AdjDE Pomeroy interaction model
  - Totals betting signal: flags games where model diverges from market >3 pts

MARKET LINES: populate from Pinnacle for 65% market / 35% BartTorvik ensemble.
Generate with: python scripts/pull_tournament_lines.py
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

# Add project root for src imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Enhanced modules (graceful fallback if not installed)
try:
    from src.model.adjustments import apply_all_adjustments, coaching_tempo_blend
    from src.model.score_model import project_game, print_projection_table
    _ENHANCED = True
except ImportError:
    _ENHANCED = False

N_SIMS             = 250_000
TOURNEY_MULT       = 1.07   # favorites win more often in tournament vs regular season
SIGMA_BASE         = 11.0
SIGMA_3PT          = 12.5   # bump for 3pt-heavy matchups (higher variance)
MARKET_WEIGHT      = 0.65   # ensemble: market weight when line is available
TORVIK_WEIGHT      = 0.35   # ensemble: BartTorvik weight
USE_LUCK_ADJ       = True   # apply 30% luck regression from adjustments module
USE_COACHING_TEMPO = True   # 60/40 slower-team tempo blend

# ---------------------------------------------------------------------------
# MARKET LINES (de-vigged)
# Key: (team_a, team_b) exactly matching BRACKET order
# Value: P(team_a wins), range 0-1
# Populate with Pinnacle closing lines on game day for maximum accuracy.
# ---------------------------------------------------------------------------
MARKET_LINES: dict = {
    # Example (fill in real lines):
    # ("Duke", "Siena"): 0.985,
}

# Auto-load from brackets/2026/market_lines.json if present.
# Generate it with: python scripts/pull_tournament_lines.py
import pathlib as _pathlib
_ml_path = _pathlib.Path(__file__).parent / "market_lines.json"
if _ml_path.exists():
    with open(_ml_path) as _f:
        import json as _json
        _ml_data = _json.load(_f)
    MARKET_LINES = {tuple(k.split(" vs ")): v for k, v in _ml_data.items()}
    print(f"[simulate] Auto-loaded {len(MARKET_LINES)} market lines from market_lines.json")

# Auto-load from brackets/2026/market_lines_totals.json if present.
# Generate it with: python scripts/pull_tournament_lines.py --totals
MARKET_TOTALS: dict = {}
_mt_path = _pathlib.Path(__file__).parent / "market_lines_totals.json"
if _mt_path.exists():
    with open(_mt_path) as _f:
        import json as _json
        _mt_data = _json.load(_f)
    MARKET_TOTALS = dict(_mt_data)  # "Team A vs Team B" -> float
    print(f"[simulate] Auto-loaded {len(MARKET_TOTALS)} market totals from market_lines_totals.json")

# ---------------------------------------------------------------------------
# INJURY ADJUSTMENTS
# Only injuries NOT already reflected in BartTorvik ratings (post-data-cutoff)
# BartTorvik cutoff: approximately Feb 17-22, 2026
# ---------------------------------------------------------------------------
INJURY_ADJ = {
    "Alabama":    -1.20,  # Holloway arrested ~Feb 22 -- NOT in Torvik yet
    "Duke":       -0.66,  # Foster doubtful (most recent, likely post-cutoff)
    "Texas Tech": -0.70,  # Toppin: only partially in data since Feb 17
    "Louisville": -0.50,  # Brown questionable (partial)
    # Already in Torvik ratings (no additional adj needed):
    # BYU (Saunders), UNC (Wilson), Gonzaga (Huff)
}

# ---------------------------------------------------------------------------
# TEAM DATA
# AdjEM = AdjOE - AdjDE from BartTorvik (real scraped values)
# Teams with data_available=False use seed-based estimates
# Tempo = actual D1 possessions per 40 min (Torvik AdjT col is relative rank)
# 3pt = True -> sigma bumped to 12.5 (higher 3pt volume = more variance)
# ---------------------------------------------------------------------------
_NATL_AVG = 107.0  # national average AdjOE/AdjDE (Pomeroy baseline)

def _est_ao(em: float) -> float:
    """Estimate AdjOE from AdjEM when actual value unknown (symmetric split)."""
    return _NATL_AVG + em / 2.0

def _est_ad(em: float) -> float:
    """Estimate AdjDE from AdjEM when actual value unknown (symmetric split)."""
    return _NATL_AVG - em / 2.0

_TEAMS_RAW = {
    # ao = AdjOE (pts per 100 poss), ad = AdjDE (pts allowed per 100 poss)
    # Source: barttorvik.com (real values); "est" = derived from AdjEM
    # --- EAST ---
    "Duke":            {"em": 37.35, "ao": 128.2, "ad":  90.8, "tempo": 72, "seed": 1,  "region": "East",    "3pt": True},
    "UConn":           {"em": 28.11, "ao": 123.1, "ad":  95.0, "tempo": 66, "seed": 2,  "region": "East",    "3pt": False},
    "Michigan St":     {"em": 26.73, "ao": 122.9, "ad":  96.2, "tempo": 70, "seed": 3,  "region": "East",    "3pt": False},
    "Kansas":          {"em": 23.46, "ao": 117.8, "ad":  94.4, "tempo": 70, "seed": 4,  "region": "East",    "3pt": True},
    "St Johns":        {"em": 25.71, "ao": 119.8, "ad":  94.1, "tempo": 69, "seed": 5,  "region": "East",    "3pt": False},
    "Louisville":      {"em": 25.87, "ao": 124.0, "ad":  98.2, "tempo": 70, "seed": 6,  "region": "East",    "3pt": False},
    "UCLA":            {"em": 22.77, "ao": 124.6, "ad": 101.8, "tempo": 70, "seed": 7,  "region": "East",    "3pt": False},
    "Ohio St":         {"em": 23.62, "ao": 125.2, "ad": 101.6, "tempo": 70, "seed": 8,  "region": "East",    "3pt": False},
    "TCU":             {"em": 15.21, "ao": 114.9, "ad":  99.6, "tempo": 69, "seed": 9,  "region": "East",    "3pt": False},
    "UCF":             {"em": 14.31, "ao": 120.0, "ad": 105.7, "tempo": 70, "seed": 10, "region": "East",    "3pt": False},
    "South Florida":   {"em":  9.00, "ao": _est_ao( 9.0), "ad": _est_ad( 9.0), "tempo": 68, "seed": 11, "region": "East",    "3pt": False},
    "Northern Iowa":   {"em":  6.50, "ao": _est_ao( 6.5), "ad": _est_ad( 6.5), "tempo": 63, "seed": 12, "region": "East",    "3pt": False},
    "Cal Baptist":     {"em":  2.50, "ao": _est_ao( 2.5), "ad": _est_ad( 2.5), "tempo": 68, "seed": 13, "region": "East",    "3pt": False},
    "North Dakota St": {"em":  0.50, "ao": _est_ao( 0.5), "ad": _est_ad( 0.5), "tempo": 68, "seed": 14, "region": "East",    "3pt": False},
    "Furman":          {"em": -3.50, "ao": _est_ao(-3.5), "ad": _est_ad(-3.5), "tempo": 68, "seed": 15, "region": "East",    "3pt": False},
    "Siena":           {"em": -8.00, "ao": _est_ao(-8.0), "ad": _est_ad(-8.0), "tempo": 68, "seed": 16, "region": "East",    "3pt": False},

    # --- WEST ---
    "Arizona":         {"em": 35.54, "ao": 126.9, "ad":  91.4, "tempo": 71, "seed": 1,  "region": "West",    "3pt": False},
    "Purdue":          {"em": 33.06, "ao": 133.3, "ad": 100.3, "tempo": 71, "seed": 2,  "region": "West",    "3pt": True},
    "Gonzaga":         {"em": 26.29, "ao": 120.3, "ad":  94.0, "tempo": 72, "seed": 3,  "region": "West",    "3pt": True},
    "Arkansas":        {"em": 26.29, "ao": 127.9, "ad": 101.6, "tempo": 72, "seed": 4,  "region": "West",    "3pt": True},
    "Wisconsin":       {"em": 25.54, "ao": 127.2, "ad": 101.6, "tempo": 68, "seed": 5,  "region": "West",    "3pt": False},
    "BYU":             {"em": 20.43, "ao": 124.8, "ad": 104.3, "tempo": 70, "seed": 6,  "region": "West",    "3pt": False},
    "Miami FL":        {"em": 19.64, "ao": 121.1, "ad": 101.5, "tempo": 70, "seed": 7,  "region": "West",    "3pt": False},
    "Villanova":       {"em": 19.05, "ao": 119.6, "ad": 100.5, "tempo": 66, "seed": 8,  "region": "West",    "3pt": False},
    "Utah St":         {"em": 20.94, "ao": 123.0, "ad": 102.1, "tempo": 67, "seed": 9,  "region": "West",    "3pt": False},
    "Missouri":        {"em": 16.90, "ao": 119.9, "ad": 103.0, "tempo": 70, "seed": 10, "region": "West",    "3pt": False},
    "NCST/SMU":        {"em": 14.35, "ao": _est_ao(14.35), "ad": _est_ad(14.35), "tempo": 70, "seed": 11, "region": "West",    "3pt": False},
    "High Point":      {"em":  5.50, "ao": _est_ao( 5.5), "ad": _est_ad( 5.5), "tempo": 69, "seed": 12, "region": "West",    "3pt": False},
    "Hawaii":          {"em":  2.50, "ao": _est_ao( 2.5), "ad": _est_ad( 2.5), "tempo": 69, "seed": 13, "region": "West",    "3pt": False},
    "Kennesaw St":     {"em":  0.00, "ao": _est_ao( 0.0), "ad": _est_ad( 0.0), "tempo": 68, "seed": 14, "region": "West",    "3pt": False},
    "Queens":          {"em": -3.50, "ao": _est_ao(-3.5), "ad": _est_ad(-3.5), "tempo": 68, "seed": 15, "region": "West",    "3pt": False},
    "LIU":             {"em": -8.00, "ao": _est_ao(-8.0), "ad": _est_ad(-8.0), "tempo": 68, "seed": 16, "region": "West",    "3pt": False},

    # --- MIDWEST ---
    "Michigan":        {"em": 36.61, "ao": 127.7, "ad":  91.0, "tempo": 72, "seed": 1,  "region": "Midwest", "3pt": True},
    "Iowa St":         {"em": 31.14, "ao": 123.8, "ad":  92.6, "tempo": 67, "seed": 2,  "region": "Midwest", "3pt": False},
    "Virginia":        {"em": 26.42, "ao": 122.3, "ad":  95.8, "tempo": 64, "seed": 3,  "region": "Midwest", "3pt": False},
    "Alabama":         {"em": 26.71, "ao": 129.5, "ad": 102.8, "tempo": 72, "seed": 4,  "region": "Midwest", "3pt": True},
    "Texas Tech":      {"em": 27.49, "ao": 126.3, "ad":  98.8, "tempo": 70, "seed": 5,  "region": "Midwest", "3pt": False},
    "Tennessee":       {"em": 25.98, "ao": 121.5, "ad":  95.5, "tempo": 70, "seed": 6,  "region": "Midwest", "3pt": False},
    "Kentucky":        {"em": 19.94, "ao": 119.9, "ad": 100.0, "tempo": 71, "seed": 7,  "region": "Midwest", "3pt": False},
    "Georgia":         {"em": 19.59, "ao": 124.4, "ad": 104.8, "tempo": 71, "seed": 8,  "region": "Midwest", "3pt": False},
    "Saint Louis":     {"em": 17.26, "ao": 119.5, "ad": 102.3, "tempo": 62, "seed": 9,  "region": "Midwest", "3pt": False},
    "Santa Clara":     {"em":  7.50, "ao": _est_ao( 7.5), "ad": _est_ad( 7.5), "tempo": 69, "seed": 10, "region": "Midwest", "3pt": False},
    "Miami OH":        {"em":  9.00, "ao": _est_ao( 9.0), "ad": _est_ad( 9.0), "tempo": 70, "seed": 11, "region": "Midwest", "3pt": False},
    "Akron":           {"em":  8.00, "ao": _est_ao( 8.0), "ad": _est_ad( 8.0), "tempo": 70, "seed": 12, "region": "Midwest", "3pt": True},
    "Hofstra":         {"em":  2.50, "ao": _est_ao( 2.5), "ad": _est_ad( 2.5), "tempo": 69, "seed": 13, "region": "Midwest", "3pt": False},
    "Wright St":       {"em":  0.50, "ao": _est_ao( 0.5), "ad": _est_ad( 0.5), "tempo": 68, "seed": 14, "region": "Midwest", "3pt": False},
    "Tennessee St":    {"em": -3.50, "ao": _est_ao(-3.5), "ad": _est_ad(-3.5), "tempo": 68, "seed": 15, "region": "Midwest", "3pt": False},
    "HWD/LEH":         {"em": -8.00, "ao": _est_ao(-8.0), "ad": _est_ad(-8.0), "tempo": 68, "seed": 16, "region": "Midwest", "3pt": False},

    # --- SOUTH ---
    "Florida":         {"em": 33.82, "ao": 126.1, "ad":  92.3, "tempo": 72, "seed": 1,  "region": "South",   "3pt": False},
    "Houston":         {"em": 32.95, "ao": 125.3, "ad":  92.4, "tempo": 65, "seed": 2,  "region": "South",   "3pt": False},
    "Illinois":        {"em": 33.73, "ao": 131.9, "ad":  98.2, "tempo": 71, "seed": 3,  "region": "South",   "3pt": False},
    "Nebraska":        {"em": 23.71, "ao": 117.2, "ad":  93.5, "tempo": 69, "seed": 4,  "region": "South",   "3pt": False},
    "Vanderbilt":      {"em": 28.11, "ao": 127.5, "ad":  99.4, "tempo": 70, "seed": 5,  "region": "South",   "3pt": False},
    "North Carolina":  {"em": 21.41, "ao": 121.4, "ad": 100.0, "tempo": 70, "seed": 6,  "region": "South",   "3pt": False},
    "Saint Marys":     {"em": 21.82, "ao": 119.4, "ad":  97.5, "tempo": 63, "seed": 7,  "region": "South",   "3pt": False},
    "Clemson":         {"em": 19.56, "ao": 116.6, "ad":  97.0, "tempo": 69, "seed": 8,  "region": "South",   "3pt": False},
    "Iowa":            {"em": 21.92, "ao": 122.1, "ad": 100.2, "tempo": 68, "seed": 9,  "region": "South",   "3pt": False},
    "Texas AM":        {"em": 18.43, "ao": 119.9, "ad": 101.4, "tempo": 69, "seed": 10, "region": "South",   "3pt": False},
    "VCU":             {"em": 16.24, "ao": 119.3, "ad": 103.0, "tempo": 70, "seed": 11, "region": "South",   "3pt": False},
    "McNeese":         {"em":  5.50, "ao": _est_ao( 5.5), "ad": _est_ad( 5.5), "tempo": 69, "seed": 12, "region": "South",   "3pt": False},
    "Troy":            {"em":  2.00, "ao": _est_ao( 2.0), "ad": _est_ad( 2.0), "tempo": 68, "seed": 13, "region": "South",   "3pt": False},
    "Penn":            {"em":  1.00, "ao": _est_ao( 1.0), "ad": _est_ad( 1.0), "tempo": 68, "seed": 14, "region": "South",   "3pt": False},
    "Idaho":           {"em": -3.50, "ao": _est_ao(-3.5), "ad": _est_ad(-3.5), "tempo": 68, "seed": 15, "region": "South",   "3pt": False},
    "Prairie View":    {"em": -8.50, "ao": _est_ao(-8.5), "ad": _est_ad(-8.5), "tempo": 68, "seed": 16, "region": "South",   "3pt": False},
}

# Apply injury adjustments; carry through AdjOE/AdjDE for score model
TEAMS = {}
for name, data in _TEAMS_RAW.items():
    inj = INJURY_ADJ.get(name, 0.0)
    em_inj = round(data["em"] + inj, 3)
    TEAMS[name] = {
        "em":     em_inj,
        "ao":     round(data["ao"] + inj / 2.0, 2),  # injury splits ~50/50 offensively
        "ad":     round(data["ad"] - inj / 2.0, 2),  # defensive impact of injury
        "tempo":  data["tempo"],
        "seed":   data["seed"],
        "region": data["region"],
        "3pt":    data["3pt"],
    }

# ---------------------------------------------------------------------------
# BRACKET STRUCTURE
# Standard NCAA order per region: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
# R32 pairs: (0,1), (2,3), (4,5), (6,7)
# S16 pairs: (R32[0],R32[1]), (R32[2],R32[3])
# E8: S16[0] vs S16[1]
# ---------------------------------------------------------------------------
BRACKET = {
    "East": [
        ("Duke",         "Siena"),           # 1 vs 16
        ("Ohio St",      "TCU"),             # 8 vs 9
        ("St Johns",     "Northern Iowa"),   # 5 vs 12
        ("Kansas",       "Cal Baptist"),     # 4 vs 13
        ("Louisville",   "South Florida"),   # 6 vs 11
        ("Michigan St",  "North Dakota St"), # 3 vs 14
        ("UCLA",         "UCF"),             # 7 vs 10
        ("UConn",        "Furman"),          # 2 vs 15
    ],
    "West": [
        ("Arizona",      "LIU"),             # 1 vs 16
        ("Villanova",    "Utah St"),         # 8 vs 9
        ("Wisconsin",    "High Point"),      # 5 vs 12
        ("Arkansas",     "Hawaii"),          # 4 vs 13
        ("BYU",          "NCST/SMU"),        # 6 vs 11  (NCST/SMU = First Four winner)
        ("Gonzaga",      "Kennesaw St"),     # 3 vs 14
        ("Miami FL",     "Missouri"),        # 7 vs 10
        ("Purdue",       "Queens"),          # 2 vs 15
    ],
    "Midwest": [
        ("Michigan",     "HWD/LEH"),         # 1 vs 16  (HWD/LEH = Howard/Lehigh First Four)
        ("Georgia",      "Saint Louis"),     # 8 vs 9
        ("Texas Tech",   "Akron"),           # 5 vs 12
        ("Alabama",      "Hofstra"),         # 4 vs 13
        ("Tennessee",    "Miami OH"),        # 6 vs 11
        ("Virginia",     "Wright St"),       # 3 vs 14
        ("Kentucky",     "Santa Clara"),     # 7 vs 10
        ("Iowa St",      "Tennessee St"),    # 2 vs 15
    ],
    "South": [
        ("Florida",      "Prairie View"),    # 1 vs 16
        ("Clemson",      "Iowa"),            # 8 vs 9
        ("Vanderbilt",   "McNeese"),         # 5 vs 12
        ("Nebraska",     "Troy"),            # 4 vs 13
        ("North Carolina","VCU"),            # 6 vs 11
        ("Illinois",     "Penn"),            # 3 vs 14
        ("Saint Marys",  "Texas AM"),        # 7 vs 10
        ("Houston",      "Idaho"),           # 2 vs 15
    ],
}

REGION_ORDER = ["East", "West", "Midwest", "South"]
ALL_TEAMS    = list(TEAMS.keys())

# ---------------------------------------------------------------------------
# WIN PROBABILITY
# ---------------------------------------------------------------------------
def _effective_tempo(a: dict, b: dict) -> float:
    """
    60/40 weighted blend toward slower team (coaching tempo research).
    Falls back to arithmetic mean if enhanced module not available.
    """
    if USE_COACHING_TEMPO and _ENHANCED:
        return coaching_tempo_blend(a["tempo"], b["tempo"])
    return (a["tempo"] + b["tempo"]) / 2.0


def _adjusted_ems(team_a: str, team_b: str):
    """
    Apply luck regression and coaching style adjustments to AdjEM.
    Returns (em_a, em_b) after all pre-game adjustments.
    """
    a = TEAMS[team_a]
    b = TEAMS[team_b]
    if USE_LUCK_ADJ and _ENHANCED:
        em_a, em_b = apply_all_adjustments(team_a, a["em"], team_b, b["em"])
    else:
        em_a, em_b = a["em"], b["em"]
    return em_a, em_b


def _torvik_prob(team_a: str, team_b: str) -> float:
    """BartTorvik NormCDF win probability for team_a (with enhancements)."""
    a = TEAMS[team_a]
    b = TEAMS[team_b]
    sigma        = SIGMA_3PT if (a["3pt"] or b["3pt"]) else SIGMA_BASE
    eff_tempo    = _effective_tempo(a, b)
    tempo_factor = eff_tempo / 100.0
    em_a, em_b   = _adjusted_ems(team_a, team_b)
    point_diff   = (em_a - em_b) * tempo_factor * TOURNEY_MULT
    return float(norm.cdf(point_diff / sigma))


def win_prob(team_a: str, team_b: str) -> float:
    """Ensemble win probability: 65% market + 35% BartTorvik when line available."""
    p_torvik = _torvik_prob(team_a, team_b)
    key_ab = (team_a, team_b)
    key_ba = (team_b, team_a)
    if key_ab in MARKET_LINES:
        p_market = MARKET_LINES[key_ab]
        return MARKET_WEIGHT * p_market + TORVIK_WEIGHT * p_torvik
    if key_ba in MARKET_LINES:
        p_market = 1.0 - MARKET_LINES[key_ba]
        return MARKET_WEIGHT * p_market + TORVIK_WEIGHT * p_torvik
    return p_torvik


# Precompute win probability cache for all possible matchups
def build_wp_cache() -> dict:
    cache = {}
    for ta in ALL_TEAMS:
        for tb in ALL_TEAMS:
            if ta != tb:
                cache[(ta, tb)] = win_prob(ta, tb)
    return cache


# ---------------------------------------------------------------------------
# SIMULATION  (fully vectorized numpy -- runs in seconds not minutes)
# Strategy: map teams -> int IDs, build WP_matrix numpy array,
#           then simulate each round with advanced numpy indexing.
# ---------------------------------------------------------------------------
def run_simulation() -> dict:
    print(f"Building win probability cache (vectorized)...")

    # Map team name -> integer ID
    team_ids  = {t: i for i, t in enumerate(ALL_TEAMS)}
    N_TEAMS   = len(ALL_TEAMS)

    # WP_matrix[i, j] = P(team_i beats team_j)
    WP_matrix = np.zeros((N_TEAMS, N_TEAMS), dtype=np.float32)
    for ta in ALL_TEAMS:
        for tb in ALL_TEAMS:
            if ta != tb:
                WP_matrix[team_ids[ta], team_ids[tb]] = win_prob(ta, tb)

    # Flatten bracket into ordered list of (team_a_id, team_b_id) per region
    # Shape: 4 regions x 8 games = 32 R64 games total, in REGION_ORDER
    r64_pairs = []   # list of (ta_id, tb_id) for 32 R64 games
    for region in REGION_ORDER:
        for ta, tb in BRACKET[region]:
            r64_pairs.append((team_ids[ta], team_ids[tb]))

    print(f"Running {N_SIMS:,} vectorized simulations...")
    rng = np.random.default_rng(2026)

    # Advancement counts: shape (N_TEAMS, 6) -- rounds 0-5
    adv_np = np.zeros((N_TEAMS, 6), dtype=np.int64)

    # ----- ROUND OF 64 (all 32 games fixed) -----
    r64_ta = np.array([p[0] for p in r64_pairs], dtype=np.int32)   # shape (32,)
    r64_tb = np.array([p[1] for p in r64_pairs], dtype=np.int32)
    r64_p  = WP_matrix[r64_ta, r64_tb]                              # shape (32,)
    r64_rand = rng.random((N_SIMS, 32), dtype=np.float32)           # shape (N_SIMS, 32)
    # r64_winners[i, g] = team_id of winner of game g in sim i
    r64_w = np.where(r64_rand < r64_p[np.newaxis, :], r64_ta, r64_tb)  # (N_SIMS, 32)

    # Count R64 wins
    for g in range(32):
        np.add.at(adv_np[:, 0], r64_w[:, g], 1)

    # ----- ROUND OF 32 (16 games, per-region grouping) -----
    # Each R32 game pairs R64 slots: (0,1),(2,3),(4,5),(6,7) per region
    # Across 4 regions: game indices 0-7 (East), 8-15 (West), 16-23 (Midwest), 24-31 (South)
    r32_rand = rng.random((N_SIMS, 16), dtype=np.float32)
    r32_w    = np.empty((N_SIMS, 16), dtype=np.int32)
    for g in range(16):
        region_base = (g // 4) * 8   # 0,8,16,24 for each region
        slot_in_region = (g % 4) * 2  # 0,2,4,6
        left  = r64_w[:, region_base + slot_in_region]
        right = r64_w[:, region_base + slot_in_region + 1]
        p_arr = WP_matrix[left, right]
        r32_w[:, g] = np.where(r32_rand[:, g] < p_arr, left, right)
    for g in range(16):
        np.add.at(adv_np[:, 1], r32_w[:, g], 1)

    # ----- SWEET 16 (8 games) -----
    # Each S16 game pairs R32 slots: (0,1),(2,3) per region (4 regions = 8 games)
    s16_rand = rng.random((N_SIMS, 8), dtype=np.float32)
    s16_w    = np.empty((N_SIMS, 8), dtype=np.int32)
    for g in range(8):
        region_idx = g // 2              # 0,1,2,3
        slot_in_region = (g % 2) * 2    # 0,2
        r32_base = region_idx * 4
        left  = r32_w[:, r32_base + slot_in_region]
        right = r32_w[:, r32_base + slot_in_region + 1]
        p_arr = WP_matrix[left, right]
        s16_w[:, g] = np.where(s16_rand[:, g] < p_arr, left, right)
    for g in range(8):
        np.add.at(adv_np[:, 2], s16_w[:, g], 1)

    # ----- ELITE 8 (4 games) -----
    # Each E8 game pairs S16 slots: (0,1) per region
    e8_rand = rng.random((N_SIMS, 4), dtype=np.float32)
    e8_w    = np.empty((N_SIMS, 4), dtype=np.int32)
    for g in range(4):
        s16_base = g * 2
        left  = s16_w[:, s16_base]
        right = s16_w[:, s16_base + 1]
        p_arr = WP_matrix[left, right]
        e8_w[:, g] = np.where(e8_rand[:, g] < p_arr, left, right)
    for g in range(4):
        np.add.at(adv_np[:, 3], e8_w[:, g], 1)

    # ----- FINAL FOUR (2 games: East/West, Midwest/South) -----
    ff_rand = rng.random((N_SIMS, 2), dtype=np.float32)
    # Game 0: East(0) vs West(1)
    ff1_p = WP_matrix[e8_w[:, 0], e8_w[:, 1]]
    ff1_w = np.where(ff_rand[:, 0] < ff1_p, e8_w[:, 0], e8_w[:, 1])
    # Game 1: Midwest(2) vs South(3)
    ff2_p = WP_matrix[e8_w[:, 2], e8_w[:, 3]]
    ff2_w = np.where(ff_rand[:, 1] < ff2_p, e8_w[:, 2], e8_w[:, 3])
    np.add.at(adv_np[:, 4], ff1_w, 1)
    np.add.at(adv_np[:, 4], ff2_w, 1)

    # ----- CHAMPIONSHIP -----
    champ_rand = rng.random(N_SIMS, dtype=np.float32)
    champ_p    = WP_matrix[ff1_w, ff2_w]
    champ_w    = np.where(champ_rand < champ_p, ff1_w, ff2_w)
    np.add.at(adv_np[:, 5], champ_w, 1)

    # Convert back to name-keyed dict
    adv = {team: list(adv_np[team_ids[team]]) for team in ALL_TEAMS}

    print(f"Done. {N_SIMS:,} simulations complete.\n")
    return adv


# ---------------------------------------------------------------------------
# GENERATE OPTIMAL BRACKET
# Picks highest-probability team at each position using simulation rates
# ---------------------------------------------------------------------------
def generate_bracket(adv: dict) -> dict:
    picks = {}

    # R64: simple win prob
    r64 = {}
    for region, games in BRACKET.items():
        r64[region] = []
        for ta, tb in games:
            p = win_prob(ta, tb)
            w = ta if p >= 0.5 else tb
            r64[region].append(w)
    picks["R64"] = r64

    # R32
    r32 = {}
    for region in REGION_ORDER:
        r32[region] = []
        r64w = r64[region]
        for i in range(0, 8, 2):
            ta, tb = r64w[i], r64w[i+1]
            w = ta if adv[ta][1] >= adv[tb][1] else tb
            r32[region].append(w)
    picks["R32"] = r32

    # S16
    s16 = {}
    for region in REGION_ORDER:
        s16[region] = []
        r32w = r32[region]
        for i in range(0, 4, 2):
            ta, tb = r32w[i], r32w[i+1]
            w = ta if adv[ta][2] >= adv[tb][2] else tb
            s16[region].append(w)
    picks["S16"] = s16

    # E8
    e8 = {}
    for region in REGION_ORDER:
        ta, tb = s16[region][0], s16[region][1]
        e8[region] = ta if adv[ta][3] >= adv[tb][3] else tb
    picks["E8"] = e8

    # FF
    ff1_a, ff1_b = e8["East"],    e8["West"]
    ff2_a, ff2_b = e8["Midwest"], e8["South"]
    ff1 = ff1_a if adv[ff1_a][4] >= adv[ff1_b][4] else ff1_b
    ff2 = ff2_a if adv[ff2_a][4] >= adv[ff2_b][4] else ff2_b
    picks["FF"] = {"EastWest": ff1, "MidwestSouth": ff2}

    # Champion
    champ = ff1 if adv[ff1][5] >= adv[ff2][5] else ff2
    picks["Champion"] = champ

    return picks


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
def compute_projected_totals(bracket: dict) -> list:
    """
    Project scores and totals for all R64 matchups.
    Returns list of project_game() dicts (one per game).
    """
    if not _ENHANCED:
        return []

    projections = []
    for region, games in BRACKET.items():
        for ta, tb in games:
            a = TEAMS[ta]
            b = TEAMS[tb]
            # Look up market total from MARKET_TOTALS (try both key orderings)
            market_total = (
                MARKET_TOTALS.get(f"{ta} vs {tb}")
                or MARKET_TOTALS.get(f"{tb} vs {ta}")
            )
            proj = project_game(
                adjOE_a=a["ao"],
                adjDE_a=a["ad"],
                adjOE_b=b["ao"],
                adjDE_b=b["ad"],
                tempo_a=a["tempo"],
                tempo_b=b["tempo"],
                team_a=ta,
                team_b=tb,
                market_total=market_total,
            )
            proj["region"] = region
            projections.append(proj)
    return projections


def print_results(adv: dict, bracket: dict):
    print("=" * 70)
    print("CHAMPIONSHIP PROBABILITIES  (BartTorvik real data + injury adj)")
    print("=" * 70)
    top = sorted(ALL_TEAMS, key=lambda t: -adv[t][5])
    for team in top:
        prob = adv[team][5] / N_SIMS
        if prob < 0.001:
            continue
        t    = TEAMS[team]
        line = f"  {team:22s}  {t['region']:7s}  #{t['seed']:2d}  AdjEM={t['em']:+6.2f}"
        if team in INJURY_ADJ:
            line += f"  [inj {INJURY_ADJ[team]:+.2f}]"
        line += f"  {prob*100:5.1f}%"
        print(line)

    print()
    print("=" * 70)
    print("ROUND-BY-ROUND ADVANCEMENT PROBABILITIES")
    print("=" * 70)
    print(f"{'Team':22s} {'Sd':>3} {'R64':>6} {'R32':>6} {'S16':>6} {'E8':>6} {'FF':>6} {'CHAMP':>6}")
    print("-" * 70)
    for team in sorted(ALL_TEAMS, key=lambda t: -adv[t][5]):
        if adv[team][5] == 0 and adv[team][4] == 0:
            continue
        s  = TEAMS[team]["seed"]
        p  = [adv[team][r] / N_SIMS * 100 for r in range(6)]
        print(f"{team:22s} {s:>3}  {p[0]:5.1f}%  {p[1]:5.1f}%  {p[2]:5.1f}%  {p[3]:5.1f}%  {p[4]:5.1f}%  {p[5]:5.1f}%")

    print()
    for person in ["BRADY", "STEVIE", "STEPH"]:
        print("=" * 70)
        print(f"OPTIMAL BRACKET -- {person}  (highest-probability picks)")
        print("=" * 70)

        for region in REGION_ORDER:
            print(f"\n  -- {region.upper()} REGION --")
            r64w  = bracket["R64"][region]
            r32w  = bracket["R32"][region]
            s16w  = bracket["S16"][region]
            e8w   = bracket["E8"][region]

            print(f"  Round of 64:  {', '.join(r64w)}")
            print(f"  Round of 32:  {', '.join(r32w)}")
            print(f"  Sweet 16:     {', '.join(s16w)}")
            print(f"  Elite 8:      {e8w}   ({adv[e8w][3]/N_SIMS*100:.1f}% made E8)")

        ff1 = bracket["FF"]["EastWest"]
        ff2 = bracket["FF"]["MidwestSouth"]
        champ = bracket["Champion"]
        print(f"\n  Final Four:   {ff1} (East/West)    vs    {ff2} (Midwest/South)")
        print(f"\n  *** CHAMPION: {champ}  ({adv[champ][5]/N_SIMS*100:.1f}% win rate in simulation) ***")
        print()

    # Upset alerts: games where lower seed wins in bracket
    print("=" * 70)
    print("UPSET ALERTS  (bracket picks where lower seed wins)")
    print("=" * 70)
    upset_found = False
    for region, games in BRACKET.items():
        r64_picks = bracket["R64"][region]
        for idx, (ta, tb) in enumerate(games):
            pick = r64_picks[idx]
            pick_seed   = TEAMS[pick]["seed"]
            other       = tb if pick == ta else ta
            other_seed  = TEAMS[other]["seed"]
            if pick_seed > other_seed:
                p = adv[pick][0] / N_SIMS * 100
                print(f"  R64  {region:7s}  #{pick_seed} {pick} over #{other_seed} {other}  ({p:.1f}% sim)")
                upset_found = True
    if not upset_found:
        print("  No R64 upsets -- all higher seeds picked to win Round 1")

    # Projected totals / score model output
    projections = compute_projected_totals(bracket)
    if projections:
        print()
        print("=" * 90)
        print("PROJECTED SCORES & TOTALS  (Pomeroy AdjOE/AdjDE interaction model)")
        mode = ""
        if USE_LUCK_ADJ:
            mode += "Luck-adj "
        if USE_COACHING_TEMPO:
            mode += "| 60/40 tempo blend"
        print(f"  {mode.strip() or 'Base model'}")
        print("=" * 90)

        # Sort by region + seeding order
        region_order_map = {r: i for i, r in enumerate(REGION_ORDER)}
        projections.sort(key=lambda p: (region_order_map.get(p.get("region", ""), 99),
                                        TEAMS.get(p["team_a"], {}).get("seed", 99)))

        print_projection_table(projections)

        # Highlight extreme totals (potential betting signals)
        # HIGH TOTAL: both teams must be close (margin <20 pts = competitive game)
        # The "71% under" finding applies to games where MARKET sets high total,
        # which only happens when both teams are roughly evenly matched high-scorers.
        high_total = [p for p in projections if p["total"] > 155.0 and abs(p["spread"]) < 20.0]
        low_total  = [p for p in projections if p["total"] < 130.0]
        if high_total:
            print()
            print("  HIGH TOTAL ALERT (>155 pts, close game -- mkt likely high, 71% UNDER historically):")
            for p in high_total:
                print(f"    {p['team_a']} vs {p['team_b']}: {p['total']:.0f} proj total  (spread={p['spread']:+.0f})")
        if low_total:
            print()
            print("  LOW TOTAL ALERT (<130 pts projected -- slow pace / defensive matchup):")
            for p in low_total:
                print(f"    {p['team_a']} vs {p['team_b']}: {p['total']:.0f} proj total  ({p['possessions']:.0f} poss)")


def save_json(adv: dict, bracket: dict, out_path: str, projections: list = None):
    enhancements = []
    if USE_LUCK_ADJ and _ENHANCED:
        enhancements.append("luck_regression_30pct")
    if USE_COACHING_TEMPO and _ENHANCED:
        enhancements.append("coaching_tempo_60_40_blend")
    enhancements.append("coaching_style_zone_press")

    output = {
        "model":        "BartTorvik AdjEM + Evan Miya injury adj",
        "enhancements": enhancements,
        "sims":         N_SIMS,
        "market_lines": len(MARKET_LINES),
        "champion":     bracket["Champion"],
        "final_four":   [bracket["FF"]["EastWest"], bracket["FF"]["MidwestSouth"]],
        "elite_eight":  bracket["E8"],
        "sweet_sixteen": bracket["S16"],
        "round_of_32":  bracket["R32"],
        "round_of_64":  bracket["R64"],
        "championship_odds": {
            t: f"{adv[t][5]/N_SIMS*100:.2f}%"
            for t in sorted(ALL_TEAMS, key=lambda x: -adv[x][5])
        },
        "full_advancement": {
            t: {
                "AdjEM":    TEAMS[t]["em"],
                "AdjOE":    TEAMS[t]["ao"],
                "AdjDE":    TEAMS[t]["ad"],
                "seed":     TEAMS[t]["seed"],
                "region":   TEAMS[t]["region"],
                "R64":      f"{adv[t][0]/N_SIMS*100:.1f}%",
                "R32":      f"{adv[t][1]/N_SIMS*100:.1f}%",
                "S16":      f"{adv[t][2]/N_SIMS*100:.1f}%",
                "E8":       f"{adv[t][3]/N_SIMS*100:.1f}%",
                "FF":       f"{adv[t][4]/N_SIMS*100:.1f}%",
                "Champion": f"{adv[t][5]/N_SIMS*100:.1f}%",
            }
            for t in sorted(ALL_TEAMS, key=lambda x: -adv[x][5])
        },
    }
    if projections:
        output["projected_totals"] = [
            {
                "matchup":     f"{p['team_a']} vs {p['team_b']}",
                "region":      p.get("region", ""),
                "score":       f"{p['score_a']:.0f}-{p['score_b']:.0f}",
                "total":       p["total"],
                "spread":      p["spread"],
                "possessions": p["possessions"],
            }
            for p in projections
        ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print("2026 NCAA March Madness Simulator -- FINAL")
    print(f"Real BartTorvik AdjEM data | {N_SIMS:,} simulations")
    print(f"Market lines loaded: {len(MARKET_LINES)} (0 = pure BartTorvik mode)")
    print()

    adv         = run_simulation()
    bracket     = generate_bracket(adv)
    print_results(adv, bracket)
    projections = compute_projected_totals(bracket)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    save_json(adv, bracket, os.path.join(out_dir, "results_2026.json"), projections)
