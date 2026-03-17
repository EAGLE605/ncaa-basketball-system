#!/usr/bin/env python3
"""
2026 NCAA March Madness Monte Carlo Bracket Simulator -- FINAL
Real Data: BartTorvik AdjEM (scraped) + Evan Miya injury adjustments
Model: NormCDF(AdjEM_diff * tempo_factor * TOURNEY_MULT / sigma)
sigma=11.0 baseline | 12.5 for 3pt-heavy matchups
250,000 simulations per game

UPGRADE: populate MARKET_LINES dict with de-vigged Pinnacle closing lines
to enable 65% market / 35% BartTorvik ensemble for maximum accuracy.
De-vig formula: raw_a = |ML|/(|ML|+100) if fav else 100/(ML+100)
               p_market_a = raw_a / (raw_a + raw_b)
"""

import os
import json
import numpy as np
from scipy.stats import norm
from collections import defaultdict

N_SIMS             = 250_000
TOURNEY_MULT       = 1.07   # favorites win more often in tournament vs regular season
SIGMA_BASE         = 11.0
SIGMA_3PT          = 12.5   # bump for 3pt-heavy matchups (higher variance)
MARKET_WEIGHT      = 0.65   # ensemble: market weight when line is available
TORVIK_WEIGHT      = 0.35   # ensemble: BartTorvik weight

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
_TEAMS_RAW = {
    # --- EAST ---
    "Duke":            {"em": 37.35, "tempo": 72, "seed": 1,  "region": "East",    "3pt": True},   # AdjOE 128.2 AdjDE 90.8
    "UConn":           {"em": 28.11, "tempo": 66, "seed": 2,  "region": "East",    "3pt": False},  # AdjOE 123.1 AdjDE 95.0
    "Michigan St":     {"em": 26.73, "tempo": 70, "seed": 3,  "region": "East",    "3pt": False},  # AdjOE 122.9 AdjDE 96.2
    "Kansas":          {"em": 23.46, "tempo": 70, "seed": 4,  "region": "East",    "3pt": True},   # AdjOE 117.8 AdjDE 94.4
    "St Johns":        {"em": 25.71, "tempo": 69, "seed": 5,  "region": "East",    "3pt": False},  # AdjOE 119.8 AdjDE 94.1
    "Louisville":      {"em": 25.87, "tempo": 70, "seed": 6,  "region": "East",    "3pt": False},  # AdjOE 124.0 AdjDE 98.2
    "UCLA":            {"em": 22.77, "tempo": 70, "seed": 7,  "region": "East",    "3pt": False},  # AdjOE 124.6 AdjDE 101.8
    "Ohio St":         {"em": 23.62, "tempo": 70, "seed": 8,  "region": "East",    "3pt": False},  # AdjOE 125.2 AdjDE 101.6
    "TCU":             {"em": 15.21, "tempo": 69, "seed": 9,  "region": "East",    "3pt": False},  # AdjOE 114.9 AdjDE 99.6
    "UCF":             {"em": 14.31, "tempo": 70, "seed": 10, "region": "East",    "3pt": False},  # AdjOE 120.0 AdjDE 105.7
    "South Florida":   {"em":  9.00, "tempo": 68, "seed": 11, "region": "East",    "3pt": False},  # est (below T100)
    "Northern Iowa":   {"em":  6.50, "tempo": 63, "seed": 12, "region": "East",    "3pt": False},  # est -- slow MVC pace
    "Cal Baptist":     {"em":  2.50, "tempo": 68, "seed": 13, "region": "East",    "3pt": False},  # est
    "North Dakota St": {"em":  0.50, "tempo": 68, "seed": 14, "region": "East",    "3pt": False},  # est
    "Furman":          {"em": -3.50, "tempo": 68, "seed": 15, "region": "East",    "3pt": False},  # est
    "Siena":           {"em": -8.00, "tempo": 68, "seed": 16, "region": "East",    "3pt": False},  # est

    # --- WEST ---
    "Arizona":         {"em": 35.54, "tempo": 71, "seed": 1,  "region": "West",    "3pt": False},  # AdjOE 126.9 AdjDE 91.4
    "Purdue":          {"em": 33.06, "tempo": 71, "seed": 2,  "region": "West",    "3pt": True},   # AdjOE 133.3 AdjDE 100.3
    "Gonzaga":         {"em": 26.29, "tempo": 72, "seed": 3,  "region": "West",    "3pt": True},   # AdjOE 120.3 AdjDE 94.0
    "Arkansas":        {"em": 26.29, "tempo": 72, "seed": 4,  "region": "West",    "3pt": True},   # AdjOE 127.9 AdjDE 101.6 -- Acuff 3pt
    "Wisconsin":       {"em": 25.54, "tempo": 68, "seed": 5,  "region": "West",    "3pt": False},  # AdjOE 127.2 AdjDE 101.6
    "BYU":             {"em": 20.43, "tempo": 70, "seed": 6,  "region": "West",    "3pt": False},  # AdjOE 124.8 AdjDE 104.3
    "Miami FL":        {"em": 19.64, "tempo": 70, "seed": 7,  "region": "West",    "3pt": False},  # AdjOE 121.1 AdjDE 101.5
    "Villanova":       {"em": 19.05, "tempo": 66, "seed": 8,  "region": "West",    "3pt": False},  # AdjOE 119.6 AdjDE 100.5
    "Utah St":         {"em": 20.94, "tempo": 67, "seed": 9,  "region": "West",    "3pt": False},  # AdjOE 123.0 AdjDE 102.1
    "Missouri":        {"em": 16.90, "tempo": 70, "seed": 10, "region": "West",    "3pt": False},  # AdjOE 119.9 AdjDE 103.0
    "NCST/SMU":        {"em": 14.35, "tempo": 70, "seed": 11, "region": "West",    "3pt": False},  # First Four: avg(NC St 14.69, SMU 14.00)
    "High Point":      {"em":  5.50, "tempo": 69, "seed": 12, "region": "West",    "3pt": False},  # est
    "Hawaii":          {"em":  2.50, "tempo": 69, "seed": 13, "region": "West",    "3pt": False},  # est
    "Kennesaw St":     {"em":  0.00, "tempo": 68, "seed": 14, "region": "West",    "3pt": False},  # est
    "Queens":          {"em": -3.50, "tempo": 68, "seed": 15, "region": "West",    "3pt": False},  # est
    "LIU":             {"em": -8.00, "tempo": 68, "seed": 16, "region": "West",    "3pt": False},  # est

    # --- MIDWEST ---
    "Michigan":        {"em": 36.61, "tempo": 72, "seed": 1,  "region": "Midwest", "3pt": True},   # AdjOE 127.7 AdjDE 91.0
    "Iowa St":         {"em": 31.14, "tempo": 67, "seed": 2,  "region": "Midwest", "3pt": False},  # AdjOE 123.8 AdjDE 92.6
    "Virginia":        {"em": 26.42, "tempo": 64, "seed": 3,  "region": "Midwest", "3pt": False},  # AdjOE 122.3 AdjDE 95.8 -- slow
    "Alabama":         {"em": 26.71, "tempo": 72, "seed": 4,  "region": "Midwest", "3pt": True},   # AdjOE 129.5 AdjDE 102.8
    "Texas Tech":      {"em": 27.49, "tempo": 70, "seed": 5,  "region": "Midwest", "3pt": False},  # AdjOE 126.3 AdjDE 98.8
    "Tennessee":       {"em": 25.98, "tempo": 70, "seed": 6,  "region": "Midwest", "3pt": False},  # AdjOE 121.5 AdjDE 95.5
    "Kentucky":        {"em": 19.94, "tempo": 71, "seed": 7,  "region": "Midwest", "3pt": False},  # AdjOE 119.9 AdjDE 100.0
    "Georgia":         {"em": 19.59, "tempo": 71, "seed": 8,  "region": "Midwest", "3pt": False},  # AdjOE 124.4 AdjDE 104.8
    "Saint Louis":     {"em": 17.26, "tempo": 62, "seed": 9,  "region": "Midwest", "3pt": False},  # AdjOE 119.5 AdjDE 102.3 -- very slow
    "Santa Clara":     {"em":  7.50, "tempo": 69, "seed": 10, "region": "Midwest", "3pt": False},  # est (WCC)
    "Miami OH":        {"em":  9.00, "tempo": 70, "seed": 11, "region": "Midwest", "3pt": False},  # est (MAC)
    "Akron":           {"em":  8.00, "tempo": 70, "seed": 12, "region": "Midwest", "3pt": True},   # est -- 3 senior guards 37%+ from 3
    "Hofstra":         {"em":  2.50, "tempo": 69, "seed": 13, "region": "Midwest", "3pt": False},  # est
    "Wright St":       {"em":  0.50, "tempo": 68, "seed": 14, "region": "Midwest", "3pt": False},  # est
    "Tennessee St":    {"em": -3.50, "tempo": 68, "seed": 15, "region": "Midwest", "3pt": False},  # est
    "HWD/LEH":         {"em": -8.00, "tempo": 68, "seed": 16, "region": "Midwest", "3pt": False},  # First Four: Howard vs Lehigh

    # --- SOUTH ---
    "Florida":         {"em": 33.82, "tempo": 72, "seed": 1,  "region": "South",   "3pt": False},  # AdjOE 126.1 AdjDE 92.3
    "Houston":         {"em": 32.95, "tempo": 65, "seed": 2,  "region": "South",   "3pt": False},  # AdjOE 125.3 AdjDE 92.4 -- methodical
    "Illinois":        {"em": 33.73, "tempo": 71, "seed": 3,  "region": "South",   "3pt": False},  # AdjOE 131.9 AdjDE 98.2
    "Nebraska":        {"em": 23.71, "tempo": 69, "seed": 4,  "region": "South",   "3pt": False},  # AdjOE 117.2 AdjDE 93.5
    "Vanderbilt":      {"em": 28.11, "tempo": 70, "seed": 5,  "region": "South",   "3pt": False},  # AdjOE 127.5 AdjDE 99.4
    "North Carolina":  {"em": 21.41, "tempo": 70, "seed": 6,  "region": "South",   "3pt": False},  # AdjOE 121.4 AdjDE 100.0
    "Saint Marys":     {"em": 21.82, "tempo": 63, "seed": 7,  "region": "South",   "3pt": False},  # AdjOE 119.4 AdjDE 97.5 -- very slow
    "Clemson":         {"em": 19.56, "tempo": 69, "seed": 8,  "region": "South",   "3pt": False},  # AdjOE 116.6 AdjDE 97.0
    "Iowa":            {"em": 21.92, "tempo": 68, "seed": 9,  "region": "South",   "3pt": False},  # AdjOE 122.1 AdjDE 100.2
    "Texas AM":        {"em": 18.43, "tempo": 69, "seed": 10, "region": "South",   "3pt": False},  # AdjOE 119.9 AdjDE 101.4
    "VCU":             {"em": 16.24, "tempo": 70, "seed": 11, "region": "South",   "3pt": False},  # AdjOE 119.3 AdjDE 103.0
    "McNeese":         {"em":  5.50, "tempo": 69, "seed": 12, "region": "South",   "3pt": False},  # est
    "Troy":            {"em":  2.00, "tempo": 68, "seed": 13, "region": "South",   "3pt": False},  # est
    "Penn":            {"em":  1.00, "tempo": 68, "seed": 14, "region": "South",   "3pt": False},  # est (Ivy)
    "Idaho":           {"em": -3.50, "tempo": 68, "seed": 15, "region": "South",   "3pt": False},  # est
    "Prairie View":    {"em": -8.50, "tempo": 68, "seed": 16, "region": "South",   "3pt": False},  # est
}

# Apply injury adjustments
TEAMS = {}
for name, data in _TEAMS_RAW.items():
    adj = INJURY_ADJ.get(name, 0.0)
    TEAMS[name] = {
        "em":     round(data["em"] + adj, 3),
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
def _torvik_prob(team_a: str, team_b: str) -> float:
    """BartTorvik NormCDF win probability for team_a."""
    a = TEAMS[team_a]
    b = TEAMS[team_b]
    sigma = SIGMA_3PT if (a["3pt"] or b["3pt"]) else SIGMA_BASE
    tempo_factor = (a["tempo"] + b["tempo"]) / 200.0
    point_diff   = (a["em"] - b["em"]) * tempo_factor * TOURNEY_MULT
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


def save_json(adv: dict, bracket: dict, out_path: str):
    output = {
        "model":        "BartTorvik AdjEM + Evan Miya injury adj",
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

    adv     = run_simulation()
    bracket = generate_bracket(adv)
    print_results(adv, bracket)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    save_json(adv, bracket, os.path.join(out_dir, "results_2026.json"))
