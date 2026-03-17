"""
Vectorized NCAA Men's Basketball Tournament Monte Carlo simulator.

Runs 250k simulations in ~3-5 seconds using numpy advanced indexing.
All 64 teams (post-First-Four) mapped to integer IDs; win probabilities
stored in an NxN float32 matrix for O(1) vectorized lookup per round.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Standard NCAA bracket seed order within each region
# Slot indices: 0=1-seed, 1=16, 2=8, 3=9, 4=5, 5=12, 6=4, 7=13,
#               8=6, 9=11, 10=3, 11=14, 12=7, 13=10, 14=2, 15=15
SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

# R64 pairs by slot index within each region
R64_SLOT_PAIRS = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)]

# R32 pairs (winners of R64 games)
R32_SLOT_PAIRS = [(0,1),(2,3),(4,5),(6,7)]

# Sweet 16 pairs
S16_SLOT_PAIRS = [(0,1),(2,3)]

# Elite 8 pair
E8_SLOT_PAIR   = (0,1)

# Region order: East=0, West=1, Midwest=2, South=3
# Final Four: East vs West (slots 0,1), Midwest vs South (slots 2,3)
FF_PAIRS = [(0,1),(2,3)]


def _build_wp_matrix(
    teams: List[str],
    wp_table: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """
    Convert {team: {team: p}} lookup to flat NxN float32 numpy matrix.
    wp_matrix[i][j] = P(teams[i] beats teams[j])
    """
    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}
    mat = np.zeros((n, n), dtype=np.float32)
    for a, probs in wp_table.items():
        if a not in team_idx:
            continue
        i = team_idx[a]
        for b, p in probs.items():
            if b not in team_idx:
                continue
            j = team_idx[b]
            mat[i, j] = p
    return mat


def simulate_tournament(
    bracket: Dict[str, List[str]],
    wp_table: Dict[str, Dict[str, float]],
    n_sims: int = 250_000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Run N Monte Carlo simulations of the NCAA Men's Basketball Tournament.

    Args:
        bracket:  {region: [team_name x 16]} — teams in SEED_ORDER positions.
                  Regions: "East", "West", "Midwest", "South"
        wp_table: {team_a: {team_b: p(a beats b)}} from win_probability.build_wp_table()
        n_sims:   Number of simulations (250k recommended)
        seed:     RNG seed for reproducibility

    Returns:
        {team: {
            "R64": float,       # P(advances past R64)
            "R32": float,       # P(advances past R32 / reaches Sweet 16)
            "S16": float,       # P(advances past Sweet 16 / reaches Elite 8)
            "E8": float,        # P(advances past Elite 8 / reaches Final Four)
            "FF": float,        # P(reaches Final Four)
            "F": float,         # P(reaches Championship game)
            "champion": float,  # P(wins Championship)
        }}
    """
    # Flatten bracket to ordered list: [East[0..15], West[0..15], Midwest[0..15], South[0..15]]
    regions_order = ["East", "West", "Midwest", "South"]
    all_teams: List[str] = []
    for reg in regions_order:
        all_teams.extend(bracket[reg])

    n_teams  = len(all_teams)  # 64
    team_idx = {t: i for i, t in enumerate(all_teams)}

    wp_matrix = _build_wp_matrix(all_teams, wp_table)

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Round of 64 — all matchups are fixed
    # ------------------------------------------------------------------
    r64_pairs_idx: List[Tuple[int, int]] = []
    for reg_i in range(4):
        base = reg_i * 16
        for slot_a, slot_b in R64_SLOT_PAIRS:
            r64_pairs_idx.append((base + slot_a, base + slot_b))

    # 32 games, fixed
    ta_arr = np.array([p[0] for p in r64_pairs_idx], dtype=np.int32)
    tb_arr = np.array([p[1] for p in r64_pairs_idx], dtype=np.int32)
    r64_p  = wp_matrix[ta_arr, tb_arr]                      # shape (32,)
    r64_rand = rng.random((n_sims, 32), dtype=np.float32)
    r64_w    = np.where(r64_rand < r64_p[np.newaxis, :],
                        ta_arr[np.newaxis, :],
                        tb_arr[np.newaxis, :])               # shape (N, 32)

    # ------------------------------------------------------------------
    # Round of 32 — 16 games, matchup determined per-sim
    # ------------------------------------------------------------------
    r32_w    = np.empty((n_sims, 16), dtype=np.int32)
    r32_rand = rng.random((n_sims, 16), dtype=np.float32)

    for reg_i in range(4):
        base = reg_i * 8  # 8 R64 games per region, 4 R32 games per region
        for gi, (slot_a, slot_b) in enumerate(R32_SLOT_PAIRS):
            game_idx = reg_i * 4 + gi
            left  = r64_w[:, base + slot_a]    # winner of R64 game slot_a
            right = r64_w[:, base + slot_b]
            p_arr = wp_matrix[left, right]
            r32_w[:, game_idx] = np.where(r32_rand[:, game_idx] < p_arr, left, right)

    # ------------------------------------------------------------------
    # Sweet 16 — 8 games
    # ------------------------------------------------------------------
    s16_w    = np.empty((n_sims, 8), dtype=np.int32)
    s16_rand = rng.random((n_sims, 8), dtype=np.float32)

    for reg_i in range(4):
        r32_base = reg_i * 4
        for gi, (slot_a, slot_b) in enumerate(S16_SLOT_PAIRS):
            game_idx = reg_i * 2 + gi
            left  = r32_w[:, r32_base + slot_a]
            right = r32_w[:, r32_base + slot_b]
            p_arr = wp_matrix[left, right]
            s16_w[:, game_idx] = np.where(s16_rand[:, game_idx] < p_arr, left, right)

    # ------------------------------------------------------------------
    # Elite 8 — 4 games (one per region)
    # ------------------------------------------------------------------
    e8_w    = np.empty((n_sims, 4), dtype=np.int32)
    e8_rand = rng.random((n_sims, 4), dtype=np.float32)

    for reg_i in range(4):
        s16_base = reg_i * 2
        left  = s16_w[:, s16_base]
        right = s16_w[:, s16_base + 1]
        p_arr = wp_matrix[left, right]
        e8_w[:, reg_i] = np.where(e8_rand[:, reg_i] < p_arr, left, right)

    # ------------------------------------------------------------------
    # Final Four — 2 games (East vs West, Midwest vs South)
    # ------------------------------------------------------------------
    ff_w    = np.empty((n_sims, 2), dtype=np.int32)
    ff_rand = rng.random((n_sims, 2), dtype=np.float32)

    for gi, (reg_a, reg_b) in enumerate(FF_PAIRS):
        left  = e8_w[:, reg_a]
        right = e8_w[:, reg_b]
        p_arr = wp_matrix[left, right]
        ff_w[:, gi] = np.where(ff_rand[:, gi] < p_arr, left, right)

    # ------------------------------------------------------------------
    # Championship
    # ------------------------------------------------------------------
    champ_rand = rng.random(n_sims, dtype=np.float32)
    left  = ff_w[:, 0]
    right = ff_w[:, 1]
    p_arr = wp_matrix[left, right]
    champ_w = np.where(champ_rand < p_arr, left, right)

    # ------------------------------------------------------------------
    # Aggregate probabilities
    # ------------------------------------------------------------------
    results: Dict[str, Dict[str, float]] = {t: {
        "R64":      0.0,
        "R32":      0.0,
        "S16":      0.0,
        "E8":       0.0,
        "FF":       0.0,
        "F":        0.0,
        "champion": 0.0,
    } for t in all_teams}

    # Count wins per team per round
    for team, i in team_idx.items():
        results[team]["R64"]      = float(np.mean(np.any(r64_w == i, axis=1)))
        results[team]["R32"]      = float(np.mean(np.any(r32_w == i, axis=1)))
        results[team]["S16"]      = float(np.mean(np.any(s16_w == i, axis=1)))
        results[team]["E8"]       = float(np.mean(np.any(e8_w  == i, axis=1)))
        results[team]["FF"]       = float(np.mean(np.any(e8_w  == i, axis=1)))  # same as making FF
        results[team]["F"]        = float(np.mean(np.any(ff_w  == i, axis=1)))
        results[team]["champion"] = float(np.mean(champ_w == i))

    logger.info(
        f"Simulated {n_sims:,} tournaments. "
        f"Champion: {max(results, key=lambda t: results[t]['champion'])} "
        f"({max(results[t]['champion'] for t in results):.1%})"
    )
    return results


def top_n(
    results: Dict[str, Dict[str, float]],
    n: int = 10,
    key: str = "champion",
) -> List[Tuple[str, float]]:
    """Return top-N teams by championship probability (or other round key)."""
    ranked = sorted(results.items(), key=lambda x: x[1].get(key, 0), reverse=True)
    return [(team, data[key]) for team, data in ranked[:n]]


def save_results(results: Dict[str, Dict[str, float]], path: str) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {path}")


def load_results(path: str) -> Dict[str, Dict[str, float]]:
    with open(path) as f:
        return json.load(f)
