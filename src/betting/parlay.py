"""
Parlay and correlated Kelly sizing for NCAA Men's Basketball tournament.

Tournament legs are correlated — if Duke reaches the Elite Eight, they had
to beat their R32 opponent first. Naive Kelly on independent legs overcounts
bankroll exposure on correlated outcomes.

This module provides:
  - parlay_kelly: Kelly fraction for a 2+ leg parlay
  - correlated_kelly: adjusted sizing for correlated tournament sequence bets
  - tournament_path_kelly: Kelly for betting a team to reach a specific round
"""

import math
from typing import List, Optional, Tuple


def parlay_decimal_odds(leg_decimals: List[float]) -> float:
    """
    Combined decimal odds for a parlay (multiply all legs).

    Args:
        leg_decimals: List of decimal odds for each leg

    Returns:
        Combined parlay decimal odds
    """
    result = 1.0
    for d in leg_decimals:
        result *= d
    return result


def parlay_win_prob(leg_probs: List[float]) -> float:
    """
    Win probability for a parlay assuming independent legs.

    For correlated tournament path bets, use correlated_path_prob instead.
    """
    result = 1.0
    for p in leg_probs:
        result *= p
    return result


def parlay_kelly(
    leg_probs: List[float],
    leg_decimals: List[float],
    fraction: float = 0.25,   # quarter Kelly for parlays (more variance)
    max_bet: float = 0.02,    # tighter cap for parlays
) -> float:
    """
    Kelly fraction for a parlay bet (independent legs assumed).

    Uses quarter Kelly by default — parlays have compounding variance.

    Args:
        leg_probs:    Win probability for each leg
        leg_decimals: Decimal odds for each leg
        fraction:     Kelly fraction (default 0.25 = quarter Kelly)
        max_bet:      Hard cap on bet size

    Returns:
        Fraction of bankroll to bet. 0.0 if negative expected value.
    """
    p_win = parlay_win_prob(leg_probs)
    combined_dec = parlay_decimal_odds(leg_decimals)
    b = combined_dec - 1.0
    q = 1.0 - p_win

    numerator = b * p_win - q
    if numerator <= 0:
        return 0.0

    full_kelly = numerator / b
    return min(full_kelly * fraction, max_bet)


def correlated_kelly(
    p_win_a: float,
    p_win_b: float,
    ml_a: int,
    ml_b: int,
    correlation: float = 0.3,
    fraction: float = 0.5,
    max_bet: float = 0.05,
    min_edge: float = 0.02,
) -> Tuple[float, float]:
    """
    Adjusted Kelly fractions for two correlated bets (e.g. same team, consecutive rounds).

    When two outcomes are positively correlated (r > 0), naive Kelly
    oversizes both bets. We reduce each fraction by the correlation factor.

    Args:
        p_win_a, p_win_b: Model win probabilities
        ml_a, ml_b:       American moneylines
        correlation:       Pearson correlation between outcomes (0–1)
                           0.3 = typical for consecutive tournament rounds
        fraction:          Base Kelly fraction before correlation adjustment
        max_bet:           Hard cap
        min_edge:          Minimum edge to bet

    Returns:
        (kelly_a, kelly_b) — correlation-adjusted fractions
    """
    from src.betting.kelly import kelly_fraction, edge

    e_a = edge(p_win_a, ml_a)
    e_b = edge(p_win_b, ml_b)

    f_a = kelly_fraction(p_win_a, ml_a, fraction=fraction, max_bet=max_bet, min_edge=min_edge)
    f_b = kelly_fraction(p_win_b, ml_b, fraction=fraction, max_bet=max_bet, min_edge=min_edge)

    # Correlation adjustment: reduce each bet by correlation factor
    adj = 1.0 - correlation
    return round(f_a * adj, 4), round(f_b * adj, 4)


def tournament_path_kelly(
    path_probs: List[float],
    futures_ml: int,
    fraction: float = 0.25,
    max_bet: float = 0.02,
) -> float:
    """
    Kelly sizing for a tournament futures bet (e.g. team to win championship).

    Args:
        path_probs:  List of win probabilities for each round
                     e.g. [0.97, 0.80, 0.70, 0.60, 0.55, 0.50] for R64..F
        futures_ml:  American moneyline for the futures bet (e.g. +600)
        fraction:    Kelly fraction (quarter Kelly recommended for futures)
        max_bet:     Hard cap

    Returns:
        Kelly fraction of bankroll to bet.
    """
    p_champion = math.prod(path_probs)
    return parlay_kelly(
        leg_probs=[p_champion],
        leg_decimals=[_american_to_decimal(futures_ml)],
        fraction=fraction,
        max_bet=max_bet,
    )


def expected_parlay_value(
    leg_probs: List[float],
    leg_decimals: List[float],
    stake: float = 1.0,
) -> float:
    """Expected value of a parlay bet."""
    p_win = parlay_win_prob(leg_probs)
    combined = parlay_decimal_odds(leg_decimals)
    return p_win * (combined - 1.0) * stake - (1.0 - p_win) * stake


def _american_to_decimal(ml: int) -> float:
    if ml > 0:
        return ml / 100.0 + 1.0
    return 100.0 / abs(ml) + 1.0
