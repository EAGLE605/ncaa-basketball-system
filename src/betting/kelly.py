"""
Kelly Criterion bet sizing for NCAA Men's Basketball.

Full Kelly: f* = (bp - q) / b
    where b = decimal odds - 1, p = model win prob, q = 1 - p

Half/Quarter Kelly used in practice to account for model uncertainty.
"""

from typing import Optional, Tuple


def american_to_decimal(ml: int) -> float:
    """Convert American moneyline to decimal odds."""
    if ml > 0:
        return (ml / 100.0) + 1.0
    else:
        return (100.0 / abs(ml)) + 1.0


def decimal_to_implied(decimal_odds: float) -> float:
    """Decimal odds to implied probability (including vig)."""
    return 1.0 / decimal_odds


def kelly_fraction(
    p_model: float,
    ml: int,
    fraction: float = 0.5,
    max_bet: float = 0.05,
    min_edge: float = 0.02,
) -> float:
    """
    Compute Kelly fraction for a single bet.

    Args:
        p_model:   Model win probability (de-vigged)
        ml:        American moneyline for the side we're betting
        fraction:  Kelly fraction (0.5 = half Kelly, 0.25 = quarter Kelly)
        max_bet:   Hard cap on bet size as fraction of bankroll
        min_edge:  Minimum edge required to bet (default 2%)

    Returns:
        Bet size as fraction of bankroll. 0.0 if no positive edge.
    """
    dec_odds = american_to_decimal(ml)
    b        = dec_odds - 1.0     # net profit per unit staked
    q        = 1.0 - p_model      # probability of loss

    numerator = b * p_model - q
    if numerator <= min_edge * b:
        return 0.0

    full_kelly = numerator / b
    sized      = full_kelly * fraction
    return min(sized, max_bet)


def edge(p_model: float, ml: int) -> float:
    """
    Expected edge over the implied market probability.

    Edge = p_model - p_implied
    Positive edge = value bet.
    """
    dec_odds  = american_to_decimal(ml)
    p_implied = decimal_to_implied(dec_odds)
    return p_model - p_implied


def expected_value(p_model: float, ml: int, stake: float = 1.0) -> float:
    """
    Expected value of a bet.

    EV = p_model * profit - (1 - p_model) * stake
    """
    dec_odds = american_to_decimal(ml)
    profit   = (dec_odds - 1.0) * stake
    return p_model * profit - (1.0 - p_model) * stake


def closing_line_value(
    open_ml: int,
    close_ml: int,
    bet_side_is_favorite: bool,
) -> float:
    """
    Closing Line Value (CLV): measure of bet quality.

    CLV > 0 means you got better odds than the closing line (beating the market).
    """
    open_prob  = decimal_to_implied(american_to_decimal(open_ml))
    close_prob = decimal_to_implied(american_to_decimal(close_ml))

    if bet_side_is_favorite:
        # Favorite: lower ML = more expensive = better for bettor if CL went even lower
        return close_prob - open_prob
    else:
        return open_prob - close_prob


def kelly_spread(
    p_model: float,
    spread: float,
    spread_ml: int = -110,
    sigma: float = 11.0,
    fraction: float = 0.5,
    max_bet: float = 0.05,
    min_edge: float = 0.02,
) -> float:
    """
    Kelly sizing for a spread bet using win probability against the spread.

    Args:
        p_model:   Model probability of covering the spread
        spread:    Spread to cover (negative = favorite, positive = underdog)
        spread_ml: Moneyline for the spread bet (usually -110)
        sigma:     Point spread uncertainty (same as win probability model)
        fraction:  Kelly fraction
        max_bet:   Hard cap
        min_edge:  Minimum edge

    Returns:
        Bet size as fraction of bankroll.
    """
    return kelly_fraction(p_model, spread_ml, fraction, max_bet, min_edge)


def simulate_bankroll_growth(
    bets: list,  # list of {"p_model", "ml", "kelly_fraction", "outcome": 1/0}
    initial: float = 10_000.0,
) -> Tuple[float, float, float]:
    """
    Simulate bankroll growth over a sequence of bets.

    Args:
        bets:    List of bet dicts
        initial: Starting bankroll

    Returns:
        (final_bankroll, total_roi_pct, max_drawdown_pct)
    """
    bankroll = initial
    peak     = initial
    max_dd   = 0.0

    for bet in bets:
        stake    = bankroll * bet["kelly_fraction"]
        dec_odds = american_to_decimal(bet["ml"])
        if bet["outcome"] == 1:
            bankroll += stake * (dec_odds - 1.0)
        else:
            bankroll -= stake
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak
        if dd > max_dd:
            max_dd = dd

    roi = (bankroll - initial) / initial * 100.0
    return bankroll, roi, max_dd * 100.0
