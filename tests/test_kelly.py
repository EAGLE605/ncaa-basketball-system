"""
Tests for src/betting/kelly.py

Kelly sizing errors = incorrect bet sizing = real money lost.
Every formula must be verified against manual calculations.
"""

import pytest

from src.betting.kelly import (
    american_to_decimal,
    decimal_to_implied,
    kelly_fraction,
    edge,
    expected_value,
    closing_line_value,
    simulate_bankroll_growth,
)


# ---------------------------------------------------------------------------
# american_to_decimal
# ---------------------------------------------------------------------------

class TestAmericanToDecimal:
    def test_plus_100(self):
        assert abs(american_to_decimal(+100) - 2.0) < 1e-9

    def test_minus_100(self):
        assert abs(american_to_decimal(-100) - 2.0) < 1e-9

    def test_minus_110(self):
        """Standard vig: -110 → 1.9090..."""
        d = american_to_decimal(-110)
        assert abs(d - (100/110 + 1)) < 1e-9

    def test_plus_200(self):
        """Underdog +200 → 3.0"""
        assert abs(american_to_decimal(+200) - 3.0) < 1e-9

    def test_minus_200(self):
        """Heavy favorite -200 → 1.5"""
        assert abs(american_to_decimal(-200) - 1.5) < 1e-9

    def test_plus_300(self):
        assert abs(american_to_decimal(+300) - 4.0) < 1e-9

    def test_minus_150(self):
        # -150 → 100/150 + 1 = 0.6667 + 1 = 1.6667
        assert abs(american_to_decimal(-150) - (100/150 + 1)) < 1e-6


# ---------------------------------------------------------------------------
# decimal_to_implied
# ---------------------------------------------------------------------------

class TestDecimalToImplied:
    def test_even_money(self):
        assert abs(decimal_to_implied(2.0) - 0.5) < 1e-9

    def test_3_to_1_underdog(self):
        assert abs(decimal_to_implied(4.0) - 0.25) < 1e-9

    def test_heavy_favorite(self):
        # 1.5 decimal → 1/1.5 = 0.6667
        assert abs(decimal_to_implied(1.5) - (2/3)) < 1e-6


# ---------------------------------------------------------------------------
# edge
# ---------------------------------------------------------------------------

class TestEdge:
    def test_no_edge_at_fair_line(self):
        """If model prob == implied prob, edge == 0."""
        # +100 implied = 0.5; model also 0.5
        e = edge(p_model=0.5, ml=+100)
        assert abs(e) < 0.001

    def test_positive_edge(self):
        """Model says 60% on a line that implies 50% → +10% edge."""
        e = edge(p_model=0.60, ml=+100)
        assert abs(e - 0.10) < 0.001

    def test_negative_edge(self):
        """Model says 40% on a line that implies 52% → negative edge."""
        e = edge(p_model=0.40, ml=-110)
        assert e < 0

    def test_heavy_vig_reduces_edge(self):
        """Even with correct model, heavy vig eats into edge."""
        e_fair = edge(p_model=0.55, ml=+100)   # fair -110 equivalent
        e_vigged = edge(p_model=0.55, ml=-120)  # worse price
        assert e_fair > e_vigged


# ---------------------------------------------------------------------------
# kelly_fraction
# ---------------------------------------------------------------------------

class TestKellyFraction:
    def test_zero_fraction_on_no_edge(self):
        """No edge → bet nothing."""
        f = kelly_fraction(p_model=0.5, ml=+100, min_edge=0.02)
        assert f == 0.0

    def test_positive_fraction_on_positive_edge(self):
        """Clear edge → positive fraction."""
        f = kelly_fraction(p_model=0.65, ml=+100)
        assert f > 0.0

    def test_respects_max_bet_cap(self):
        """Even massive edge is capped at max_bet."""
        f = kelly_fraction(p_model=0.99, ml=+1000, max_bet=0.05)
        assert f <= 0.05

    def test_half_kelly_is_half_of_full(self):
        """fraction=0.5 should give exactly half of fraction=1.0."""
        full  = kelly_fraction(p_model=0.60, ml=+100, fraction=1.0, max_bet=1.0)
        half  = kelly_fraction(p_model=0.60, ml=+100, fraction=0.5, max_bet=1.0)
        assert abs(half - full / 2) < 1e-6

    def test_manual_kelly_calculation(self):
        """
        Manual: p=0.6, ml=+200 → b=2.0, q=0.4
        full_kelly = (2*0.6 - 0.4) / 2 = 0.8/2 = 0.40
        half_kelly = 0.20
        """
        f = kelly_fraction(p_model=0.60, ml=+200, fraction=0.5, max_bet=1.0, min_edge=0.0)
        assert abs(f - 0.20) < 0.001

    def test_negative_edge_returns_zero(self):
        f = kelly_fraction(p_model=0.45, ml=-110)
        assert f == 0.0

    def test_fraction_bounded_0_1(self):
        f = kelly_fraction(p_model=0.70, ml=+150, fraction=1.0, max_bet=1.0)
        assert 0.0 <= f <= 1.0


# ---------------------------------------------------------------------------
# expected_value
# ---------------------------------------------------------------------------

class TestExpectedValue:
    def test_negative_ev_at_fair_line_with_vig(self):
        """-110 on 50/50 proposition is negative EV."""
        ev = expected_value(p_model=0.5, ml=-110)
        assert ev < 0

    def test_positive_ev_on_edge(self):
        """Strong edge at fair price → positive EV."""
        ev = expected_value(p_model=0.65, ml=+100)
        assert ev > 0

    def test_manual_ev_calculation(self):
        """
        p=0.6, ml=+100, stake=10
        EV = 0.6*10 - 0.4*10 = 6 - 4 = 2.0
        """
        ev = expected_value(p_model=0.60, ml=+100, stake=10.0)
        assert abs(ev - 2.0) < 0.001

    def test_ev_scales_with_stake(self):
        ev1 = expected_value(0.6, +100, stake=1.0)
        ev10 = expected_value(0.6, +100, stake=10.0)
        assert abs(ev10 - 10 * ev1) < 1e-6


# ---------------------------------------------------------------------------
# closing_line_value
# ---------------------------------------------------------------------------

class TestClosingLineValue:
    def test_positive_clv_when_got_better_price(self):
        """Bet -110 open, line moved to -130 close → positive CLV for favorite bettor."""
        clv = closing_line_value(open_ml=-110, close_ml=-130, bet_side_is_favorite=True)
        assert clv > 0

    def test_negative_clv_when_line_moved_against(self):
        clv = closing_line_value(open_ml=-130, close_ml=-110, bet_side_is_favorite=True)
        assert clv < 0

    def test_zero_clv_at_same_line(self):
        clv = closing_line_value(open_ml=-110, close_ml=-110, bet_side_is_favorite=True)
        assert abs(clv) < 0.001


# ---------------------------------------------------------------------------
# simulate_bankroll_growth
# ---------------------------------------------------------------------------

class TestSimulateBankrollGrowth:
    def test_all_wins_grows_bankroll(self):
        bets = [
            {"p_model": 0.60, "ml": +100, "kelly_fraction": 0.05, "outcome": 1},
            {"p_model": 0.60, "ml": +100, "kelly_fraction": 0.05, "outcome": 1},
            {"p_model": 0.60, "ml": +100, "kelly_fraction": 0.05, "outcome": 1},
        ]
        final, roi, dd = simulate_bankroll_growth(bets, initial=1000.0)
        assert final > 1000.0
        assert roi > 0

    def test_all_losses_shrinks_bankroll(self):
        bets = [
            {"p_model": 0.60, "ml": +100, "kelly_fraction": 0.10, "outcome": 0},
            {"p_model": 0.60, "ml": +100, "kelly_fraction": 0.10, "outcome": 0},
        ]
        final, roi, dd = simulate_bankroll_growth(bets, initial=1000.0)
        assert final < 1000.0
        assert dd > 0

    def test_empty_bets_returns_initial(self):
        final, roi, dd = simulate_bankroll_growth([], initial=5000.0)
        assert final == 5000.0
        assert roi == 0.0
        assert dd == 0.0
