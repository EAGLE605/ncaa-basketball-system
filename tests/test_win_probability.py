"""
Tests for src/model/win_probability.py

TDD: these tests define the contract for win probability calculations.
Wrong probabilities = wrong Kelly sizing = lost money. No shortcuts.
"""

import math
import pytest

from src.model.win_probability import (
    torvik_win_prob,
    market_win_prob,
    ensemble_win_prob,
    build_wp_table,
    DEFAULT_TEMPO,
    WEIGHT_MARKET,
    WEIGHT_TORVIK,
)


# ---------------------------------------------------------------------------
# torvik_win_prob
# ---------------------------------------------------------------------------

class TestTorVikWinProb:
    def test_equal_teams_returns_50pct(self):
        p = torvik_win_prob(20.0, 20.0)
        assert abs(p - 0.5) < 0.001

    def test_better_team_wins_more_than_half(self):
        p = torvik_win_prob(36.69, 28.0)   # Duke vs solid team
        assert p > 0.5

    def test_worse_team_wins_less_than_half(self):
        p = torvik_win_prob(10.0, 36.69)   # weak team vs Duke
        assert p < 0.5

    def test_probability_bounded_0_1(self):
        # Extreme AdjEM diff saturates NormCDF to exactly 0.0 or 1.0 in float64 — use <=
        p_high = torvik_win_prob(100.0, -100.0)
        p_low  = torvik_win_prob(-100.0, 100.0)
        assert 0.0 <= p_high <= 1.0
        assert 0.0 <= p_low  <= 1.0
        # Verify realistic values are strictly bounded
        p_realistic = torvik_win_prob(36.69, 10.0)
        assert 0.0 < p_realistic < 1.0

    def test_probabilities_sum_to_1(self):
        em_a, em_b = 30.0, 20.0
        p_ab = torvik_win_prob(em_a, em_b)
        p_ba = torvik_win_prob(em_b, em_a)
        assert abs(p_ab + p_ba - 1.0) < 1e-5

    def test_1_seed_vs_16_seed_probability(self):
        """1-seed (Duke ~36.69 AdjEM) vs 16-seed (~-8.0 AdjEM) should be ~97%+"""
        p = torvik_win_prob(36.69, -8.0, team_a="Duke", team_b="NCST")
        assert p > 0.95, f"1 vs 16 seed probability too low: {p:.3f}"

    def test_8_vs_9_seed_near_50pct(self):
        """8 vs 9 seed games should be very close to 50/50"""
        p = torvik_win_prob(14.0, 13.0)    # typical 8/9 seed AdjEM gap
        assert 0.45 < p < 0.55

    def test_tempo_factor_increases_spread(self):
        """Higher combined tempo should widen the probability gap."""
        em_diff = 10.0
        p_slow = torvik_win_prob(em_diff, 0, tempo_a=60.0, tempo_b=60.0)
        p_fast = torvik_win_prob(em_diff, 0, tempo_a=75.0, tempo_b=75.0)
        assert p_fast > p_slow

    def test_3pt_heavy_team_widens_sigma(self):
        """3pt-heavy matchup uses sigma=12.5, reducing the probability gap."""
        em_diff = 10.0
        # Auburn and Tennessee are NOT in TEAMS_3PT_HEAVY → sigma=11.0
        p_standard  = torvik_win_prob(em_diff, 0, team_a="Auburn",  team_b="Tennessee")
        # Duke and Michigan ARE 3pt-heavy → sigma=12.5 → more compressed
        p_3pt_heavy = torvik_win_prob(em_diff, 0, team_a="Duke",    team_b="Michigan")
        assert p_standard > p_3pt_heavy, (
            "3pt-heavy matchup should have more compressed win probability"
        )


# ---------------------------------------------------------------------------
# market_win_prob (de-vig)
# ---------------------------------------------------------------------------

class TestMarketWinProb:
    def test_even_moneylines_return_50pct(self):
        """Both teams at +100 → 50/50 after de-vig."""
        p = market_win_prob(ml_home=+100, ml_away=+100, team_a_is_home=True)
        assert abs(p - 0.5) < 0.001

    def test_heavy_favorite_high_probability(self):
        """-300 home favorite should be ~75%+ after de-vig."""
        p = market_win_prob(ml_home=-300, ml_away=+240, team_a_is_home=True)
        assert p > 0.70

    def test_underdog_lower_probability(self):
        """Away dog at +240 should be < 30%."""
        p = market_win_prob(ml_home=-300, ml_away=+240, team_a_is_home=False)
        assert p < 0.30

    def test_probabilities_sum_to_1(self):
        """Home + away prob must sum to 1.0 after de-vig."""
        ml_h, ml_a = -150, +130
        p_home = market_win_prob(ml_h, ml_a, team_a_is_home=True)
        p_away = market_win_prob(ml_h, ml_a, team_a_is_home=False)
        assert abs(p_home + p_away - 1.0) < 1e-5

    def test_vig_removed(self):
        """-110/-110 (standard vig) should de-vig to 50/50 not 52.4%."""
        p = market_win_prob(ml_home=-110, ml_away=-110, team_a_is_home=True)
        assert abs(p - 0.5) < 0.001

    def test_positive_moneyline_formula(self):
        """Manual check: +200 → raw = 100/300 = 0.333"""
        p = market_win_prob(ml_home=+200, ml_away=-200, team_a_is_home=True)
        # raw_home = 100/300 = 0.333, raw_away = 200/300 = 0.667 → devig = 0.333
        assert abs(p - (1/3)) < 0.01

    def test_negative_moneyline_formula(self):
        """Manual check: -200 → raw = 200/300 = 0.667"""
        p = market_win_prob(ml_home=-200, ml_away=+200, team_a_is_home=True)
        assert abs(p - (2/3)) < 0.01


# ---------------------------------------------------------------------------
# ensemble_win_prob
# ---------------------------------------------------------------------------

class TestEnsembleWinProb:
    def test_no_market_line_returns_torvik(self):
        """Without market lines, ensemble == pure torvik."""
        em_a, em_b = 30.0, 20.0
        p_ensemble = ensemble_win_prob(em_a, em_b)
        p_torvik   = torvik_win_prob(em_a, em_b)
        assert abs(p_ensemble - p_torvik) < 1e-5

    def test_with_market_lines_uses_ensemble_weights(self):
        """With market lines, result should blend torvik + market."""
        em_a, em_b = 36.69, 28.0
        # Must pass team names so sigma matches what ensemble_win_prob uses
        p_torvik = torvik_win_prob(em_a, em_b, team_a="Duke", team_b="Iowa")
        p_market = market_win_prob(-200, +170, team_a_is_home=True)
        expected = WEIGHT_MARKET * p_market + WEIGHT_TORVIK * p_torvik

        p_ensemble = ensemble_win_prob(
            em_a, em_b,
            team_a="Duke", team_b="Iowa",
            ml_team_a=-200, ml_team_b=+170
        )
        assert abs(p_ensemble - expected) < 1e-4

    def test_ensemble_bounded_0_1(self):
        p = ensemble_win_prob(50.0, -50.0, ml_team_a=-1000, ml_team_b=+800)
        assert 0.0 < p < 1.0

    def test_ensemble_weights_sum_correctly(self):
        """WEIGHT_MARKET + WEIGHT_TORVIK must equal 1.0."""
        assert abs(WEIGHT_MARKET + WEIGHT_TORVIK - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# build_wp_table
# ---------------------------------------------------------------------------

class TestBuildWpTable:
    def test_returns_all_teams(self):
        teams = {
            "Duke":    {"AdjEM": 36.69, "AdjT": 67.5},
            "Arizona": {"AdjEM": 35.54, "AdjT": 69.1},
            "Florida": {"AdjEM": 33.82, "AdjT": 68.3},
        }
        table = build_wp_table(teams)
        assert set(table.keys()) == {"Duke", "Arizona", "Florida"}

    def test_self_matchup_is_50pct(self):
        teams = {"Duke": {"AdjEM": 36.69, "AdjT": 67.5}}
        table = build_wp_table(teams)
        assert abs(table["Duke"]["Duke"] - 0.5) < 0.001

    def test_table_is_symmetric_complement(self):
        """p(A beats B) + p(B beats A) == 1.0"""
        teams = {
            "Duke":    {"AdjEM": 36.69, "AdjT": 67.5},
            "Arizona": {"AdjEM": 35.54, "AdjT": 69.1},
        }
        table = build_wp_table(teams)
        p_ab = table["Duke"]["Arizona"]
        p_ba = table["Arizona"]["Duke"]
        assert abs(p_ab + p_ba - 1.0) < 1e-5

    def test_bad_adjt_is_clamped_to_default(self):
        """AdjT values outside 55-80 (BartTorvik rank bug) are clamped to default."""
        teams = {
            "TeamA": {"AdjEM": 20.0, "AdjT": 5},    # rank, not tempo — should clamp
            "TeamB": {"AdjEM": 10.0, "AdjT": 67.5},
        }
        table = build_wp_table(teams)  # should not raise
        p = table["TeamA"]["TeamB"]
        assert 0.0 < p < 1.0

    def test_market_lines_shift_probability(self):
        """Providing market lines should change the ensemble probability."""
        teams = {
            "Duke":    {"AdjEM": 36.69, "AdjT": 67.5},
            "Arizona": {"AdjEM": 35.54, "AdjT": 69.1},
        }
        table_no_market = build_wp_table(teams)
        market_lines = {
            "Duke vs Arizona": {
                "home": "Duke", "away": "Arizona",
                "home_ml": -300, "away_ml": +250,
                "p_home": 0.75, "p_away": 0.25,
            }
        }
        table_with_market = build_wp_table(teams, market_lines)
        p_no = table_no_market["Duke"]["Arizona"]
        p_with = table_with_market["Duke"]["Arizona"]
        assert p_no != p_with, "Market lines should change the probability"
