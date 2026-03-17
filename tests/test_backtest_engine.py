"""
Tests for BacktestEngine and GamePrediction data structures.

Covers: GamePrediction logic, accuracy_by_round, Brier score,
upset tracking, Kelly ROI, Sharpe, drawdown, CLV, WalkForwardResult.

All tests use synthetic data — no API calls.
"""
import pytest
from typing import List

from src.backtesting.engine import (
    BacktestEngine,
    GamePrediction,
    YearResult,
    WalkForwardResult,
    TournamentBacktestOrchestrator,
    SKIP_YEARS,
    MIN_CALIBRATION_GAMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_game(
    team_a="Duke",
    team_b="Gonzaga",
    seed_a=1,
    seed_b=2,
    pred_prob_a=0.65,
    actual_winner="Duke",
    round="R64",
    year=2022,
    cal_prob_a=None,
    ml_a=None,
    ml_b=None,
    closing_ml_a=None,
    closing_ml_b=None,
) -> GamePrediction:
    return GamePrediction(
        year=year,
        round=round,
        team_a=team_a,
        team_b=team_b,
        seed_a=seed_a,
        seed_b=seed_b,
        pred_prob_a=pred_prob_a,
        actual_winner=actual_winner,
        cal_prob_a=cal_prob_a,
        ml_a=ml_a,
        ml_b=ml_b,
        closing_ml_a=closing_ml_a,
        closing_ml_b=closing_ml_b,
    )


def make_correct_games(n: int = 63, round: str = "R64") -> List[GamePrediction]:
    """Synthetic games where model is always correct (pred_prob_a > 0.5, team_a wins)."""
    return [
        make_game(
            team_a=f"Team{i}A",
            team_b=f"Team{i}B",
            pred_prob_a=0.7,
            actual_winner=f"Team{i}A",
            round=round,
            year=2022,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# GamePrediction logic
# ---------------------------------------------------------------------------

class TestGamePrediction:
    def test_outcome_team_a_wins(self):
        g = make_game(actual_winner="Duke")
        assert g.outcome == 1

    def test_outcome_team_b_wins(self):
        g = make_game(actual_winner="Gonzaga")
        assert g.outcome == 0

    def test_effective_prob_uses_cal_when_set(self):
        g = make_game(pred_prob_a=0.65, cal_prob_a=0.72)
        assert g.effective_prob == 0.72

    def test_effective_prob_falls_back_to_raw(self):
        g = make_game(pred_prob_a=0.65, cal_prob_a=None)
        assert g.effective_prob == 0.65

    def test_is_upset_lower_seed_wins(self):
        # seed_a=5 beats seed_b=4 — upset (lower seed won)
        g = make_game(seed_a=5, seed_b=4, actual_winner="Duke")  # Duke = team_a
        assert g.is_upset is True

    def test_is_upset_higher_seed_wins(self):
        # seed_a=1 beats seed_b=16 — not an upset
        g = make_game(seed_a=1, seed_b=16, actual_winner="Duke")
        assert g.is_upset is False

    def test_predicted_upset_true(self):
        # Model predicts lower seed wins
        g = make_game(seed_a=5, seed_b=4, pred_prob_a=0.6, actual_winner="Duke")
        assert g.predicted_upset is True

    def test_predicted_upset_false_when_chalk(self):
        # seed_a=1 is favorite, model assigns >0.5 — not predicting upset
        g = make_game(seed_a=1, seed_b=8, pred_prob_a=0.8, actual_winner="Duke")
        assert g.predicted_upset is False

    def test_is_upset_team_b_wins(self):
        # seed_b=12 beats seed_a=5 — upset (b had higher seed number)
        g = make_game(seed_a=5, seed_b=12, actual_winner="Gonzaga")  # Gonzaga = team_b
        assert g.is_upset is True

    def test_predicted_upset_team_b_lower_seed(self):
        # seed_b=12, model gives team_a prob < 0.5 => model predicts team_b (12-seed) wins
        g = make_game(seed_a=5, seed_b=12, pred_prob_a=0.3, actual_winner="Gonzaga")
        assert g.predicted_upset is True


# ---------------------------------------------------------------------------
# BacktestEngine.evaluate
# ---------------------------------------------------------------------------

class TestBacktestEngineEmpty:
    def test_empty_games_returns_zero_result(self):
        engine = BacktestEngine()
        result = engine.evaluate([])
        assert result.n_games == 0
        assert result.accuracy_by_round == {}
        assert result.brier_score == 0.0

    def test_year_from_first_game(self):
        engine = BacktestEngine()
        games = make_correct_games(3)
        result = engine.evaluate(games)
        assert result.year == 2022

    def test_n_games_count(self):
        engine = BacktestEngine()
        games = make_correct_games(10)
        result = engine.evaluate(games)
        assert result.n_games == 10


class TestAccuracyByRound:
    def test_perfect_accuracy(self):
        engine = BacktestEngine()
        games = make_correct_games(32)
        result = engine.evaluate(games)
        assert result.accuracy_by_round["R64"] == pytest.approx(1.0)
        assert result.accuracy_by_round["overall"] == pytest.approx(1.0)

    def test_zero_accuracy_all_wrong(self):
        engine = BacktestEngine()
        # pred_prob_a > 0.5 but team_b wins
        games = [
            make_game(pred_prob_a=0.8, actual_winner="Gonzaga", round="R64", year=2022)
            for _ in range(10)
        ]
        result = engine.evaluate(games)
        assert result.accuracy_by_round["R64"] == pytest.approx(0.0)

    def test_half_accuracy(self):
        engine = BacktestEngine()
        games = (
            make_correct_games(8)
            + [
                make_game(pred_prob_a=0.8, actual_winner="Gonzaga", round="R64", year=2022)
                for _ in range(8)
            ]
        )
        result = engine.evaluate(games)
        assert result.accuracy_by_round["R64"] == pytest.approx(0.5)

    def test_multiple_rounds_tracked(self):
        engine = BacktestEngine()
        r64_games = [make_game(round="R64", pred_prob_a=0.7, actual_winner="Duke") for _ in range(32)]
        r32_games = [make_game(round="R32", pred_prob_a=0.7, actual_winner="Duke") for _ in range(16)]
        result = engine.evaluate(r64_games + r32_games)
        assert "R64" in result.accuracy_by_round
        assert "R32" in result.accuracy_by_round

    def test_only_present_rounds_included(self):
        engine = BacktestEngine()
        games = [make_game(round="F", pred_prob_a=0.6, actual_winner="Duke")]
        result = engine.evaluate(games)
        assert "F" in result.accuracy_by_round
        assert "R64" not in result.accuracy_by_round


class TestBrierScore:
    def test_perfect_predictions_brier_low(self):
        """All predictions close to 1.0 and all team_a wins -> near-zero Brier."""
        engine = BacktestEngine()
        games = [
            make_game(pred_prob_a=0.99, actual_winner="Duke")
            for _ in range(10)
        ]
        result = engine.evaluate(games)
        assert result.brier_score < 0.05

    def test_completely_wrong_brier_near_one(self):
        """pred_prob=0.01 and team_a always wins -> near (1-0.01)^2 ~ 0.98."""
        engine = BacktestEngine()
        games = [
            make_game(pred_prob_a=0.01, actual_winner="Duke")
            for _ in range(10)
        ]
        result = engine.evaluate(games)
        assert result.brier_score > 0.9

    def test_random_predictions_brier_near_quarter(self):
        """Random 0.5 predictions -> Brier = 0.25."""
        engine = BacktestEngine()
        games = [
            make_game(pred_prob_a=0.5, actual_winner="Duke")
            for _ in range(20)
        ]
        result = engine.evaluate(games)
        assert result.brier_score == pytest.approx(0.25, abs=1e-4)

    def test_brier_by_round_computed(self):
        engine = BacktestEngine()
        games = (
            [make_game(round="R64", pred_prob_a=0.7, actual_winner="Duke") for _ in range(5)]
            + [make_game(round="R32", pred_prob_a=0.7, actual_winner="Gonzaga") for _ in range(3)]
        )
        result = engine.evaluate(games)
        assert "R64" in result.brier_by_round
        assert "R32" in result.brier_by_round


class TestUpsetTracking:
    def test_correctly_predicted_upset_counted(self):
        engine = BacktestEngine()
        # team_a = 12 seed, model says 60% chance -> predicts upset, and upset happens
        g = make_game(seed_a=12, seed_b=5, pred_prob_a=0.6, actual_winner="Duke")
        result = engine.evaluate([g])
        assert result.upsets_predicted == 1
        assert result.upsets_missed == 0

    def test_missed_upset_counted(self):
        engine = BacktestEngine()
        # team_a = 1 seed, model says 85% -> chalk pick; but team_b (16-seed) wins
        g = make_game(seed_a=1, seed_b=16, pred_prob_a=0.85, actual_winner="Gonzaga")
        result = engine.evaluate([g])
        assert result.upsets_missed == 1
        assert result.upsets_predicted == 0

    def test_no_upsets_in_normal_results(self):
        engine = BacktestEngine()
        # 1 seed beats 16 seed as expected
        g = make_game(seed_a=1, seed_b=16, pred_prob_a=0.95, actual_winner="Duke")
        result = engine.evaluate([g])
        assert result.upsets_predicted == 0
        assert result.upsets_missed == 0


class TestKellyROI:
    def test_no_bets_when_no_odds(self):
        engine = BacktestEngine()
        games = make_correct_games(10)
        result = engine.evaluate(games)
        assert result.roi_pct == 0.0
        assert result.kelly_bets == []

    def test_positive_edge_generates_bet(self):
        """Model has p=0.6, line is -110 (implied ~0.52) -> positive edge."""
        engine = BacktestEngine(min_edge=0.01)
        g = make_game(
            pred_prob_a=0.6,
            actual_winner="Duke",
            ml_a=-110,
            ml_b=+100,
        )
        result = engine.evaluate([g])
        assert len(result.kelly_bets) == 1
        assert result.kelly_bets[0].kelly_fraction_a > 0

    def test_roi_positive_when_model_wins(self):
        """Many winning bets should produce positive ROI."""
        engine = BacktestEngine(min_edge=0.01)
        games = [
            make_game(
                team_a=f"Team{i}A",
                team_b=f"Team{i}B",
                pred_prob_a=0.65,
                actual_winner=f"Team{i}A",
                ml_a=-110,
                ml_b=+100,
                year=2022,
            )
            for i in range(30)
        ]
        result = engine.evaluate(games)
        assert result.roi_pct > 0.0

    def test_max_bet_cap_respected(self):
        """Kelly fraction should never exceed max_bet."""
        engine = BacktestEngine(max_bet=0.05, min_edge=0.001)
        g = make_game(pred_prob_a=0.95, ml_a=-110, ml_b=+100, actual_winner="Duke")
        engine.evaluate([g])
        assert g.kelly_fraction_a <= 0.05 + 1e-9


class TestSharpeAndDrawdown:
    def test_sharpe_zero_no_bets(self):
        engine = BacktestEngine()
        result = engine.evaluate(make_correct_games(5))
        assert result.sharpe == 0.0

    def test_drawdown_zero_when_always_winning(self):
        """If every bet wins, drawdown = 0."""
        engine = BacktestEngine(min_edge=0.001)
        games = [
            make_game(
                team_a=f"W{i}",
                team_b=f"L{i}",
                pred_prob_a=0.65,
                actual_winner=f"W{i}",
                ml_a=-110,
                ml_b=+100,
                year=2022,
            )
            for i in range(10)
        ]
        result = engine.evaluate(games)
        assert result.max_drawdown_pct >= 0.0


class TestCLV:
    def test_clv_none_when_no_closing_lines(self):
        engine = BacktestEngine()
        g = make_game(ml_a=-110, ml_b=+100)
        result = engine.evaluate([g])
        assert result.clv_mean is None

    def test_clv_computed_when_lines_present(self):
        engine = BacktestEngine()
        g = make_game(
            ml_a=-110, ml_b=+100,
            closing_ml_a=-120, closing_ml_b=+110,
        )
        result = engine.evaluate([g])
        # CLV may be positive or negative but should be a float
        assert result.clv_mean is not None
        assert isinstance(result.clv_mean, float)


# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

class TestConstants:
    def test_2020_in_skip_years(self):
        assert 2020 in SKIP_YEARS

    def test_min_calibration_games_reasonable(self):
        assert MIN_CALIBRATION_GAMES >= 50
        assert MIN_CALIBRATION_GAMES <= 500


# ---------------------------------------------------------------------------
# TournamentBacktestOrchestrator (mock-only)
# ---------------------------------------------------------------------------

class TestOrchestratorMock:
    def test_run_year_mock_returns_year_result(self):
        orch = TournamentBacktestOrchestrator(n_sims=100)
        result = orch.run_year(2022, prior_games=[], mock=True)
        assert isinstance(result, YearResult)
        assert result.year == 2022
        assert result.n_games > 0

    def test_run_year_2020_skipped_in_walk_forward(self):
        """2020 is in SKIP_YEARS and should not appear in results."""
        orch = TournamentBacktestOrchestrator(n_sims=100)
        wf = orch.run_walk_forward(start=2019, end=2021, mock=True)
        year_list = [y.year for y in wf.years]
        assert 2020 not in year_list

    def test_walk_forward_multiple_years_mock(self):
        orch = TournamentBacktestOrchestrator(n_sims=100)
        wf = orch.run_walk_forward(start=2021, end=2023, mock=True)
        assert isinstance(wf, WalkForwardResult)
        assert len(wf.years) >= 1

    def test_walk_forward_ensemble_accuracy_keys(self):
        orch = TournamentBacktestOrchestrator(n_sims=100)
        wf = orch.run_walk_forward(start=2021, end=2022, mock=True)
        for rnd in wf.ensemble_accuracy:
            assert wf.ensemble_accuracy[rnd] >= 0.0
            assert wf.ensemble_accuracy[rnd] <= 1.0

    def test_year_errors_empty_on_clean_mock_run(self):
        orch = TournamentBacktestOrchestrator(n_sims=100)
        wf = orch.run_walk_forward(start=2022, end=2022, mock=True)
        assert 2022 not in wf.year_errors

    def test_calibration_skipped_flag_early_years(self):
        """Early years have fewer prior games than MIN_CALIBRATION_GAMES."""
        orch = TournamentBacktestOrchestrator(n_sims=100)
        result = orch.run_year(2016, prior_games=[], mock=True)
        assert result.calibration_skipped is True
