"""
Walk-forward backtesting engine for NCAAB Men's Basketball tournament prediction.

Architecture (from Planner+Architect+Critic consensus plan):
  - BacktestEngine:               pure computation — List[GamePrediction] -> YearResult
  - TournamentBacktestOrchestrator: owns I/O + year iteration + calibration
  - GamePrediction:               typed intermediate between data and metrics
  - YearResult / WalkForwardResult: output data structures

Key design decisions:
  - Walk-forward: train on years 1..N-1, test on year N (rolling)
  - Calibration: cumulative (pool all prior years), min N>=150 guard
  - 2020: skipped (no NCAA tournament, COVID)
  - AdjT: uses default tempo (67.5) for all years — BartTorvik AdjT column is reliable
    as of 2026-03-17 research but historical values were in fact actual tempo (col[15])
  - CLV: Optional[Dict] — None when no market lines, never empty dict
  - Per-year error isolation: partial results returned even when a year fails
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 2020 had no NCAA Tournament
SKIP_YEARS = {2020}

# Minimum prior-game sample size for isotonic calibration to be meaningful
MIN_CALIBRATION_GAMES = 150


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GamePrediction:
    """Typed intermediate between data loader and backtest engine."""
    year: int
    round: str                        # R64 / R32 / S16 / E8 / FF / F
    team_a: str                       # canonical name
    team_b: str                       # canonical name
    seed_a: int
    seed_b: int
    pred_prob_a: float                # raw model probability (team_a wins)
    actual_winner: str                # canonical name
    cal_prob_a: Optional[float] = None   # isotonic-calibrated prob (None if not fit)
    ml_a: Optional[int] = None        # American ML for team_a at open
    ml_b: Optional[int] = None        # American ML for team_b at open
    closing_ml_a: Optional[int] = None
    closing_ml_b: Optional[int] = None
    kelly_fraction_a: float = 0.0     # half-Kelly for team_a (0 if no edge)
    stake: float = 0.0                # fraction of bankroll
    outcome: int = 0                  # 1 if team_a won, 0 if team_b won

    def __post_init__(self):
        self.outcome = 1 if self.actual_winner == self.team_a else 0

    @property
    def effective_prob(self) -> float:
        """Calibrated probability if available, otherwise raw."""
        return self.cal_prob_a if self.cal_prob_a is not None else self.pred_prob_a

    @property
    def is_upset(self) -> bool:
        """True if the lower seed (higher seed number) won."""
        return (
            (self.outcome == 1 and self.seed_a > self.seed_b) or
            (self.outcome == 0 and self.seed_b > self.seed_a)
        )

    @property
    def predicted_upset(self) -> bool:
        """True if model predicted the lower seed to win."""
        return (
            (self.effective_prob > 0.5 and self.seed_a > self.seed_b) or
            (self.effective_prob < 0.5 and self.seed_b > self.seed_a)
        )


@dataclass
class YearResult:
    year: int
    n_games: int
    accuracy_by_round: Dict[str, float]     # R64/R32/S16/E8/FF/F -> fraction correct
    brier_score: float                       # overall Brier (lower = better)
    brier_by_round: Dict[str, float]         # per-round Brier
    upsets_predicted: int                    # model correctly predicted an upset
    upsets_missed: int                       # lower seed won, model backed chalk
    kelly_bets: List[GamePrediction] = field(default_factory=list)
    roi_pct: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    clv_mean: Optional[float] = None        # None when no closing lines
    calibration_skipped: bool = False        # True when N_train < MIN_CALIBRATION_GAMES
    metadata: Dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    years: List[YearResult]
    year_errors: Dict[int, str]              # years that failed with error strings
    ensemble_accuracy: Dict[str, float]      # weighted avg across all years
    calibration_lift: float                  # Brier improvement from calibration
    clv_summary: Optional[Dict]             # None if no odds data
    total_roi_pct: float
    total_sharpe: float


# ---------------------------------------------------------------------------
# Pure computation engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Pure computation layer: given a list of GamePrediction objects for one
    tournament year, compute YearResult metrics.

    No I/O. No data loading. No model calls.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        min_edge: float = 0.02,
        max_bet: float = 0.05,
    ):
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_bet = max_bet

    def evaluate(self, games: List[GamePrediction]) -> YearResult:
        """
        Compute all metrics for a single tournament year.

        Args:
            games: List of GamePrediction for one year (all rounds)

        Returns:
            YearResult with accuracy, Brier, Kelly ROI, Sharpe, drawdown
        """
        if not games:
            return YearResult(
                year=0, n_games=0,
                accuracy_by_round={}, brier_score=0.0, brier_by_round={},
                upsets_predicted=0, upsets_missed=0,
            )

        year = games[0].year
        rounds = ["R64", "R32", "S16", "E8", "FF", "F"]

        accuracy_by_round = self._accuracy_by_round(games, rounds)
        brier_score, brier_by_round = self._brier_by_round(games, rounds)
        upsets_predicted, upsets_missed = self._upset_tracking(games)
        kelly_bets, roi, sharpe, drawdown = self._kelly_series(games)
        clv = self._clv_mean(games)
        cal_skipped = any(g.cal_prob_a is None for g in games)

        return YearResult(
            year=year,
            n_games=len(games),
            accuracy_by_round=accuracy_by_round,
            brier_score=brier_score,
            brier_by_round=brier_by_round,
            upsets_predicted=upsets_predicted,
            upsets_missed=upsets_missed,
            kelly_bets=kelly_bets,
            roi_pct=roi,
            sharpe=sharpe,
            max_drawdown_pct=drawdown,
            clv_mean=clv,
            calibration_skipped=cal_skipped,
            metadata={"tempo_mode": "default_67.5"},
        )

    # ------------------------------------------------------------------
    # Metric calculations
    # ------------------------------------------------------------------

    def _accuracy_by_round(
        self, games: List[GamePrediction], rounds: List[str]
    ) -> Dict[str, float]:
        result = {}
        for rnd in rounds:
            rnd_games = [g for g in games if g.round == rnd]
            if not rnd_games:
                continue
            correct = sum(
                1 for g in rnd_games
                if (g.effective_prob >= 0.5) == (g.outcome == 1)
            )
            result[rnd] = round(correct / len(rnd_games), 4)
        # Overall accuracy
        if games:
            total_correct = sum(
                1 for g in games
                if (g.effective_prob >= 0.5) == (g.outcome == 1)
            )
            result["overall"] = round(total_correct / len(games), 4)
        return result

    def _brier_by_round(
        self, games: List[GamePrediction], rounds: List[str]
    ) -> tuple[float, Dict[str, float]]:
        """Brier score = mean((p - y)^2). Lower = better. Random = 0.25."""
        by_round = {}
        for rnd in rounds:
            rnd_games = [g for g in games if g.round == rnd]
            if not rnd_games:
                continue
            scores = [(g.effective_prob - g.outcome) ** 2 for g in rnd_games]
            by_round[rnd] = round(float(np.mean(scores)), 4)

        overall_scores = [(g.effective_prob - g.outcome) ** 2 for g in games]
        overall = round(float(np.mean(overall_scores)), 4) if overall_scores else 0.0
        return overall, by_round

    def _upset_tracking(self, games: List[GamePrediction]) -> tuple[int, int]:
        predicted = sum(1 for g in games if g.is_upset and g.predicted_upset)
        missed    = sum(1 for g in games if g.is_upset and not g.predicted_upset)
        return predicted, missed

    def _kelly_series(
        self, games: List[GamePrediction]
    ) -> tuple[List[GamePrediction], float, float, float]:
        """
        Simulate Kelly-sized bankroll over tournament games.
        Returns (bet_games, roi_pct, sharpe, max_drawdown_pct).
        """
        from src.betting.kelly import kelly_fraction as kelly_f, edge as edge_f
        from src.betting.kelly import american_to_decimal

        bankroll = 1.0
        peak = 1.0
        max_dd = 0.0
        returns = []
        bet_games = []

        for g in games:
            if g.ml_a is None or g.ml_b is None:
                continue
            e = edge_f(g.effective_prob, g.ml_a)
            if e < self.min_edge:
                continue
            f = kelly_f(g.effective_prob, g.ml_a,
                        fraction=self.kelly_fraction, max_bet=self.max_bet)
            if f <= 0:
                continue

            g.kelly_fraction_a = f
            g.stake = f
            bet_games.append(g)

            stake = bankroll * f
            dec = american_to_decimal(g.ml_a)
            if g.outcome == 1:
                bankroll += stake * (dec - 1.0)
            else:
                bankroll -= stake

            ret = (bankroll - peak) / peak
            returns.append(ret)
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak
            if dd > max_dd:
                max_dd = dd

        roi = round((bankroll - 1.0) * 100, 3)
        sharpe = self._sharpe(returns)
        return bet_games, roi, sharpe, round(max_dd * 100, 3)

    def _sharpe(self, returns: List[float], risk_free: float = 0.0) -> float:
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns, dtype=float)
        excess = arr - risk_free
        std = float(np.std(excess, ddof=1))
        if std == 0:
            return 0.0
        return round(float(np.mean(excess)) / std * np.sqrt(len(arr)), 4)

    def _clv_mean(self, games: List[GamePrediction]) -> Optional[float]:
        """Mean closing line value. None if no closing lines available."""
        from src.betting.kelly import closing_line_value
        clv_vals = []
        for g in games:
            if g.ml_a is None or g.closing_ml_a is None:
                continue
            is_fav = g.ml_a < 0
            clv = closing_line_value(g.ml_a, g.closing_ml_a, bet_side_is_favorite=is_fav)
            clv_vals.append(clv)
        if not clv_vals:
            return None
        return round(float(np.mean(clv_vals)), 4)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TournamentBacktestOrchestrator:
    """
    Owns data loading, year iteration, calibration fitting, and aggregation.
    Calls BacktestEngine for pure metric computation.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        n_sims: int = 10_000,      # lower for backtest speed
        kelly_fraction: float = 0.5,
        min_edge: float = 0.02,
        max_bet: float = 0.05,
    ):
        self.n_sims = n_sims
        self.engine = BacktestEngine(kelly_fraction, min_edge, max_bet)
        self._data_dir = data_dir

    def run_year(self, year: int, prior_games: List[GamePrediction], mock: bool = False) -> YearResult:
        """
        Run backtest for a single tournament year.

        Args:
            year:         Tournament year to evaluate
            prior_games:  All GamePrediction from prior years (for calibration)
            mock:         True = skip API calls, use fixture data

        Returns:
            YearResult for this year
        """
        from src.backtesting.data_loader import NcaabDataLoader
        from src.model.win_probability import build_wp_table
        from src.model.calibration import Calibrator

        loader = NcaabDataLoader(data_dir=self._data_dir)

        # Load data
        torvik_data = loader.load_torvik_season(year, mock=mock)
        results    = loader.load_tournament_results(year, mock=mock)
        bracket    = loader.build_bracket(year, mock=mock)
        odds       = loader.load_historical_odds(year)

        if not results:
            raise RuntimeError(f"No tournament results for year {year} — verify data/tournament_results/{year}.csv")

        # Build win probability table
        wp_table = build_wp_table(torvik_data, odds)

        # Fit calibrator on prior years
        calibrator = None
        cal_skipped = False
        if len(prior_games) >= MIN_CALIBRATION_GAMES:
            cal = Calibrator()
            cal.fit(
                [g.pred_prob_a for g in prior_games],
                [g.outcome for g in prior_games],
            )
            calibrator = cal
        else:
            cal_skipped = True
            logger.info(
                f"Year {year}: calibration skipped — only {len(prior_games)} prior games "
                f"(need {MIN_CALIBRATION_GAMES})"
            )

        # Build GamePrediction list from results
        games = []
        for res in results:
            team_a = res["team_a"]
            team_b = res["team_b"]
            p_raw = wp_table.get(team_a, {}).get(team_b)
            if p_raw is None:
                logger.warning(f"No WP for {team_a} vs {team_b} in {year} — skipping")
                continue

            cal_p = calibrator.transform(p_raw) if calibrator else None
            game = GamePrediction(
                year=year,
                round=res["round"],
                team_a=team_a,
                team_b=team_b,
                seed_a=res.get("seed_a", 0),
                seed_b=res.get("seed_b", 0),
                pred_prob_a=p_raw,
                cal_prob_a=cal_p,
                actual_winner=res["winner"],
                ml_a=res.get("ml_a"),
                ml_b=res.get("ml_b"),
                closing_ml_a=res.get("closing_ml_a"),
                closing_ml_b=res.get("closing_ml_b"),
            )
            games.append(game)

        result = self.engine.evaluate(games)
        result.calibration_skipped = cal_skipped
        return result

    def run_walk_forward(
        self,
        start: int = 2016,
        end: int = 2025,
        mock: bool = False,
    ) -> WalkForwardResult:
        """
        Run walk-forward backtest across multiple tournament years.

        Skips SKIP_YEARS (2020). Each year is tested on out-of-sample data
        with calibration trained on all prior years.

        Args:
            start:  First test year (e.g. 2016)
            end:    Last test year inclusive (e.g. 2025)
            mock:   True = use fixture data (no network)

        Returns:
            WalkForwardResult with per-year metrics and aggregates
        """
        year_results: List[YearResult] = []
        year_errors: Dict[int, str] = {}
        prior_games: List[GamePrediction] = []

        for year in range(start, end + 1):
            if year in SKIP_YEARS:
                logger.info(f"Skipping {year} (no tournament)")
                continue

            logger.info(f"=== Backtesting {year} ===")
            try:
                result = self.run_year(year, prior_games, mock=mock)
                year_results.append(result)
                # Add this year's games to the cumulative calibration pool
                prior_games.extend(result.kelly_bets)
                logger.info(
                    f"  {year}: accuracy={result.accuracy_by_round.get('overall', 0):.1%}, "
                    f"Brier={result.brier_score:.4f}, ROI={result.roi_pct:+.1f}%"
                )
            except Exception as e:
                year_errors[year] = str(e)
                logger.error(f"Year {year} failed: {e}")

        return self._aggregate(year_results, year_errors)

    def _aggregate(
        self,
        years: List[YearResult],
        errors: Dict[int, str],
    ) -> WalkForwardResult:
        """Aggregate per-year results into summary metrics."""
        if not years:
            return WalkForwardResult(
                years=[], year_errors=errors,
                ensemble_accuracy={}, calibration_lift=0.0,
                clv_summary=None, total_roi_pct=0.0, total_sharpe=0.0,
            )

        rounds = ["R64", "R32", "S16", "E8", "FF", "F", "overall"]
        ensemble_acc = {}
        for rnd in rounds:
            vals = [y.accuracy_by_round.get(rnd) for y in years if rnd in y.accuracy_by_round]
            if vals:
                ensemble_acc[rnd] = round(float(np.mean(vals)), 4)

        # Calibration lift: compare raw vs calibrated Brier
        raw_briers  = [y.brier_score for y in years]
        avg_brier   = float(np.mean(raw_briers)) if raw_briers else 0.0

        # CLV summary
        clv_vals = [y.clv_mean for y in years if y.clv_mean is not None]
        clv_summary = {"mean": round(float(np.mean(clv_vals)), 4)} if clv_vals else None

        total_roi = sum(y.roi_pct for y in years)

        all_returns = []
        for y in years:
            all_returns.extend([g.kelly_fraction_a for g in y.kelly_bets])
        total_sharpe = self.engine._sharpe(all_returns)

        return WalkForwardResult(
            years=years,
            year_errors=errors,
            ensemble_accuracy=ensemble_acc,
            calibration_lift=0.0,  # computed after both raw+cal runs
            clv_summary=clv_summary,
            total_roi_pct=round(total_roi, 3),
            total_sharpe=total_sharpe,
        )
