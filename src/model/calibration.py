"""
Model calibration for NCAA Men's Basketball win probabilities.

Uses isotonic regression to correct systematic over/under-confidence.
Run on historical predictions vs outcomes to build a calibration curve,
then apply to future predictions before using for Kelly sizing.

Typical usage:
    cal = Calibrator()
    cal.fit(predicted_probs, actual_outcomes)
    calibrated_prob = cal.transform(raw_prob)
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Calibrator:
    """
    Isotonic regression calibrator for win probabilities.

    Isotonic regression is preferred over Platt scaling for sports
    because it makes no parametric assumptions about miscalibration shape.
    """

    def __init__(self):
        self._ir = None
        self._fitted = False

    def fit(self, predicted: List[float], actual: List[int]) -> "Calibrator":
        """
        Fit calibration model.

        Args:
            predicted: Raw model win probabilities [0, 1]
            actual:    Binary outcomes — 1 if predicted team won, 0 if lost

        Returns:
            self (for chaining)
        """
        from sklearn.isotonic import IsotonicRegression

        predicted_arr = np.array(predicted, dtype=float)
        actual_arr    = np.array(actual, dtype=float)

        if len(predicted_arr) < 50:
            logger.warning(
                f"Calibration sample size only {len(predicted_arr)} — "
                "need 200+ games for reliable calibration"
            )

        self._ir = IsotonicRegression(out_of_bounds="clip")
        self._ir.fit(predicted_arr, actual_arr)
        self._fitted = True

        # Diagnostic: compute Brier score before/after
        brier_raw = float(np.mean((predicted_arr - actual_arr) ** 2))
        cal_probs  = self._ir.predict(predicted_arr)
        brier_cal  = float(np.mean((cal_probs - actual_arr) ** 2))
        logger.info(
            f"Calibration fit: n={len(predicted_arr)}, "
            f"Brier raw={brier_raw:.4f}, cal={brier_cal:.4f}"
        )
        return self

    def transform(self, prob: float) -> float:
        """
        Apply calibration to a single probability.
        Returns raw prob if not fitted yet.
        """
        if not self._fitted:
            return prob
        return float(self._ir.predict([prob])[0])

    def transform_all(self, probs: List[float]) -> List[float]:
        """Apply calibration to a list of probabilities."""
        if not self._fitted:
            return probs
        return list(self._ir.predict(np.array(probs, dtype=float)))

    def brier_score(self, predicted: List[float], actual: List[int]) -> float:
        """Brier score (lower = better; random=0.25, perfect=0.0)."""
        p = np.array(predicted, dtype=float)
        y = np.array(actual, dtype=float)
        return float(np.mean((p - y) ** 2))

    def calibration_curve(
        self, predicted: List[float], actual: List[int], n_bins: int = 10
    ) -> List[dict]:
        """
        Compute calibration curve for plotting.

        Returns list of {bin_center, mean_predicted, mean_actual, count}.
        """
        p = np.array(predicted, dtype=float)
        y = np.array(actual, dtype=float)
        bins = np.linspace(0, 1, n_bins + 1)
        result = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi)
            n = int(mask.sum())
            if n == 0:
                continue
            result.append({
                "bin_center":    round((lo + hi) / 2, 3),
                "mean_predicted": round(float(p[mask].mean()), 4),
                "mean_actual":   round(float(y[mask].mean()), 4),
                "count":         n,
            })
        return result

    @property
    def is_fitted(self) -> bool:
        return self._fitted
