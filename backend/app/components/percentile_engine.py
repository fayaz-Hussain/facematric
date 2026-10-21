"""
Percentile Engine Component

Converts golden ratio scores to population percentile rankings using
the empirical distribution derived from the FairFace public dataset.
"""

import os
import json
import csv
from pathlib import Path
from typing import Optional, List, Tuple
from scipy import stats
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "score_reference.json"


class PercentileEngine:
    """
    Converts golden ratio scores (0-100) to percentile rankings and population bands.
    
    Calibrated empirically against the FairFace adult face dataset (CC BY 4.0),
    balanced across 7 race groups and gender.
    Empirical distribution: μ ≈ 59.06, σ ≈ 2.03, within-person SD = 1.26.
    Percentiles are interpolated across 101 empirical quantiles and clamped to [1, 99].
    """
    
    def __init__(
        self,
        reference_path: Optional[str or Path] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None
    ):
        """
        Initialize percentile engine with empirical reference data.
        
        Args:
            reference_path: Path to score_reference.json (default: backend/data/score_reference.json)
            mean: Optional custom mean (fallback/override)
            std: Optional custom std (fallback/override)
        """
        self.quantiles = None
        self.ranks = np.linspace(0, 100, 101)
        self.dataset_name = "FairFace"
        self.sample_size = 2017
        self.within_person_sd = 1.26
        self.is_empirical = False
        self.kde = None
        self.empirical_scores = None
        self.empirical_mean = None
        self.empirical_std = None

        # Custom distribution mode (when custom parameters provided for testing)
        if (mean is not None or std is not None) and reference_path is None:
            self.mean = mean if mean is not None else 59.058
            self.std = std if std is not None else 2.029
            self.dataset_name = "Custom"
            self.distribution = stats.norm(loc=self.mean, scale=self.std)
            norm_cdf_points = np.linspace(0.0001, 0.9999, 101)
            self.quantiles = np.array([self.distribution.ppf(p) for p in norm_cdf_points])
            self.p25 = float(self.distribution.ppf(0.25))
            self.p75 = float(self.distribution.ppf(0.75))
            return

        ref_file = Path(reference_path) if reference_path else DEFAULT_REFERENCE_PATH
        if ref_file.exists():
            try:
                with open(ref_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.quantiles = np.array(data["quantiles_0_to_100"], dtype=np.float64)
                self.dataset_name = data.get("dataset_name", "FairFace")
                self.sample_size = data.get("n_accepted_total", 2017)
                self.within_person_sd = data.get("within_person_sd", 1.26)
                self.mean = float(data.get("pooled_mean", 59.058))
                self.std = float(data.get("pooled_std", 2.029))
                self.p25 = float(self.quantiles[25])
                self.p75 = float(self.quantiles[75])
                self.distribution = stats.norm(loc=self.mean, scale=self.std)
                logger.info(f"Loaded empirical percentile reference from {ref_file} (N={self.sample_size})")
                return
            except Exception as e:
                logger.error(f"Failed to load {ref_file}: {e}")

        # Fallback default
        self.mean = 59.058
        self.std = 2.029
        self.distribution = stats.norm(loc=self.mean, scale=self.std)
        self.quantiles = np.array([self.distribution.ppf(p) for p in np.linspace(0.0001, 0.9999, 101)])
        self.p25 = float(self.distribution.ppf(0.25))
        self.p75 = float(self.distribution.ppf(0.75))

    def score_to_percentile(self, score: float) -> int:
        """
        Convert golden ratio score to percentile rank (clamped to 1-99, monotonic).
        
        Uses piecewise linear interpolation against empirical quantiles.
        
        Args:
            score: Golden ratio score (0-100)
            
        Returns:
            Percentile rank (1-99)
        """
        score = max(0.0, min(100.0, float(score)))
        
        # Piecewise linear interpolation across 101 empirical quantiles
        raw_pct = float(np.interp(score, self.quantiles, self.ranks))
        
        # Round to integer and clamp strictly to [1, 99] (Requirement 4f)
        percentile_int = int(round(raw_pct))
        percentile_int = max(1, min(99, percentile_int))
        
        return percentile_int

    def get_percentile_range(self, score: float) -> Tuple[int, int]:
        """
        Compute typically expected percentile range based on within-person SD.
        
        Args:
            score: Raw golden ratio score
            
        Returns:
            Tuple of (Plow, Phigh) rounded to whole numbers
        """
        p_low = self.score_to_percentile(score - self.within_person_sd)
        p_high = self.score_to_percentile(score + self.within_person_sd)
        return p_low, p_high

    def get_percentile_wording(self, score: float, percentile: Optional[int] = None) -> dict:
        """
        Generate honest results page wording per Requirement 4g.
        """
        p = percentile if percentile is not None else self.score_to_percentile(score)
        p_low, p_high = self.get_percentile_range(score)
        
        main_text = (
            f"About the {p}th percentile: higher than about {p}% of {self.sample_size} "
            f"faces in the {self.dataset_name} dataset (adults, balanced across 7 groups)."
        )
        range_text = (
            f"Different photos of you typically land between the {p_low}th and {p_high}th percentile"
        )
        return {
            "percentile": p,
            "p_low": p_low,
            "p_high": p_high,
            "main_message": main_text,
            "range_message": range_text
        }

    def get_band(self, score: float) -> str:
        """Classify score into population band based on quartiles."""
        if score < self.p25:
            return "below typical"
        elif score <= self.p75:
            return "typical"
        else:
            return "above typical"

    def get_band_message(self, score: float) -> str:
        """Generate band message comparing against the benchmark dataset."""
        band = self.get_band(score)
        return f"{band} (compared with {self.sample_size} faces in the {self.dataset_name} dataset)"

    def get_percentile_message(self, percentile: int, score: Optional[float] = None) -> str:
        """Generate band message comparing against benchmark dataset."""
        if score is not None:
            return self.get_band_message(score)
        if percentile < 25:
            band = "below typical"
        elif percentile <= 75:
            band = "typical"
        else:
            band = "above typical"
        return f"{band} (compared with {self.sample_size} faces in the {self.dataset_name} dataset)"

    def update_distribution(self, user_scores: list[float], min_samples: int = 100) -> bool:
        """Update distribution parameters based on real scores."""
        if len(user_scores) < min_samples:
            return False
        scores_arr = np.array(user_scores, dtype=np.float64)
        self.mean = float(np.mean(scores_arr))
        self.std = float(np.std(scores_arr))
        self.distribution = stats.norm(loc=self.mean, scale=self.std)
        self.quantiles = np.percentile(scores_arr, self.ranks)
        self.p25 = float(self.quantiles[25])
        self.p75 = float(self.quantiles[75])
        self.sample_size = len(user_scores)
        return True

    def load_empirical_distribution(self, csv_path: str) -> None:
        """Load empirical distribution from CSV file."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        scores = self._parse_csv_scores(csv_path)
        if len(scores) < 100:
            raise ValueError(f"Insufficient samples: {len(scores)} provided, 100 required")
        scores_arr = np.array(scores, dtype=np.float64)
        self.quantiles = np.percentile(scores_arr, self.ranks)
        self.mean = float(np.mean(scores_arr))
        self.std = float(np.std(scores_arr))
        self.p25 = float(self.quantiles[25])
        self.p75 = float(self.quantiles[75])
        self.sample_size = len(scores)
        self.empirical_scores = scores_arr
        self.empirical_mean = float(np.mean(scores_arr))
        self.empirical_std = float(np.std(scores_arr))
        self.kde = stats.gaussian_kde(scores)
        self.is_empirical = True
        self.distribution = stats.norm(loc=self.mean, scale=self.std)

    def _parse_csv_scores(self, csv_path: Path) -> list[float]:
        """Parse CSV file and extract scores."""
        scores = []
        score_column = None
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file appears to be empty")
            for fieldname in reader.fieldnames:
                if fieldname.lower() in ["score", "golden_ratio_score", "golden_ratio"]:
                    score_column = fieldname
                    break
            if score_column is None:
                raise ValueError(f"Score column not found in {reader.fieldnames}")
            for row in reader:
                val = row.get(score_column, "").strip()
                if val:
                    try:
                        s = float(val)
                        if 0 <= s <= 100:
                            scores.append(s)
                    except ValueError:
                        pass
        return scores


# Global instance loaded from backend/data/score_reference.json
percentile_engine = PercentileEngine()
