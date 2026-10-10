"""
Percentile Engine Component

Converts golden ratio scores to population percentile rankings.
"""

import os
import csv
from pathlib import Path
from scipy import stats
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


class PercentileEngine:
    """
    Converts golden ratio scores (0-100) to percentile rankings.
    
    Uses a bootstrap normal distribution until sufficient real user data collected.
    Distribution: N(μ=70, σ=10)
    """
    
    def __init__(self, mean: float = 70.0, std: float = 10.0):
        """
        Initialize percentile engine with distribution parameters.
        
        Args:
            mean: Mean of score distribution (default: 70)
            std: Standard deviation of score distribution (default: 10)
        """
        self.mean = mean
        self.std = std
        self.distribution = stats.norm(loc=mean, scale=std)
        
        # Initialize empirical distribution attributes
        self.kde = None
        self.is_empirical = False
        self.empirical_scores = None
        self.empirical_mean = None
        self.empirical_std = None
        
        logger.info(
            f"PercentileEngine initialized with bootstrap distribution: "
            f"N(μ={mean}, σ={std})"
        )
    
    def score_to_percentile(self, score: float) -> int:
        """
        Convert golden ratio score to percentile rank.
        
        Uses empirical distribution if loaded, otherwise uses bootstrap distribution.
        
        Args:
            score: Golden ratio score (0-100)
        
        Returns:
            Percentile rank (0-100) representing percentage of population scored below
        """
        # Clamp score to valid range
        score = max(0.0, min(100.0, score))
        
        # Use empirical distribution if available
        if hasattr(self, 'is_empirical') and self.is_empirical and self.kde is not None:
            # Calculate percentile using empirical CDF
            # Count scores less than or equal to current score
            percentile = (np.sum(self.empirical_scores <= score) / len(self.empirical_scores)) * 100
        else:
            # Use bootstrap distribution CDF
            percentile = self.distribution.cdf(score) * 100
        
        # Round to integer and clamp to [0, 100]
        percentile_int = int(round(percentile))
        percentile_int = max(0, min(100, percentile_int))
        
        distribution_type = "empirical" if (hasattr(self, 'is_empirical') and self.is_empirical) else "bootstrap"
        logger.debug(f"Score {score:.1f} → Percentile {percentile_int} ({distribution_type})")
        
        return percentile_int
    
    def update_distribution(
        self,
        user_scores: list[float],
        min_samples: int = 100
    ) -> bool:
        """
        Update distribution parameters based on real user scores.
        
        This method allows the system to learn from actual user data
        and replace the bootstrap distribution with empirical statistics.
        
        Args:
            user_scores: List of historical user scores
            min_samples: Minimum number of scores required to update (default: 100)
        
        Returns:
            True if distribution was updated, False if insufficient data
        """
        if len(user_scores) < min_samples:
            logger.warning(
                f"Insufficient samples to update distribution: "
                f"{len(user_scores)}/{min_samples}"
            )
            return False
        
        # Calculate empirical statistics
        scores_array = np.array(user_scores)
        new_mean = float(np.mean(scores_array))
        new_std = float(np.std(scores_array))
        
        # Update distribution
        self.mean = new_mean
        self.std = new_std
        self.distribution = stats.norm(loc=new_mean, scale=new_std)
        
        logger.info(
            f"Distribution updated from {len(user_scores)} samples: "
            f"N(μ={new_mean:.2f}, σ={new_std:.2f})"
        )
        
        return True
    
    def get_percentile_message(self, percentile: int) -> str:
        """
        Generate user-friendly percentile message.
        
        Args:
            percentile: Percentile rank (0-100)
        
        Returns:
            Formatted message string
        """
        return f"You score higher than {percentile}% of people"
    
    def load_empirical_distribution(self, csv_path: str) -> None:
        """
        Load empirical distribution from CSV file of golden ratio scores.
        
        Loads real user golden ratio scores from a CSV file, fits an empirical
        distribution, and replaces the bootstrap normal distribution with the
        fitted empirical distribution. This enables percentile rankings based
        on actual population data rather than theoretical assumptions.
        
        **CSV Format:**
        - File should contain a header row
        - Required column: "score" or "golden_ratio_score" (case-insensitive)
        - All values in score column must be numeric (0-100)
        - Rows with missing or non-numeric scores will be skipped with warning
        
        **Empirical Distribution:**
        The empirical distribution is fitted using Gaussian KDE (Kernel Density
        Estimation) with a Gaussian kernel. This allows the distribution to adapt
        to the actual shape of the population data without assuming normality.
        
        **Minimum Data Requirements:**
        - At least 100 unique scores required for reliable empirical fitting
        - If fewer than 100 scores provided, method logs warning and aborts
        - Bootstrap distribution remains unchanged if loading fails
        
        Args:
            csv_path: Path to CSV file containing score data
        
        Raises:
            FileNotFoundError: If CSV file does not exist
            ValueError: If CSV is malformed or score column not found
            IOError: If file cannot be read
        
        Examples:
            >>> engine = PercentileEngine()
            >>> engine.load_empirical_distribution("user_scores.csv")
            >>> # Now percentile rankings based on empirical distribution
        
        """
        csv_path = Path(csv_path)
        
        # Validate file exists
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Validate file is readable
        if not os.access(csv_path, os.R_OK):
            logger.error(f"CSV file not readable: {csv_path}")
            raise IOError(f"CSV file not readable: {csv_path}")
        
        try:
            scores = self._parse_csv_scores(csv_path)
        except (ValueError, csv.Error) as e:
            logger.error(f"Failed to parse CSV file: {str(e)}")
            raise ValueError(f"Failed to parse CSV file: {str(e)}")
        
        # Validate minimum sample size
        if len(scores) < 100:
            logger.warning(
                f"Insufficient samples to fit empirical distribution: "
                f"{len(scores)}/100 minimum"
            )
            raise ValueError(
                f"Insufficient samples: {len(scores)} provided, 100 required"
            )
        
        # Fit empirical distribution using Gaussian KDE
        try:
            kde = stats.gaussian_kde(scores)
            logger.info(
                f"Empirical distribution fitted from {len(scores)} scores using "
                f"Gaussian KDE"
            )
        except Exception as e:
            logger.error(f"Failed to fit Gaussian KDE: {str(e)}")
            raise ValueError(f"Failed to fit distribution: {str(e)}")
        
        # Replace bootstrap distribution with empirical distribution
        # Store KDE and mark that we're using empirical distribution
        self.kde = kde
        self.is_empirical = True
        self.empirical_scores = np.array(scores)
        self.empirical_mean = float(np.mean(scores))
        self.empirical_std = float(np.std(scores))
        
        logger.info(
            f"Empirical distribution loaded successfully. "
            f"Mean: {self.empirical_mean:.2f}, Std: {self.empirical_std:.2f}. "
            f"Empirical distribution loaded"
        )
    
    def _parse_csv_scores(self, csv_path: Path) -> list[float]:
        """
        Parse CSV file and extract golden ratio scores.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            List of valid numeric scores
        
        Raises:
            ValueError: If CSV format invalid or score column not found
            csv.Error: If CSV parsing fails
        """
        scores = []
        score_column = None
        rows_skipped = 0
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Find score column (case-insensitive)
                if reader.fieldnames is None:
                    raise ValueError("CSV file appears to be empty")
                
                for fieldname in reader.fieldnames:
                    if fieldname.lower() in ['score', 'golden_ratio_score', 'golden_ratio']:
                        score_column = fieldname
                        break
                
                if score_column is None:
                    raise ValueError(
                        f"Score column not found. Expected 'score' or "
                        f"'golden_ratio_score', found columns: {reader.fieldnames}"
                    )
                
                # Extract and validate scores
                for row_idx, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                    score_str = row.get(score_column, '').strip()
                    
                    if not score_str:
                        rows_skipped += 1
                        continue
                    
                    try:
                        score = float(score_str)
                        
                        # Validate score in valid range (0-100)
                        if not (0 <= score <= 100):
                            logger.warning(
                                f"Row {row_idx}: Score {score} out of range [0-100], "
                                f"skipping"
                            )
                            rows_skipped += 1
                            continue
                        
                        scores.append(score)
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Row {row_idx}: Invalid score value '{score_str}', "
                            f"skipping"
                        )
                        rows_skipped += 1
                        continue
        
        except UnicodeDecodeError as e:
            raise ValueError(f"File encoding error: {str(e)}")
        except IOError as e:
            raise ValueError(f"File read error: {str(e)}")
        
        if len(scores) == 0:
            raise ValueError("No valid scores found in CSV file")
        
        if rows_skipped > 0:
            logger.info(
                f"Parsed {len(scores)} valid scores from CSV, "
                f"skipped {rows_skipped} invalid rows"
            )
        
        return scores


# Global instance
percentile_engine = PercentileEngine()

