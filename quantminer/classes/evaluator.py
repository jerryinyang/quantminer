import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import quantstats as qt
from scipy import stats


@dataclass
class ClusterStats:
    """Statistics for cluster performance evaluation."""

    mean_score: float
    score_std: float
    skewness: float
    kurtosis: float
    confidence_interval: Tuple[float, float]
    sample_size: int


@dataclass
class VectorPatternPerformance:
    """Detailed pattern performance metrics for a cluster."""

    mean_score: float
    median_score: float
    score_std: float
    skewness: float
    kurtosis: float
    confidence_interval: Tuple[float, float]
    directional_consistency: float
    sample_size: int
    statistical_significance: float  # p-value from statistical tests


@dataclass
class HoldingPeriodMetrics:
    """
    Performance metrics for holding period analysis.

    Attributes:
        martin_ratio: Ulcer Performance Index
        sharpe_ratio: Annualized Sharpe ratio
        profit_factor: Ratio of gross profits to gross losses
        win_rate: Percentage of winning trades
        avg_return: Average return per trade
        max_drawdown: Maximum peak to trough decline
        sample_size: Number of trades in analysis
    """

    martin_ratio: float
    sharpe_ratio: float
    profit_factor: float
    win_rate: float
    avg_return: float
    max_drawdown: float
    sample_size: int


@dataclass
class ScoringStats:
    """Statistics for score normalization and interpretation."""

    feature_weights: np.ndarray
    score_mean: float = 0.0
    score_std: float = 1.0
    directional_threshold: float = 0.6


class BaseClusterEvaluator(ABC):
    """Abstract base class for cluster evaluation strategies."""

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.cluster_stats: Dict[int, ClusterStats] = {}
        self.selected_long: List[int] = []
        self.selected_short: List[int] = []
        self.logger = logging.getLogger(__name__)

    def validate_data(self, data: np.ndarray, labels: np.ndarray) -> None:
        """
        Perform basic validation checks on input data and labels.

        Validates:
        1. Data and labels dimensions match
        2. Labels contain valid cluster IDs
        3. Data contains valid numeric values
        4. Sufficient samples for evaluation

        Args:
            data: Input data array
            labels: Cluster label array

        Raises:
            ValueError: If validation fails
            TypeError: If input types are incorrect
        """
        # Type checking
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a numpy array")

        # Dimension matching
        if len(data) != len(labels):
            raise ValueError(
                f"Data and labels must have same length. Got data: {len(data)}, "
                f"labels: {len(labels)}"
            )

        # Label validation
        unique_labels = np.unique(labels)
        if not np.all(unique_labels < self.n_clusters):
            invalid_labels = unique_labels[unique_labels >= self.n_clusters]
            raise ValueError(
                f"Labels contain invalid cluster IDs: {invalid_labels}. "
                f"Max allowed: {self.n_clusters-1}"
            )

        # Data validation
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("Data must contain numeric values")
        if np.any(~np.isfinite(data)):
            raise ValueError("Data contains NaN or infinite values")

        # Sample size validation
        min_samples_per_cluster = 3  # Minimum required for basic statistics
        label_counts = np.bincount(labels[labels >= 0])
        small_clusters = np.where(label_counts < min_samples_per_cluster)[0]
        if len(small_clusters) > 0:
            self.logger.warning(
                f"Clusters {small_clusters} have fewer than {min_samples_per_cluster} samples"
            )

    @abstractmethod
    def evaluate_cluster(
        self, cluster_id: int, data: np.ndarray, labels: np.ndarray
    ) -> ClusterStats:
        """
        Evaluate performance statistics for a single cluster.

        Args:
            cluster_id: Cluster identifier
            data: Full dataset
            labels: Cluster labels

        Returns:
            ClusterStats: Statistical measures of cluster performance
        """
        pass

    @abstractmethod
    def select_clusters(self) -> Tuple[List[int], List[int]]:
        """
        Select best performing clusters for long and short positions.

        Returns:
            Tuple[List[int], List[int]]: Selected cluster IDs for long and short
        """
        pass

    def evaluate_all_clusters(
        self, data: np.ndarray, labels: np.ndarray
    ) -> Dict[int, ClusterStats]:
        """
        Evaluate all clusters and store their statistics.

        Args:
            data: Full dataset
            labels: Cluster labels

        Returns:
            Dict[int, ClusterStats]: Statistics for each cluster

        Raises:
            ValueError: If data validation fails
        """
        # Validate input data
        self.validate_data(data, labels)

        self.cluster_stats.clear()
        evaluation_errors = []

        for cluster_id in range(self.n_clusters):
            try:
                stats = self.evaluate_cluster(cluster_id, data, labels)
                self.cluster_stats[cluster_id] = stats
            except Exception as e:
                self.logger.error(f"Error evaluating cluster {cluster_id}: {e!s}")
                evaluation_errors.append((cluster_id, str(e)))

        if evaluation_errors:
            error_msg = "\n".join(
                f"Cluster {cid}: {err}" for cid, err in evaluation_errors
            )
            self.logger.warning(f"Evaluation errors occurred:\n{error_msg}")

        self.selected_long, self.selected_short = self.select_clusters()
        return self.cluster_stats

    def get_cluster_signal(self, cluster_id: int) -> Optional[int]:
        """
        Get the trading signal (-1, 0, 1) for a given cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Optional[int]: Trading signal or None if cluster not selected
        """
        if cluster_id in self.selected_long:
            return 1
        elif cluster_id in self.selected_short:
            return -1
        return 0


class MovementScorer:
    """
    Scores movement patterns based on their potential to predict significant price moves.

    This class learns optimal weights for movement features (vertical, horizontal, angle,
    speed, acceleration, magnitude, volatility, trend) to identify patterns that lead to
    large, clean directional moves.

    The scoring mechanism is designed to:
    - Maximize correlation with profitable future moves
    - Penalize high volatility/noise
    - Normalize scores to [-1, 1] range for directional interpretation

    Attributes:
        n_features (int): Number of features per movement vector
        min_samples (int): Minimum samples required for fitting
        l2_reg (float): L2 regularization strength
        stats (ScoringStats): Scoring statistics and parameters
    """

    def __init__(
        self,
        n_features: int = 8,  # Default for MVR's FEATURES_PER_PIVOT
        min_samples: int = 100,
        l2_reg: float = 0.01,
    ):
        """
        Initialize the MovementScorer.

        Args:
            n_features: Number of features per movement vector
            min_samples: Minimum samples required for fitting
            l2_reg: L2 regularization strength for weight optimization
        """
        self.n_features = n_features
        self.min_samples = min_samples
        self.l2_reg = l2_reg
        self.stats: Optional[ScoringStats] = None

    def fit(
        self,
        current_vectors: np.ndarray,
        future_vectors: np.ndarray,
        price_changes: np.ndarray,
    ) -> None:
        """
        Learn optimal feature weights from training data.

        The fitting process:
        1. Calculates target scores based on price changes and movement cleanliness
        2. Optimizes feature weights to predict these target scores
        3. Normalizes the scoring mechanism to produce [-1, 1] range

        Args:
            current_vectors: Movement vectors for current windows (n_samples, n_features)
            future_vectors: Sequence of future movement vectors (n_samples, n_future, n_features)
            price_changes: Actual price changes for each sample (n_samples,)

        Raises:
            ValueError: If insufficient samples or invalid input shapes
        """
        if len(current_vectors) < self.min_samples:
            raise ValueError(
                f"Requires at least {self.min_samples} samples for fitting"
            )

        if current_vectors.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {current_vectors.shape[1]}"
            )

        # Calculate target scores based on price changes and movement quality
        target_scores = self._calculate_target_scores(future_vectors, price_changes)

        # Optimize feature weights using regularized linear regression
        weights = self._optimize_weights(current_vectors, target_scores)

        # Calculate score statistics for normalization
        raw_scores = self._compute_raw_scores(current_vectors, weights)
        score_mean = np.mean(raw_scores)
        score_std = np.std(raw_scores)

        # Store scoring statistics
        self.stats = ScoringStats(
            feature_weights=weights,
            score_mean=score_mean,
            score_std=score_std,
            directional_threshold=0.6,  # Can be adjusted based on requirements
        )

    def score_movement(self, vectors: np.ndarray) -> float:
        """
        Score a sequence of movement vectors.

        Args:
            vectors: Movement vectors to score (n_vectors, n_features)

        Returns:
            float: Normalized score between -1 and 1

        Raises:
            ValueError: If scorer hasn't been fitted or invalid input shape
        """
        if self.stats is None:
            raise ValueError("Scorer must be fitted before scoring")

        if vectors.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {vectors.shape[1]}"
            )

        # Compute raw score
        raw_score = self._compute_raw_scores(vectors, self.stats.feature_weights)

        # Normalize score to [-1, 1] range
        normalized_score = self._normalize_score(raw_score)

        return float(normalized_score)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get relative importance of each feature in the scoring model.

        Returns:
            Dict[str, float]: Feature importance scores normalized to sum to 1

        Raises:
            ValueError: If scorer hasn't been fitted
        """
        if self.stats is None:
            raise ValueError("Scorer must be fitted before getting feature importance")

        # Get absolute weights
        abs_weights = np.abs(self.stats.feature_weights)

        # Normalize to sum to 1
        importance_scores = abs_weights / np.sum(abs_weights)

        # Map to feature names (assuming MVR features)
        feature_names = [
            "vertical",
            "horizontal",
            "angle",
            "speed",
            "acceleration",
            "magnitude",
            "volatility",
            "trend",
        ]

        return dict(zip(feature_names[: self.n_features], importance_scores))

    def _calculate_target_scores(
        self, future_vectors: np.ndarray, price_changes: np.ndarray
    ) -> np.ndarray:
        """
        Calculate target scores based on future movement characteristics.

        Considers:
        - Magnitude of price change (larger is better)
        - Directional consistency (less volatility is better)
        - Movement efficiency (more direct moves are better)

        Args:
            future_vectors: Future movement vectors
            price_changes: Actual price changes

        Returns:
            np.ndarray: Target scores for each sample
        """
        # Calculate directional consistency
        directions = np.sign(future_vectors[:, :, 0])  # Vertical movement
        consistency = np.mean(directions == np.sign(price_changes)[:, None], axis=1)

        # Calculate movement efficiency (ratio of net move to total path length)
        net_move = np.abs(price_changes)
        total_path = np.sum(np.abs(future_vectors[:, :, 0]), axis=1)
        efficiency = net_move / (total_path + 1e-10)

        # Calculate normalized price changes
        norm_price_changes = stats.zscore(price_changes)

        # Combine factors into target score
        target_scores = (
            0.4 * norm_price_changes
            + 0.3 * stats.zscore(consistency)
            + 0.3 * stats.zscore(efficiency)
        )

        return target_scores

    def _optimize_weights(
        self, vectors: np.ndarray, target_scores: np.ndarray
    ) -> np.ndarray:
        """
        Optimize feature weights using regularized linear regression.

        Args:
            vectors: Input movement vectors
            target_scores: Target scores to predict

        Returns:
            np.ndarray: Optimized feature weights
        """
        # Add L2 regularization to prevent overfitting
        reg_matrix = self.l2_reg * np.eye(vectors.shape[1])

        # Solve regularized least squares
        weights = np.linalg.solve(
            vectors.T @ vectors + reg_matrix, vectors.T @ target_scores
        )

        return weights

    def _compute_raw_scores(
        self, vectors: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute raw scores using feature weights.

        Args:
            vectors: Input movement vectors
            weights: Feature weights

        Returns:
            np.ndarray: Raw scores
        """
        return vectors @ weights

    def _normalize_score(self, raw_score: float) -> float:
        """
        Normalize raw score to [-1, 1] range.

        Args:
            raw_score: Raw score to normalize

        Returns:
            float: Normalized score
        """
        z_score = (raw_score - self.stats.score_mean) / self.stats.score_std
        return np.clip(z_score / 2, -1, 1)  # Divide by 2 to target [-1, 1] range


class VectorPatternEvaluator(BaseClusterEvaluator):
    """
    Evaluates clusters based on the quality of their subsequent movement patterns.

    This evaluator analyzes how patterns in each cluster tend to evolve by scoring
    their future movement vectors. It uses statistical analysis to identify clusters
    that consistently lead to high-quality directional moves.

    Key features:
    - Uses MovementScorer for pattern quality assessment
    - Performs robust statistical analysis of pattern performance
    - Considers both score magnitude and consistency
    - Accounts for sample size in confidence calculations

    Attributes:
        n_future_vectors (int): Number of future vectors to analyze
        min_samples (int): Minimum samples required per cluster
        confidence_level (float): Confidence level for statistical tests
        scorer (MovementScorer): Movement pattern scorer
        cluster_performance (Dict[int, PatternPerformance]): Detailed cluster metrics
    """

    def __init__(
        self,
        n_clusters: int,
        n_future_vectors: int,
        n_features: int = 8,
        min_samples: int = 50,
        confidence_level: float = 0.95,
        min_movement: float = 0.001,  # Minimum movement threshold for PIP
    ):
        """
        Initialize the VectorPatternEvaluator.

        Args:
            n_clusters: Number of clusters to evaluate
            n_future_vectors: Number of future vectors to analyze per pattern
            n_features: Number of features per vector
            min_samples: Minimum samples required per cluster
            confidence_level: Confidence level for statistical tests
        """

        super().__init__(n_clusters)
        self.n_future_vectors = n_future_vectors
        self.n_features = n_features
        self.min_samples = min_samples
        self.min_movement = min_movement
        self.confidence_level = confidence_level

        # Initialize scorer and performance tracking
        self.scorer = MovementScorer(n_features=n_features, min_samples=min_samples)
        self.cluster_performance: Dict[int, VectorPatternPerformance] = {}

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def evaluate_cluster(
        self, cluster_id: int, data: np.ndarray, labels: np.ndarray
    ) -> ClusterStats:
        """
        Evaluate a single cluster's pattern performance.

        Args:
            cluster_id: Cluster to evaluate
            data: Full movement vector dataset
            labels: Cluster labels

        Returns:
            ClusterStats: Statistical measures of cluster performance

        Raises:
            ValueError: If insufficient samples for evaluation
        """
        # Get indices for this cluster
        cluster_mask = labels == cluster_id
        if np.sum(cluster_mask) < self.min_samples:
            raise ValueError(
                f"Insufficient samples for cluster {cluster_id}: "
                f"{np.sum(cluster_mask)} < {self.min_samples}"
            )

        # Extract current and future vectors for cluster patterns
        current_vectors, future_vectors = self._extract_vector_sequences(
            data, cluster_mask
        )

        # Calculate pattern scores
        pattern_scores = self._calculate_pattern_scores(current_vectors, future_vectors)

        # Compute detailed performance metrics
        performance = self._compute_performance_metrics(pattern_scores)
        self.cluster_performance[cluster_id] = performance

        # Convert to ClusterStats for base class compatibility
        return ClusterStats(
            mean_score=performance.mean_score,
            score_std=performance.score_std,
            skewness=performance.skewness,
            kurtosis=performance.kurtosis,
            confidence_interval=performance.confidence_interval,
            sample_size=performance.sample_size,
        )

    def select_clusters(self) -> Tuple[List[int], List[int]]:
        """
        Select best performing clusters for long and short positions.

        The selection process:
        1. Filters clusters based on statistical significance
        2. Ranks remaining clusters by score magnitude and consistency
        3. Separates into long/short based on score direction

        Returns:
            Tuple[List[int], List[int]]: Selected cluster IDs for long and short
        """
        selected_long = []
        selected_short = []

        # Filter for statistically significant clusters
        significant_clusters = {
            cid: perf
            for cid, perf in self.cluster_performance.items()
            if perf.statistical_significance < 0.05  # Standard significance level
        }

        for cluster_id, performance in significant_clusters.items():
            # Calculate selection score combining magnitude and consistency
            selection_score = (
                abs(performance.mean_score)
                * (1 - performance.score_std)  # Lower std is better
                * performance.directional_consistency
            )

            # Require minimum directional consistency
            if performance.directional_consistency >= 0.6:
                if performance.mean_score > 0:
                    selected_long.append((cluster_id, selection_score))
                else:
                    selected_short.append((cluster_id, selection_score))

        # Sort by selection score and extract cluster IDs
        selected_long = [
            cid for cid, _ in sorted(selected_long, key=lambda x: x[1], reverse=True)
        ]
        selected_short = [
            cid for cid, _ in sorted(selected_short, key=lambda x: x[1], reverse=True)
        ]

        return selected_long, selected_short

    def get_cluster_performance(
        self, cluster_id: int
    ) -> Optional[VectorPatternPerformance]:
        """
        Get detailed performance metrics for a specific cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Optional[PatternPerformance]: Detailed performance metrics or None
        """
        return self.cluster_performance.get(cluster_id)

    def _extract_vector_sequences(
        self, data: np.ndarray, cluster_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract current and future vectors for cluster patterns using PIP transformation.

        This method carefully handles the extraction of movement vectors while ensuring:
        1. Proper window boundaries with no out-of-bounds access
        2. No overlap between current and future windows
        3. Correct temporal alignment of sequences
        4. Validation of sequence consistency and completeness

        The extraction process:
        1. Identifies cluster pattern locations
        2. For each location:
        - Extracts non-overlapping current and future windows
        - Applies PIP transformation to each window
        - Calculates movement features
        - Validates sequence completeness
        3. Ensures all sequences maintain temporal order and alignment

        Args:
            data: Full price dataset of shape (n_samples,)
            cluster_mask: Boolean mask for cluster members of shape (n_samples,)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Current vectors of shape (n_valid_sequences, n_future_vectors, n_features)
                - Future vectors of shape (n_valid_sequences, n_future_vectors, n_features)

        Raises:
            ValueError: If no valid sequences found or sequence validation fails
        """
        cluster_indices = np.where(cluster_mask)[0]
        sequences = []
        future_sequences = []

        # Calculate required window sizes
        window_size = self.n_future_vectors * 2  # Minimum points needed for n vectors
        total_required_size = (
            window_size * 2
        )  # Space for both current and future windows

        # Validate data length
        if len(data) < total_required_size:
            raise ValueError(
                f"Data length {len(data)} insufficient for extraction. "
                f"Need at least {total_required_size} points."
            )

        # Track sequence metadata for validation
        sequence_metadata = []

        for idx in cluster_indices:
            # Boundary check
            if idx + total_required_size > len(data):
                self.logger.debug(
                    f"Skipping sequence at index {idx}: insufficient future data"
                )
                continue

            # Extract non-overlapping windows
            current_start = idx
            current_end = idx + window_size
            future_start = current_end
            future_end = future_start + window_size

            # Validate window separation
            assert current_end == future_start, "Window overlap detected"

            # Extract windows
            current_window = data[current_start:current_end]
            future_window = data[future_start:future_end]

            # Verify window contents
            if np.any(np.isnan(current_window)) or np.any(np.isnan(future_window)):
                self.logger.warning(f"NaN values detected in windows at index {idx}")
                continue

            try:
                # Find pivots for both windows
                current_pivots = self._find_pivots(
                    current_window, self.n_future_vectors + 1
                )
                future_pivots = self._find_pivots(
                    future_window, self.n_future_vectors + 1
                )

                # Verify pivot point counts
                if (
                    len(current_pivots) < self.n_future_vectors + 1
                    or len(future_pivots) < self.n_future_vectors + 1
                ):
                    self.logger.debug(
                        f"Insufficient pivot points at index {idx}: "
                        f"current={len(current_pivots)}, future={len(future_pivots)}"
                    )
                    continue

                # Calculate movement features
                current_vectors = self._calculate_movement_features(
                    current_window, current_pivots
                )
                future_vectors = self._calculate_movement_features(
                    future_window, future_pivots
                )

                # Verify feature calculations
                if (
                    current_vectors.shape[1] != self.n_features
                    or future_vectors.shape[1] != self.n_features
                ):
                    raise ValueError(
                        f"Feature calculation error at index {idx}: "
                        f"current shape={current_vectors.shape}, "
                        f"future shape={future_vectors.shape}"
                    )

                # Store sequence if valid
                if (
                    len(current_vectors) >= self.n_future_vectors
                    and len(future_vectors) >= self.n_future_vectors
                ):
                    # Trim to exact size needed
                    current_vectors = current_vectors[: self.n_future_vectors]
                    future_vectors = future_vectors[: self.n_future_vectors]

                    sequences.append(current_vectors)
                    future_sequences.append(future_vectors)

                    # Store metadata for validation
                    sequence_metadata.append(
                        {
                            "index": idx,
                            "current_range": (current_start, current_end),
                            "future_range": (future_start, future_end),
                            "price_range_current": (
                                np.min(current_window),
                                np.max(current_window),
                            ),
                            "price_range_future": (
                                np.min(future_window),
                                np.max(future_window),
                            ),
                        }
                    )

            except Exception as e:
                self.logger.warning(f"Error processing sequence at index {idx}: {e!s}")
                continue

        if not sequences:
            raise ValueError("No valid sequences found for cluster")

        # Validate sequence alignment
        self._validate_sequence_alignment(sequence_metadata)

        # Convert to arrays with consistent shapes
        current_array = np.array(sequences)
        future_array = np.array(future_sequences)

        # Final shape validation
        expected_shape = (len(sequences), self.n_future_vectors, self.n_features)
        if (
            current_array.shape != expected_shape
            or future_array.shape != expected_shape
        ):
            raise ValueError(
                f"Inconsistent sequence shapes. Expected {expected_shape}, "
                f"got current={current_array.shape}, future={future_array.shape}"
            )

        return current_array, future_array

    def _validate_sequence_alignment(self, metadata: List[Dict]) -> None:
        """
        Validate temporal alignment and consistency of extracted sequences.

        Args:
            metadata: List of sequence metadata dictionaries

        Raises:
            ValueError: If sequence validation fails
        """
        if not metadata:
            return

        # Check temporal ordering
        for i in range(len(metadata) - 1):
            current = metadata[i]
            next_seq = metadata[i + 1]

            # Verify no backwards time travel
            assert (
                current["current_range"][1] <= next_seq["current_range"][0]
            ), f"Sequence overlap detected between indices {current['index']} and {next_seq['index']}"

            # Verify future windows don't overlap with next current window
            assert (
                current["future_range"][1] <= next_seq["current_range"][0]
            ), f"Future-current overlap detected between indices {current['index']} and {next_seq['index']}"

            # Verify consistent window sizes
            current_size = current["current_range"][1] - current["current_range"][0]
            future_size = current["future_range"][1] - current["future_range"][0]
            assert (
                current_size == future_size == self.n_future_vectors * 2
            ), f"Inconsistent window sizes at index {current['index']}"

        # Log sequence statistics
        self.logger.info(
            f"Validated {len(metadata)} sequences with consistent alignment"
        )

    def _calculate_pattern_scores(
        self, current_vectors: np.ndarray, future_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Calculate scores for pattern sequences.

        Args:
            current_vectors: Current movement vectors
            future_vectors: Future movement vectors

        Returns:
            np.ndarray: Pattern scores
        """
        # Calculate price changes for fitting
        price_changes = future_vectors[:, -1, 0] - future_vectors[:, 0, 0]

        # Fit scorer if not already fitted
        if self.scorer.stats is None:
            self.scorer.fit(current_vectors, future_vectors, price_changes)

        # Score each pattern
        scores = np.array(
            [self.scorer.score_movement(future_vec) for future_vec in future_vectors]
        )

        return scores

    def _compute_performance_metrics(
        self, scores: np.ndarray
    ) -> VectorPatternPerformance:
        """
        Compute comprehensive performance metrics for a set of pattern scores.

        Args:
            scores: Pattern scores to analyze

        Returns:
            PatternPerformance: Detailed performance metrics
        """
        # Basic statistics
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        score_std = np.std(scores)
        skewness = stats.skew(scores)
        kurtosis = stats.kurtosis(scores)

        # Confidence interval
        confidence_interval = stats.t.interval(
            self.confidence_level,
            len(scores) - 1,
            loc=mean_score,
            scale=stats.sem(scores),
        )

        # Directional consistency
        directional_consistency = np.mean(np.sign(scores) == np.sign(mean_score))

        # Statistical significance
        _, p_value = stats.ttest_1samp(scores, 0)

        return VectorPatternPerformance(
            mean_score=mean_score,
            median_score=median_score,
            score_std=score_std,
            skewness=skewness,
            kurtosis=kurtosis,
            confidence_interval=confidence_interval,
            directional_consistency=directional_consistency,
            sample_size=len(scores),
            statistical_significance=p_value,
        )

    def _find_pivots(self, data: np.ndarray, n_pivots: int) -> np.ndarray:
        """
        Identify pivot points in the data series using PIP algorithm.

        Args:
            data: Input price data
            n_pivots: Number of pivot points to find

        Returns:
            np.ndarray: Indices of identified pivot points
        """
        # Initialize with endpoints
        pivot_indices = [0, len(data) - 1]
        price_range = np.ptp(data)
        min_movement_abs = price_range * self.min_movement

        while len(pivot_indices) < n_pivots:
            # Initialize variables for finding maximum deviation
            starts = pivot_indices[:-1]
            ends = pivot_indices[1:]
            max_deviation = 0
            max_idx = -1
            insert_idx = -1

            # Find maximum deviation point in each segment
            for i, (start, end) in enumerate(zip(starts, ends)):
                segment = data[start : end + 1]

                # Skip small movements
                if abs(segment[-1] - segment[0]) < min_movement_abs:
                    continue

                # Calculate deviations from line
                x = np.arange(len(segment))
                line = np.interp(x, [0, len(segment) - 1], [segment[0], segment[-1]])
                deviations = np.abs(segment - line)

                segment_max_idx = np.argmax(deviations)
                if deviations[segment_max_idx] > max_deviation:
                    max_deviation = deviations[segment_max_idx]
                    max_idx = start + segment_max_idx
                    insert_idx = i + 1

            if max_idx == -1:
                break

            pivot_indices.insert(insert_idx, max_idx)

        return np.array(sorted(pivot_indices[:n_pivots]))

    def _calculate_movement_features(
        self, data: np.ndarray, pivot_indices: np.ndarray
    ) -> np.ndarray:
        """
        Calculate movement features from pivot points.

        Features calculated:
        1. Vertical movement (price change)
        2. Horizontal movement (time duration)
        3. Movement angle (direction)
        4. Movement speed (rate of change)
        5. Acceleration (changes in speed)
        6. Relative magnitude
        7. Volatility during movement
        8. Trend alignment

        Args:
            data: Price data
            pivot_indices: Indices of pivot points

        Returns:
            np.ndarray: Movement feature vectors
        """
        n_movements = len(pivot_indices) - 1
        vectors = np.zeros((n_movements, self.n_features))

        # Prepare movement segments
        starts = pivot_indices[:-1]
        ends = pivot_indices[1:]

        # Calculate primary features
        price_changes = data[ends] - data[starts]
        time_changes = ends - starts
        angles = np.arctan2(price_changes, time_changes)
        speeds = np.divide(price_changes, time_changes)

        # Store primary features
        vectors[:, 0] = price_changes  # Vertical movement
        vectors[:, 1] = time_changes  # Horizontal movement
        vectors[:, 2] = angles / np.pi  # Normalized angle
        vectors[:, 3] = speeds  # Speed

        # Calculate secondary features
        speed_changes = np.diff(speeds, prepend=speeds[0])
        vectors[:, 4] = speed_changes  # Acceleration

        # Calculate relative magnitude
        avg_magnitude = np.mean(np.abs(price_changes))
        vectors[:, 5] = price_changes / (avg_magnitude + 1e-10)

        # Calculate volatility and trend for each movement
        for i in range(n_movements):
            segment = data[starts[i] : ends[i] + 1]
            if len(segment) > 1:
                vectors[i, 6] = np.std(np.diff(segment))  # Volatility
                vectors[i, 7] = np.polyfit(range(len(segment)), segment, 1)[0]  # Trend

        return vectors

    def get_evaluation_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Generate a summary of evaluation results for all clusters.

        Returns:
            Dict[str, Dict[str, float]]: Summary statistics for each cluster
        """
        summary = {}

        for cluster_id, performance in self.cluster_performance.items():
            summary[f"Cluster_{cluster_id}"] = {
                "mean_score": performance.mean_score,
                "directional_consistency": performance.directional_consistency,
                "statistical_significance": performance.statistical_significance,
                "sample_size": performance.sample_size,
            }

        return summary


class HoldingPeriodEvaluator(BaseClusterEvaluator):
    """
    Evaluates clusters based on forward returns over a fixed holding period.
    This implementation maintains consistency with the existing system's evaluation
    approach while fitting into the new evaluator framework.

    The evaluator:
    1. Applies holding period to cluster signals without overlap
    2. Calculates daily returns during the holding period
    3. Evaluates performance using standard metrics
    4. Selects clusters based on risk-adjusted performance

    Attributes:
        n_clusters: Number of clusters to evaluate
        hold_period: Number of periods to hold positions
        min_samples: Minimum samples required per cluster
        baseline_threshold: Minimum performance vs baseline
        cluster_metrics: Performance metrics per cluster
    """

    def __init__(
        self,
        n_clusters: int,
        hold_period: int = 5,
        min_samples: int = 50,
        baseline_threshold: float = 0.0,
    ):
        """
        Initialize the HoldingPeriodEvaluator.

        Args:
            n_clusters: Number of clusters to evaluate
            hold_period: Number of periods to hold positions
            min_samples: Minimum samples required per cluster
            baseline_threshold: Minimum performance threshold vs baseline
        """
        super().__init__(n_clusters)
        self.hold_period = hold_period
        self.min_samples = min_samples
        self.baseline_threshold = baseline_threshold
        self.cluster_metrics: Dict[int, HoldingPeriodMetrics] = {}

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def evaluate_cluster(
        self, cluster_id: int, data: np.ndarray, labels: np.ndarray
    ) -> ClusterStats:
        """
        Evaluate a single cluster's performance over the holding period.

        The evaluation process:
        1. Identifies cluster instances
        2. Applies holding period rules
        3. Calculates daily returns
        4. Computes performance metrics

        Args:
            cluster_id: Cluster to evaluate
            data: Price series data
            labels: Cluster labels

        Returns:
            ClusterStats: Statistical measures of cluster performance

        Raises:
            ValueError: If insufficient samples for evaluation
        """
        # Validate sample size
        cluster_mask = labels == cluster_id
        sample_size = np.sum(cluster_mask)
        if sample_size < self.min_samples:
            raise ValueError(
                f"Insufficient samples for cluster {cluster_id}: "
                f"{sample_size} < {self.min_samples}"
            )

        # Generate signals for the cluster
        signals = np.zeros_like(data)
        signals[cluster_mask] = 1  # Start with all positive signals

        # Apply holding period to signals
        signals = self._apply_holding_period(signals)

        # Calculate returns
        returns = self._calculate_returns(data, signals)
        returns_series = pd.Series(returns)

        # Calculate metrics
        metrics = HoldingPeriodMetrics(
            martin_ratio=float(qt.stats.ulcer_performance_index(returns_series)),
            sharpe_ratio=float(qt.stats.sharpe(returns_series)),
            profit_factor=float(qt.stats.profit_factor(returns_series)),
            win_rate=float(qt.stats.win_rate(returns_series)),
            avg_return=float(returns_series.mean()),
            max_drawdown=float(qt.stats.max_drawdown(returns_series)),
            sample_size=sample_size,
        )

        self.cluster_metrics[cluster_id] = metrics

        # Calculate statistics for base class
        return ClusterStats(
            mean_score=metrics.avg_return,
            score_std=float(np.std(returns)),
            skewness=float(stats.skew(returns)),
            kurtosis=float(stats.kurtosis(returns)),
            confidence_interval=tuple(
                stats.t.interval(
                    0.95,
                    len(returns) - 1,
                    loc=metrics.avg_return,
                    scale=stats.sem(returns),
                )
            ),
            sample_size=sample_size,
        )

    def select_clusters(self) -> Tuple[List[int], List[int]]:
        """
        Select best performing clusters for long and short positions.

        Selection process:
        1. Calculate combined performance score for each cluster
        2. Filter based on minimum performance thresholds
        3. Sort clusters by performance score
        4. Separate into long and short based on return direction

        Returns:
            Tuple[List[int], List[int]]: Selected cluster IDs for long and short
        """
        selected_long = []
        selected_short = []

        for cluster_id, metrics in self.cluster_metrics.items():
            # Skip clusters with insufficient data
            if metrics.sample_size < self.min_samples:
                continue

            # Calculate combined performance score
            performance_score = (
                0.4 * metrics.martin_ratio
                + 0.3 * metrics.sharpe_ratio
                + 0.2 * metrics.profit_factor
                + 0.1 * metrics.win_rate
            )

            # Apply baseline threshold
            if abs(performance_score) <= self.baseline_threshold:
                continue

            # Classify based on return direction and performance
            if metrics.avg_return > 0 and metrics.martin_ratio > 0:
                selected_long.append((cluster_id, performance_score))
            elif metrics.avg_return < 0 and metrics.martin_ratio > 0:
                selected_short.append((cluster_id, -performance_score))

        # Sort by performance score
        selected_long.sort(key=lambda x: x[1], reverse=True)
        selected_short.sort(key=lambda x: x[1], reverse=True)

        return ([cid for cid, _ in selected_long], [cid for cid, _ in selected_short])

    def _apply_holding_period(self, signals: np.ndarray) -> np.ndarray:
        """
        Apply holding period to trading signals without overlap.

        Args:
            signals: Original trading signals

        Returns:
            np.ndarray: Signals with holding period applied
        """
        new_signals = np.zeros_like(signals)
        nonzero = np.where(signals != 0)[0]
        prev_index = -self.hold_period - 1

        for index in nonzero:
            if index > prev_index + self.hold_period:
                prev_index = index
                start_index = index + 1
                end_index = min(index + self.hold_period + 1, len(signals))
                new_signals[start_index:end_index] = signals[index]

        return new_signals

    def _calculate_returns(self, data: np.ndarray, signals: np.ndarray) -> np.ndarray:
        """
        Calculate returns based on signals and price data.

        Args:
            data: Price series data
            signals: Trading signals with holding period applied

        Returns:
            np.ndarray: Returns series
        """
        # Calculate price returns
        price_returns = np.diff(data) / data[:-1]

        # Align signals with returns (signals[1:] for alignment)
        return signals[1:] * price_returns

    def get_cluster_metrics(self, cluster_id: int) -> Optional[HoldingPeriodMetrics]:
        """
        Get detailed performance metrics for a specific cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Optional[HoldingPeriodMetrics]: Performance metrics or None
        """
        return self.cluster_metrics.get(cluster_id)

    def get_evaluation_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Generate summary of evaluation results for all clusters.

        Returns:
            Dict[str, Dict[str, float]]: Summary statistics for each cluster
        """
        summary = {}

        for cluster_id, metrics in self.cluster_metrics.items():
            summary[f"Cluster_{cluster_id}"] = {
                "martin_ratio": metrics.martin_ratio,
                "sharpe_ratio": metrics.sharpe_ratio,
                "profit_factor": metrics.profit_factor,
                "win_rate": metrics.win_rate,
                "avg_return": metrics.avg_return,
                "max_drawdown": metrics.max_drawdown,
                "sample_size": metrics.sample_size,
            }

        return summary
