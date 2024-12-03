from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .reducers import BaseReducer


@dataclass
class MovementStats:
    """Statistics for movement normalization and feature calculation."""

    price_std: float = 1.0
    time_std: float = 1.0
    speed_std: float = 1.0
    acceleration_std: float = 1.0
    volatility_std: float = 1.0
    magnitude_std: float = 1.0
    price_mean: float = 0.0
    speed_mean: float = 0.0
    volatility_mean: float = 0.0

    def update(self, stats_dict: Dict[str, float]) -> None:
        """Vectorized update of multiple statistics at once."""
        for key, value in stats_dict.items():
            setattr(self, key, value)


class ReducerMVR(BaseReducer):
    """
    Movement Vector Reducer that transforms price series into rich movement-based features.

    This reducer captures comprehensive characteristics of price movements between pivot
    points, including direction, magnitude, speed, acceleration, and market context.

    Features per movement:
    1. Primary Features:
        - Vertical movement (price change)
        - Horizontal movement (time duration)
        - Movement angle (direction)
        - Movement speed (rate of change)
    2. Secondary Features:
        - Acceleration (changes in speed)
        - Relative magnitude (compared to recent movements)
        - Volatility during movement
        - Trend alignment

    Args:
        n_pivots: Number of features to return (must be divisible by FEATURES_PER_PIVOT)
        normalize_window: Window size for normalization calculations
        min_movement: Minimum movement threshold as fraction of price
        feature_selection: Optional list of specific features to use
        trend_window: Window size for trend calculations
        alpha: Exponential forgetting factor for statistics (0-1)
    """

    FEATURES_PER_PIVOT = 8
    FEATURE_NAMES = [
        "vertical",
        "horizontal",
        "angle",
        "speed",
        "acceleration",
        "magnitude",
        "volatility",
        "trend",
    ]

    def __init__(
        self,
        n_pivots: int,
        normalize_window: int = 100,
        min_movement: float = 0.001,
        feature_selection: Optional[List[str]] = None,
        trend_window: int = 20,
        alpha: float = 0.1,
    ):
        # Validate and setup feature selection
        if feature_selection:
            self.selected_features = np.array(
                [
                    i
                    for i, name in enumerate(self.FEATURE_NAMES)
                    if name in feature_selection
                ]
            )
            effective_features = len(self.selected_features)
        else:
            self.selected_features = np.arange(self.FEATURES_PER_PIVOT)
            effective_features = self.FEATURES_PER_PIVOT

        if n_pivots % effective_features != 0:
            raise ValueError(
                f"n_pivots ({n_pivots}) must be divisible by number of "
                f"selected features ({effective_features})"
            )

        self.n_pivots = n_pivots
        self.n_actual_pivots = n_pivots // effective_features
        self.normalize_window = normalize_window
        self.min_movement = min_movement
        self.trend_window = trend_window
        self.alpha = alpha

        # Initialize statistics and history
        self.stats = MovementStats()
        self.movement_history = deque(maxlen=normalize_window)
        self._last_movement = np.zeros(self.FEATURES_PER_PIVOT)

    def transform(self, data: Union[NDArray, List[NDArray]]) -> NDArray:
        """
        Transform price series into movement vectors.

        Args:
            data: Input price series data (single series or list of series)

        Returns:
            NDArray: Transformed movement features
        """
        # Validate input
        data = self._validate_input(data, min_length=3)

        # Handle multiple series case with vectorized operation
        if data.ndim > 1:
            return np.vstack([self.transform(series) for series in data])

        # Update statistics
        self._update_statistics(data)

        # Find pivot points and calculate movements
        pivot_indices = self._find_pivots(data)
        vectors = self._calculate_movement_vectors(data, pivot_indices)

        # Select features and pack into output format
        selected_vectors = vectors[:, self.selected_features]
        return self._pack_vectors(selected_vectors)

    def _update_statistics(self, data: NDArray) -> None:
        """Vectorized update of movement statistics."""
        changes = np.diff(data)
        window = min(self.normalize_window, len(data))
        recent_data = data[-window:]
        recent_changes = changes[-window + 1 :]

        # Vectorized statistics calculation
        stats_update = {
            "price_std": np.std(changes) + 1e-10,
            "price_mean": np.mean(recent_data),
            "speed_std": np.std(recent_changes / np.arange(1, len(recent_changes) + 1))
            + 1e-10,
            "speed_mean": np.mean(np.abs(recent_changes))
            / np.mean(np.arange(1, len(recent_changes) + 1)),
            "volatility_std": np.std(np.abs(recent_changes)) + 1e-10,
            "volatility_mean": np.mean(np.abs(recent_changes)),
        }

        # Apply exponential forgetting
        for key, value in stats_update.items():
            current = getattr(self.stats, key)
            stats_update[key] = (1 - self.alpha) * current + self.alpha * value

        self.stats.update(stats_update)

    def _find_pivots(self, data: NDArray) -> NDArray:
        """
        Vectorized pivot point identification.

        Uses a modified PIP algorithm optimized for vectorized operations.
        """
        # Initialize with endpoints
        pivot_indices = [0, len(data) - 1]
        price_range = np.ptp(data)
        min_movement_abs = price_range * self.min_movement

        while len(pivot_indices) < self.n_actual_pivots:
            # Prepare segments for vectorized operations
            starts = pivot_indices[:-1]
            ends = pivot_indices[1:]
            max_deviation = 0
            max_idx = -1
            insert_idx = -1

            # Vectorized deviation calculation for all segments
            for i, (start, end) in enumerate(zip(starts, ends)):
                segment = data[start : end + 1]

                # Skip small movements
                if abs(segment[-1] - segment[0]) < min_movement_abs:
                    continue

                # Vectorized line and deviation calculation
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

        return np.array(sorted(pivot_indices[: self.n_actual_pivots]))

    def _calculate_movement_vectors(
        self, data: NDArray, pivot_indices: NDArray
    ) -> NDArray:
        """Vectorized calculation of movement vectors."""
        n_movements = len(pivot_indices) - 1
        vectors = np.zeros((n_movements, self.FEATURES_PER_PIVOT))

        # Calculate trends once
        trends = self._calculate_trends(data)

        # Prepare movement segments
        starts = pivot_indices[:-1]
        ends = pivot_indices[1:]

        # Vectorized primary feature calculation
        price_changes = data[ends] - data[starts]
        time_changes = ends - starts
        angles = np.arctan2(price_changes, time_changes)
        speeds = np.divide(price_changes, time_changes)

        # Store primary features
        vectors[:, 0] = (price_changes - self.stats.price_mean) / self.stats.price_std
        vectors[:, 1] = time_changes / self.stats.time_std
        vectors[:, 2] = angles / np.pi
        vectors[:, 3] = (speeds - self.stats.speed_mean) / self.stats.speed_std

        # Vectorized secondary feature calculation
        # Acceleration
        speed_changes = np.diff(speeds, prepend=self._last_movement[3])
        vectors[:, 4] = speed_changes / self.stats.acceleration_std

        # Relative magnitude
        if self.movement_history:
            recent_movements = np.array(list(self.movement_history))
            avg_magnitude = np.mean(np.abs(recent_movements[:, 0]))
            vectors[:, 5] = price_changes / (avg_magnitude + 1e-10)
        else:
            vectors[:, 5] = np.ones(n_movements)

        # Volatility and trend alignment
        for i in range(n_movements):
            segment = data[starts[i] : ends[i] + 1]
            segment_trends = trends[starts[i] : ends[i] + 1]

            # Volatility
            if len(segment) > 1:
                vectors[i, 6] = np.std(np.diff(segment)) / self.stats.volatility_std

            # Trend alignment
            vectors[i, 7] = np.mean(segment_trends)

        # Update history
        self._last_movement = vectors[-1] if n_movements > 0 else self._last_movement
        self.movement_history.append(
            vectors[-1] if n_movements > 0 else np.zeros(self.FEATURES_PER_PIVOT)
        )

        return vectors

    def _calculate_trends(self, data: NDArray) -> NDArray:
        """Vectorized trend calculation."""
        if len(data) < self.trend_window:
            return np.zeros_like(data)

        # Vectorized moving average
        weights = np.ones(self.trend_window) / self.trend_window
        ma = np.convolve(data, weights, mode="valid")

        # Pad to match original length
        pad_size = len(data) - len(ma)
        ma = np.pad(ma, (pad_size, 0), mode="edge")

        # Vectorized trend alignment calculation
        return np.clip((data - ma) / (np.std(data) + 1e-10), -1, 1)

    def _pack_vectors(self, vectors: NDArray) -> NDArray:
        """Vectorized vector packing."""
        if len(vectors) < self.n_actual_pivots:
            pad_size = self.n_actual_pivots - len(vectors)
            vectors = np.pad(vectors, ((0, pad_size), (0, 0)), mode="constant")

        return vectors.flatten()[: self.n_pivots]

    def get_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance using vectorized operations."""
        if not self.movement_history:
            return dict(zip(self.FEATURE_NAMES, np.ones(self.FEATURES_PER_PIVOT)))

        movements = np.array(list(self.movement_history))
        variances = np.var(movements, axis=0)
        importance = variances / (np.sum(variances) + 1e-10)

        return dict(zip(self.FEATURE_NAMES, importance))
