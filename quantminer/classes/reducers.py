from enum import Enum
from typing import List, Union

import numpy as np
import pywt
from numpy.typing import NDArray


class DistanceMeasure(Enum):
    """Distance measures for PIP reducer"""

    EUCLIDEAN = 1
    PERPENDICULAR = 2
    VERTICAL = 3


class BaseReducer:
    """Base class for all reducers with common validation logic"""

    def _validate_input(
        self, data: Union[NDArray, List[NDArray]], min_length: int
    ) -> NDArray:
        """
        Validates and converts input data to proper format

        Args:
            data: Input data to validate
            min_length: Minimum required length

        Returns:
            NDArray: Validated numpy array

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(data, (np.ndarray, list)):
            raise ValueError("Input must be numpy array or list")

        arr = np.asarray(data)
        if arr.size == 0:
            raise ValueError("Empty input array")

        if arr.ndim > 1 and any(len(x) < min_length for x in arr):
            raise ValueError(f"All sequences must have length >= {min_length}")
        elif arr.ndim == 1 and len(arr) < min_length:
            raise ValueError(f"Input length must be >= {min_length}")

        return arr


class ReducerPIP(BaseReducer):
    """
    Perceptually Important Points (PIP) reducer that identifies key points in time series data.
    Uses vectorized operations for improved performance.
    """

    def __init__(
        self, n_pivots: int, dist_measure: Union[DistanceMeasure, int]
    ) -> None:
        """
        Initialize PIP reducer

        Args:
            n_pivots: Number of pivot points to extract
            dist_measure: Distance measure to use (enum or int 1-3)
        """
        self.n_pivots = n_pivots
        self.dist_measure = (
            dist_measure
            if isinstance(dist_measure, DistanceMeasure)
            else DistanceMeasure(dist_measure)
        )

    def transform(self, data: Union[NDArray, List[NDArray]]) -> NDArray:
        """
        Transform data using PIP algorithm with vectorized operations

        Args:
            data: Input time series data

        Returns:
            NDArray: Array of pivot points
        """
        data = self._validate_input(data, self.n_pivots)

        if data.ndim > 1:
            return np.array([self.transform(seq) for seq in data])

        pivots = np.zeros(self.n_pivots)
        indices = np.zeros(self.n_pivots, dtype=int)

        # Initialize with endpoints
        pivots[0], pivots[-1] = data[0], data[-1]
        indices[0], indices[-1] = 0, len(data) - 1

        # Find remaining pivots
        for i in range(2, self.n_pivots):
            max_dist = 0
            max_idx = -1
            insert_idx = -1

            # Vectorized distance calculation
            for j in range(i - 1):
                left_idx, right_idx = indices[j], indices[j + 1]
                segment = data[left_idx : right_idx + 1]
                segment_indices = np.arange(left_idx, right_idx + 1)

                if self.dist_measure == DistanceMeasure.EUCLIDEAN:
                    dist = self._euclidean_distance(
                        segment,
                        segment_indices,
                        pivots[j],
                        indices[j],
                        pivots[j + 1],
                        indices[j + 1],
                    )
                elif self.dist_measure == DistanceMeasure.PERPENDICULAR:
                    dist = self._perpendicular_distance(
                        segment,
                        segment_indices,
                        pivots[j],
                        indices[j],
                        pivots[j + 1],
                        indices[j + 1],
                    )
                else:  # VERTICAL
                    dist = self._vertical_distance(
                        segment,
                        segment_indices,
                        pivots[j],
                        indices[j],
                        pivots[j + 1],
                        indices[j + 1],
                    )

                max_dist_idx = np.argmax(dist)
                if dist[max_dist_idx] > max_dist:
                    max_dist = dist[max_dist_idx]
                    max_idx = segment_indices[max_dist_idx]
                    insert_idx = j + 1

            indices = np.insert(indices[:i], insert_idx, max_idx)
            pivots = np.insert(pivots[:i], insert_idx, data[max_idx])

        return pivots

    def _euclidean_distance(
        self,
        segment: NDArray,
        indices: NDArray,
        p1_val: float,
        p1_idx: int,
        p2_val: float,
        p2_idx: int,
    ) -> NDArray:
        """Vectorized Euclidean distance calculation"""
        d1 = np.sqrt((p1_idx - indices) ** 2 + (p1_val - segment) ** 2)
        d2 = np.sqrt((p2_idx - indices) ** 2 + (p2_val - segment) ** 2)
        return d1 + d2

    def _perpendicular_distance(
        self,
        segment: NDArray,
        indices: NDArray,
        p1_val: float,
        p1_idx: int,
        p2_val: float,
        p2_idx: int,
    ) -> NDArray:
        """Vectorized perpendicular distance calculation"""
        dx = p2_idx - p1_idx
        dy = p2_val - p1_val
        slope = np.divide(dy, dx, out=np.zeros_like(dy), where=dx != 0)
        intercept = p1_val - slope * p1_idx
        return np.abs(slope * indices + intercept - segment) / np.sqrt(slope**2 + 1)

    def _vertical_distance(
        self,
        segment: NDArray,
        indices: NDArray,
        p1_val: float,
        p1_idx: int,
        p2_val: float,
        p2_idx: int,
    ) -> NDArray:
        """Vectorized vertical distance calculation"""
        dx = p2_idx - p1_idx
        dy = p2_val - p1_val
        slope = np.divide(dy, dx, out=np.zeros_like(dy), where=dx != 0)
        intercept = p1_val - slope * p1_idx
        return np.abs(slope * indices + intercept - segment)


class ReducerFFT(BaseReducer):
    """Fast Fourier Transform based reducer with optimized component selection"""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def transform(self, data: Union[NDArray, List[NDArray]]) -> NDArray:
        """
        Transform using FFT with optimized component selection

        Args:
            data: Input time series

        Returns:
            NDArray: Top frequency components
        """
        data = self._validate_input(data, self.n_components)

        if data.ndim > 1:
            return np.array([self.transform(seq) for seq in data])

        fft_result = np.fft.fft(data)
        magnitudes = np.abs(fft_result)

        # Optimize component selection
        top_indices = np.argpartition(magnitudes, -self.n_components)[
            -self.n_components :
        ]
        return magnitudes[np.sort(top_indices)]


class ReducerWavelet(BaseReducer):
    """Wavelet transform based reducer with coefficient optimization"""

    def __init__(self, n_coefficients: int, wavelet: str = "coif1") -> None:
        self.n_coefficients = n_coefficients
        self.wavelet = wavelet

    def transform(self, data: Union[NDArray, List[NDArray]]) -> NDArray:
        """
        Transform using wavelet decomposition

        Args:
            data: Input time series

        Returns:
            NDArray: Selected wavelet coefficients
        """
        data = self._validate_input(data, self.n_coefficients)

        if data.ndim > 1:
            return np.array([self.transform(seq) for seq in data])

        coeffs = pywt.wavedec(data, self.wavelet, mode="symmetric")
        flat_coeffs = np.concatenate(coeffs)

        # Optimize coefficient selection
        top_indices = np.argpartition(np.abs(flat_coeffs), -self.n_coefficients)[
            -self.n_coefficients :
        ]
        return flat_coeffs[np.sort(top_indices)]


class ReducerFFTWavelet(BaseReducer):
    """Combined FFT and Wavelet reducer with balanced component selection"""

    def __init__(self, n_components: int, wavelet: str = "db1") -> None:
        self.n_fourier = n_components // 2
        self.n_wavelet = (
            n_components - self.n_fourier
        )  # Ensure total equals n_components
        self.wavelet = wavelet

    def transform(self, data: Union[NDArray, List[NDArray]]) -> NDArray:
        """
        Transform using combined FFT and wavelet analysis

        Args:
            data: Input time series

        Returns:
            NDArray: Combined frequency and wavelet features
        """
        data = self._validate_input(data, max(self.n_fourier, self.n_wavelet))

        if data.ndim > 1:
            return np.array([self.transform(seq) for seq in data])

        # Optimized FFT computation
        fft_result = np.fft.fft(data)
        magnitudes = np.abs(fft_result)
        top_freq_indices = np.argpartition(magnitudes, -self.n_fourier)[
            -self.n_fourier :
        ]
        top_frequencies = magnitudes[np.sort(top_freq_indices)]

        # Optimized wavelet computation
        coeffs = pywt.wavedec(data, self.wavelet, mode="symmetric")
        flat_coeffs = np.concatenate(coeffs)
        top_wav_indices = np.argpartition(np.abs(flat_coeffs), -self.n_wavelet)[
            -self.n_wavelet :
        ]
        top_wavelets = flat_coeffs[np.sort(top_wav_indices)]

        return np.concatenate([top_frequencies, top_wavelets])
