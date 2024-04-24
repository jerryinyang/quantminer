from typing import List, Union

import numpy as np
import pywt


class ReducerPIP:
    def __init__(self, n_pivots: int, dist_measure: int) -> None:
        """
        Initialize the ReducerPIP with specified number of pivots and a chosen distance measure.
        
        This reducer implements the Perceptually Important Points (PIP) method, which selects key points
        in the data series that capture significant movements or trends while reducing the dimensionality
        of the data.

        Arguments:
        - n_pivots: int
            The number of pivots or key points to identify in the dataset. These pivots aim to capture
            the most informative aspects of the data.
        - dist_measure: int
            The metric used to measure distance when identifying pivots. The options are:
            * 1 = Euclidean Distance - Measures the straight line distance between points.
            * 2 = Perpendicular Distance - Measures the shortest distance to the line segment between adjacent pivots.
            * 3 = Vertical Distance - Measures the vertical distance from the data point to the line segment
              between adjacent pivots.
        """
        self.n_pivots = n_pivots
        self.dist_measure = dist_measure


    def transform(self, data: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Transform the input data by applying the Perceptually Important Points method to reduce its dimensionality.

        This method processes either a single array or a list of arrays, applying the PIP method to each individually,
        and returns a transformed version of the input where each original array is reduced to its key pivots.

        Arguments:
        - data: Union[np.ndarray, List[np.ndarray]]
            The data to be transformed. Can be a single numpy array or a list of numpy arrays.

        Returns:
        - np.ndarray
            The reduced version of the original data, where each array is represented only by its identified pivots.
        """
        data = np.array(data)

        if data.ndim > 1:
            pips = []
            for _data in data:
                pips.append(self.transform(_data))
            return np.array(pips)

        n_pivots = self.n_pivots
        dist_measure = self.dist_measure

        pips_indices = [0, len(data) - 1]  # Start and end points as initial pivots
        pips_prices = [data[0], data[-1]]  # Values at the start and end points

        for curr_point in range(2, n_pivots):
            max_distance = 0.0  # Initialize the maximum distance found to 0
            max_distance_index = -1  # Index of the point with the maximum distance
            insert_index = -1  # Index to insert the new pivot

            for k in range(0, curr_point - 1):
                left_adj = k  # Left adjacent pivot index
                right_adj = k + 1  # Right adjacent pivot index

                time_diff = pips_indices[right_adj] - pips_indices[left_adj]
                price_diff = pips_prices[right_adj] - pips_prices[left_adj] + 1e-15  # Avoid division by zero
                slope = price_diff / time_diff
                intercept = pips_prices[left_adj] - slope * pips_indices[left_adj]

                for i in range(pips_indices[left_adj] + 1, pips_indices[right_adj]):
                    if dist_measure == 1:  # Euclidean distance
                        distance = np.sqrt((pips_indices[left_adj] - i)**2 + (pips_prices[left_adj] - data[i])**2)
                        distance += np.sqrt((pips_indices[right_adj] - i)**2 + (pips_prices[right_adj] - data[i])**2)
                    elif dist_measure == 2:  # Perpendicular distance
                        distance = abs(slope * i + intercept - data[i]) / np.sqrt(slope**2 + 1)
                    elif dist_measure == 3:  # Vertical distance
                        distance = abs(slope * i + intercept - data[i])

                    if distance > max_distance:
                        max_distance = distance
                        max_distance_index = i
                        insert_index = right_adj

            pips_indices.insert(insert_index, max_distance_index)
            pips_prices.insert(insert_index, data[max_distance_index])

        return np.array(pips_prices)


class ReducerFFT:
    def __init__(self, n_components: int) -> None:
        """
        Initialize the ReducerFFT with a specified number of frequency components to retain.
        
        This reducer implements the Fourier Transform method to extract significant frequency
        components from the data, which can capture underlying periodicities and patterns.

        Arguments:
        - n_components: int
            The number of dominant frequency components to identify and retain from the FFT of the dataset.
        """
        self.n_components = n_components

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the input data by applying the Fast Fourier Transform (FFT) method to extract
        significant frequency components.

        This method processes a single numpy array, applying the FFT, and returns a transformed version
        of the input where only the specified number of dominant frequency components are retained.

        Arguments:
        - data: np.ndarray
            The data to be transformed. Should be a one-dimensional numpy array.

        Returns:
        - np.ndarray
            The reduced version of the original data, represented by the amplitudes of its top n_components
            frequency components.
        """
        if data.ndim > 1:
            # Process each row of the data array individually
            fft_results = []
            for _data in data:
                fft_results.append(self.transform(_data))  # Recursive call for each row
            return np.array(fft_results)
        
        # Perform the FFT on the data
        fft_result = np.fft.fft(data)
        # Compute magnitudes of the FFT components
        magnitudes = np.abs(fft_result)

        # Identify indices of the top n_components largest magnitudes
        indices = np.argsort(magnitudes)[-self.n_components:]

        # Create a feature array of the selected FFT magnitudes
        # We sort the indices to maintain a consistent ordering
        top_magnitudes = magnitudes[np.sort(indices)]

        return top_magnitudes

        return np.array(top_magnitudes)


class ReducerWavelet:
    def __init__(self, n_coefficients: int, wavelet: str = 'coif1') -> None:
        """
        Initialize the ReducerWavelet with specified number of wavelet coefficients and the type of wavelet.
        
        This reducer applies a discrete wavelet transform to the data to extract important frequency and
        time features using wavelets.

        Arguments:
        - n_coefficients: int
            The number of largest (by magnitude) wavelet coefficients to retain from the wavelet transform.
        - wavelet: str
            The type of wavelet to use. Default is 'db1' (Daubechies wavelet with one vanishing moment).
            Other popular choices include 'db2', 'coif1', 'haar', etc.
        """
        self.n_coefficients = n_coefficients
        self.wavelet = wavelet

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the input data by applying the discrete wavelet transform and retaining a set number
        of the largest wavelet coefficients.

        This method processes a single numpy array, applies the wavelet transform, and returns a transformed
        version of the input where only the specified number of largest coefficients are retained.

        Arguments:
        - data: np.ndarray
            The data to be transformed. Should be a one-dimensional numpy array.

        Returns:
        - np.ndarray
            The reduced version of the original data, represented by its significant wavelet coefficients.
        """

        if data.ndim > 1:
            # Process each row of the data array individually
            wavelet_results = []
            for _data in data:
                wavelet_results.append(self.transform(_data))  # Recursive call for each row
            return np.array(wavelet_results)
        

        # Apply discrete wavelet transform
        coefficients = pywt.wavedec(data, wavelet=self.wavelet, mode='symmetric')
        # Flatten the list of coefficients
        all_coefficients = np.hstack(coefficients)
        # Find the indices of the largest coefficients by magnitude
        largest_indices = np.argsort(np.abs(all_coefficients))[-self.n_coefficients:]
        # Select the largest coefficients
        top_coefficients = all_coefficients[largest_indices]
        # Sort indices for consistent feature ordering
        sorted_top_coefficients = top_coefficients[np.argsort(largest_indices)]

        return sorted_top_coefficients


class ReducerFFTWavelet:
    def __init__(self, n_components: int, wavelet: str = 'db1') -> None:
        """
        Initialize the CombinedReducer with a total number of coefficients to be divided between Fourier and
        wavelet transforms. If n_components is odd, n_wavelet takes the larger share.

        Arguments:
        - n_components: int
            The total number of Fourier and wavelet transform components to retain.
        - wavelet: str
            The type of wavelet to use, e.g., 'db1', 'db2', 'coif1', 'haar'.
        """
        self.n_fourier = n_components // 2
        self.n_wavelet = n_components // 2 + (n_components % 2)  # n_wavelet gets the larger share if odd
        self.wavelet = wavelet

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply both Fourier and wavelet transforms to the data and combine the top coefficients from each
        to form a comprehensive feature vector. Supports processing multi-dimensional arrays where each row
        is treated as a separate dataset.

        Arguments:
        - data: np.ndarray
            The data to be transformed, can be one-dimensional or multi-dimensional.

        Returns:
        - np.ndarray
            A combined feature vector consisting of selected Fourier and wavelet transform coefficients.
        """
        if data.ndim > 1:
            # Process each row of the data array individually
            combined_results = []
            for _data in data:
                combined_results.append(self.transform(_data))  # Recursive call for each row
            return np.array(combined_results)

        # Fourier Transform
        fft_result = np.fft.fft(data)
        magnitudes = np.abs(fft_result)
        top_freq_indices = np.argsort(magnitudes)[-self.n_fourier:]
        top_frequencies = magnitudes[np.sort(top_freq_indices)]

        # Wavelet Transform
        coefficients = pywt.wavedec(data, wavelet=self.wavelet, mode='symmetric')
        all_coefficients = np.hstack(coefficients)
        largest_indices = np.argsort(np.abs(all_coefficients))[-self.n_wavelet:]
        top_wavelet_coeffs = all_coefficients[np.sort(largest_indices)]

        # Combine features
        combined_features = np.concatenate([top_frequencies, top_wavelet_coeffs])

        return combined_features
