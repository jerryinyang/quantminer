import numpy as np
import pywt


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
        # Perform the FFT on the data
        fft_result = np.fft.fft(data)
        # Compute magnitudes of the FFT components
        magnitudes = np.abs(fft_result)

        # Identify indices of the top n_components largest magnitudes
        indices = np.argsort(magnitudes)[-self.n_components:]

        # Create a feature array of the selected FFT magnitudes
        # We sort the indices to maintain a consistent ordering
        top_magnitudes = magnitudes[sorted(indices)]

        return top_magnitudes


class ReducerWavelet:
    def __init__(self, n_coefficients: int, wavelet: str = 'db1') -> None:
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
    def __init__(self, n_components:int, wavelet: str = 'db1') -> None:
        """
        Initialize the CombinedReducer with the number of Fourier and wavelet coefficients to retain,
        and specify the type of wavelet.

        Arguments:
        - n_fourier: int
            The number of dominant Fourier transform components to retain.
        - n_wavelet: int
            The number of largest wavelet coefficients to retain.
        - wavelet: str
            The type of wavelet to use, e.g., 'db1', 'db2', 'coif1', 'haar'.
        """
        self.n_fourier = n_components 
        self.n_wavelet = n_components
        self.wavelet = wavelet

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply both Fourier and wavelet transforms to the data and combine the top coefficients from each
        to form a comprehensive feature vector.

        Arguments:
        - data: np.ndarray
            The data to be transformed, expected to be a one-dimensional time series.

        Returns:
        - np.ndarray
            A combined feature vector consisting of selected Fourier and wavelet transform coefficients.
        """
        # Fourier Transform
        fft_result = np.fft.fft(data)
        magnitudes = np.abs(fft_result)
        top_freq_indices = np.argsort(magnitudes)[-self.n_fourier:]
        top_frequencies = magnitudes[top_freq_indices]

        # Wavelet Transform
        coefficients = pywt.wavedec(data, wavelet=self.wavelet, mode='symmetric')
        all_coefficients = np.hstack(coefficients)
        largest_indices = np.argsort(np.abs(all_coefficients))[-self.n_wavelet:]
        top_wavelet_coeffs = all_coefficients[largest_indices]

        # Combine features
        combined_features = np.concatenate([top_frequencies, top_wavelet_coeffs])

        return combined_features



data = np.random.rand(100, 20)

reducer = ReducerFFTWavelet(5)

print(reducer.transform(data[0]))