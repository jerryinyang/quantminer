import pickle  # noqa
import warnings

from copy import copy
from pathlib import Path
from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qt
from kneed import KneeLocator
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score
from sktime.clustering.k_means import TimeSeriesKMeansTslearn

from .classes import SeqKMeans
from .classes.reducers import ReducerFFT, ReducerFFTWavelet, ReducerPIP, ReducerWavelet

# Ignore runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Miner:
    """
    A class to perform time series clustering and signal generation using various clustering techniques
    and dimensionality reduction methods.

    Args:
        n_lookback (int): Number of lookback periods.
        n_pivots (int): Number of pivot points.
        n_clusters (int, optional): Number of clusters. Default is 8.
        hold_period (int, optional): Holding period for signals. Default is 6.
        model_type (Literal["standard", "ts", "sequential"], optional): Type of clustering model. Default is 'standard'.
        reducer (Literal["FFT", "PIP", "Wavelet", "FFTWavelet"], optional): Type of dimensionality reduction. Default is 'PIP'.
        verbose (bool, optional): Verbosity mode. Default is False.
        wavelet (Literal["db1", "db2", "db3", "db4", "coif1", "haar", "sym5", "bior3.5"], optional): Type of wavelet to use. Default is 'coif2'.
        cluster_selection_mode (Literal["best", "baseline"], optional): Mode for cluster selection. Default is 'best'.

    Attributes:
        n_lookback (int): Number of lookback periods.
        n_pivots (int): Number of pivot points.
        n_clusters (int): Number of clusters.
        hold_period (int): Holding period for signals.
        _model_type (str): Type of clustering model.
        _cluster_selection_mode (str): Mode for cluster selection.
        _data (List): List to store training data.
        _price_data (List): List to store price data.
        _data_train_X (Union[List, np.ndarray]): Training input data.
        _data_train_y (Union[List, np.ndarray]): Training target data.
        _unique_pip_indices (Union[List, np.ndarray]): List of unique pivot indices.
        _cluster_labels (List): List of cluster labels.
        _cluster_labels_long (List): List of selected long cluster labels.
        _cluster_labels_short (List): List of selected short cluster labels.
        _agent_reduce (Union[ReducerPIP, ReducerFFT, ReducerWavelet, ReducerFFTWavelet]): Dimensionality reduction agent.
        _agent_cluster (Union[TimeSeriesKMeansTslearn, KMeans, SeqKMeans]): Clustering agent.
        _verbose (bool): Verbosity mode.
        _random_state (int): Random state for reproducibility.

    Methods:
        fit(data: np.ndarray, price_data: np.ndarray = None):
            Fit the Miner model to the given data.

            Args:
                data (np.ndarray): The input data to fit the model.
                price_data (np.ndarray, optional): The price data for returns computation. Default is None.

            Raises:
                ValueError: If there is a length mismatch between `price_data` and `data`.

        test(data: np.ndarray, price_data: np.ndarray = None, plot_equity=False) -> float:
            Test the Miner model on new data.

            Args:
                data (np.ndarray): The input data to test the model.
                price_data (np.ndarray, optional): The price data for returns computation. Default is None.
                plot_equity (bool, optional): Whether to plot the equity curve. Default is False.

            Returns:
                float: Martin score of the test.

            Raises:
                ValueError: If there is a length mismatch between `price_data` and `data`.

        transform(data: Union[List, np.ndarray]) -> np.ndarray:
            Generate labels for a full dataset.

            Args:
                data (Union[List, np.ndarray]): The input data to transform.

            Returns:
                np.ndarray: The generated labels.

        generate_signal(data: List[np.ndarray]) -> Tuple[int, List[float]]:
            Generate signal for one data window.

            Args:
                data (List[np.ndarray]): The input data window.

            Returns:
                Tuple[int, List[float]]: Containing the predicted signal and the list of pivot points.

        save_model(path: Union[Path, str]):
            Save the trained model to a file.

            Args:
                path (Union[Path, str]): The path to save the model.

            Raises:
                FileNotFoundError: If the specified path does not exist.

        load_model(path: Union[Path, str]) -> Miner:
            Load a trained model from a file.

            Args:
                path (Union[Path, str]): The path to load the model from.

            Returns:
                Miner: The loaded Miner model.

            Raises:
                FileNotFoundError: If the specified path does not exist.

        apply_holding_period(labels: np.ndarray, hold_period: Optional[int] = -1, selected_labels: Optional[List] = None) -> np.ndarray:
            Apply a holding period to selected cluster labels.

            Args:
                labels (np.ndarray): An array of cluster labels representing the labeling sequence.
                hold_period (Optional[int], optional): The minimum number of time steps required between consecutive occurrences of the same label. Defaults to -1.
                selected_labels (Optional[List], optional): A list of specific cluster labels to apply the holding period to. Defaults to None.

            Returns:
                np.ndarray: A new array of labels with the holding period applied.

        _preprocess_data(data: np.ndarray, **kwargs) -> np.ndarray:
            Perform series-level data transformations.

            Args:
                data (np.ndarray): The input data to preprocess.

            Returns:
                np.ndarray: The preprocessed data.

        _generate_training_set(data: np.ndarray = None):
            Generate the training input (X) and target (y) datasets.

            Args:
                data (np.ndarray, optional): The input data to generate the training set. Default is None.

            Raises:
                ValueError: If there is no data to generate the training set.

        _transform_data(data: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            Transform the generated training set data for clustering.

            Args:
                data (np.ndarray, optional): The input data to transform. Default is None.

            Returns:
                Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: The transformed data, and unique indices if in test mode.

            Raises:
                ValueError: If there is no training data to transform.
                ValueError: If the dimensionality reduction agent is not initialized.

        _generate_clusters():
            Cluster the training data.

        _assign_cluster_labels(data: np.ndarray = None, _labels: np.ndarray = None, indices: np.ndarray = None) -> np.ndarray:
            Assign cluster labels to each data point in the data.

            Args:
                data (np.ndarray, optional): The input data to assign labels. Default is None.
                _labels (np.ndarray, optional): The cluster labels. Default is None.
                indices (np.ndarray, optional): The unique indices. Default is None.

            Returns:
                np.ndarray: The assigned cluster labels.

            Raises:
                ValueError: If `_labels` or `indices` array is None.

        _assess_clusters():
            Assess each cluster's performance and select the best performing clusters.

        _compute_performance() -> float:
            Test the performance of the selected clusters.

            Returns:
                float: The computed Martin score for the selected clusters.

        _cleanup():
            Clear up memory used to store training data.

        __apply_holding_period(signals: np.ndarray) -> np.ndarray:
            Apply a holding period to a set of trading signals.

            Args:
                signals (np.ndarray): An array containing trading signals (e.g., buy/sell).

            Returns:
                np.ndarray: An array of signals with the holding period applied.

        __compute_martin(rets: np.array) -> float:
            Compute the Martin ratio for the given returns.

            Args:
                rets (np.array): The returns array.

            Returns:
                float: The computed Martin ratio.

        __find_n_clusters() -> int:
            Find the optimal number of clusters using the silhouette score.

            Returns:
                int: The optimal number of clusters.

        __normalizer_standard(points: np.array) -> np.array:
            Normalize the given points using standard normalization.

            Args:
                points (np.array): The input points to normalize.

            Returns:
                np.array: The normalized points.

        __visualize_clusters(data: np.ndarray, labels: np.ndarray):
            Visualize the clusters using Plotly.

            Args:
                data (np.ndarray): The input data.
                labels (np.ndarray): The cluster labels.

        __init_reducer(reducer: str, wavelet: str) -> Union[ReducerPIP, ReducerFFT, ReducerWavelet, ReducerFFTWavelet]:
            Initialize the dimensionality reduction agent.

            Args:
                reducer (str): The type of reducer to initialize.
                wavelet (str): The type of wavelet to use.

            Returns:
                Union[ReducerPIP, ReducerFFT, ReducerWavelet, ReducerFFTWavelet]: The initialized reducer.

        verbose_output(*args):
            Print verbose output if verbosity is enabled.

            Args:
                *args: The arguments to print.
    """

    def __init__(
        self,
        n_lookback: int,
        n_pivots: int,
        n_clusters: int = 8,
        hold_period: int = 6,
        model_type: Literal["standard", "ts", "sequential"] = "standard",
        reducer: Literal["FFT", "PIP", "Wavelet", "FFTWavelet"] = "PIP",
        verbose=False,
        wavelet: Literal[
            "db1", "db2", "db3", "db4", "coif1", "haar", "sym5", "bior3.5"
        ] = "coif2",
        cluster_selection_mode: Literal["best", "baseline"] = "best",
    ) -> None:
        """
        Initialize the Miner class with various parameters.

        Args:
            n_lookback (int): Number of lookback periods.
            n_pivots (int): Number of pivot points.
            n_clusters (int, optional): Number of clusters. Default is 8.
            hold_period (int, optional): Holding period for signals. Default is 6.
            model_type (str, optional): Type of clustering model. Default is 'standard'.
            reducer (str, optional): Type of dimensionality reduction. Default is 'PIP'.
            verbose (bool, optional): Verbosity mode. Default is False.
            wavelet (str, optional): Type of wavelet to use. Default is 'coif2'.
            cluster_selection_mode (str, optional): Mode for cluster selection. Default is 'best'.
        """
        self.n_lookback = n_lookback
        self.n_pivots = n_pivots
        self.hold_period = hold_period
        self.n_cluster = n_clusters

        self._model_type = model_type
        self._cluster_selection_mode = cluster_selection_mode

        # Store Training Data
        self._data: List = []
        self._price_data = []
        self._data_train_X: Union[List, np.ndarray] = []
        self._data_train_y: Union[List, np.ndarray] = []

        # Store Clusters Assignments
        self._unique_pip_indices: Union[List, np.ndarray] = []
        self._cluster_labels = []

        # Store Selected Cluster Labels
        self._cluster_labels_long = []
        self._cluster_labels_short = []

        # Store the dimensionality reduction agent
        self._agent_reduce: Union[
            ReducerPIP, ReducerFFT, ReducerWavelet, ReducerFFTWavelet
        ] = self.__init_reducer(reducer, wavelet=wavelet)

        # Store the clustering agent
        self._agent_cluster = None

        # Preset attributes
        self._verbose = verbose
        self._random_state = 0

    def fit(self, data: np.ndarray, price_data: np.ndarray = None):
        """
        Fit the Miner model to the given data.

        Args:
            data (np.ndarray): The input data to fit the model.
            price_data (np.ndarray, optional): The price data for returns computation. Default is None.
        """
        # Save raw price data for returns computation
        if price_data is None:
            price_data = copy(data)

        if len(price_data) != len(data):
            raise ValueError(
                f"Length mismatch: `price_data` is {len(price_data)}, `data` is {len(data)}"
            )

        # Assign self._price_data
        self._price_data = self._preprocess_data(price_data, test_mode=True)

        # Set the random state
        np.random.seed(self._random_state)

        # Preprocess the training data
        self._preprocess_data(data)

        # Create training dataset
        self._generate_training_set()

        # Transform the training data for clustering
        self._transform_data()

        # Cluster the data
        self._generate_clusters()

        # Assign Training Data to cluster
        self._assign_cluster_labels()

        # Assess Each Clusters Performance
        # Performance metric : Martin's Ratio
        # Evaluation metric : Hold Period
        self._assess_clusters()

        # Test the performance of the selected clusters
        martins = self._compute_performance()

        self.verbose_output("Training Complete : ", martins)

        # Clean up stored data
        self._cleanup()

        print(martins)

    def test(self, data: np.ndarray, price_data: np.ndarray = None, plot_equity=False):
        """
        Test the Miner model on new data.

        Args:
            data (np.ndarray): The input data to test the model.
            price_data (np.ndarray, optional): The price data for returns computation. Default is None.
            plot_equity (bool, optional): Whether to plot the equity curve. Default is False.

        Returns:
            float: Martin score of the test.
        """
        # Save raw price data for returns computation
        if price_data is None:
            price_data = copy(data)

        if len(price_data) != len(data):
            raise ValueError(
                f"Length mismatch: `price_data` is {len(price_data)}, `data` is {len(data)}"
            )

        # Preprocess Data
        data = self._preprocess_data(data, test_mode=True)
        price_data = self._preprocess_data(price_data, test_mode=True)

        _returns = np.diff(price_data, prepend=data[0])

        # Generate data windows
        windows = self._generate_training_set(data)

        # Transform the Data
        _pivots, _unique_indices = self._transform_data(windows)

        # Generate the cluster labels
        _l = self._agent_cluster.predict(_pivots)
        indices = _unique_indices + self.n_lookback - 1
        _labels = self._assign_cluster_labels(data, _l, indices)

        # Filter for only selected labels
        mask_long = np.isin(_labels, self._cluster_labels_long)
        mask_short = np.isin(_labels, self._cluster_labels_short)

        # Generate the signals
        _signals = np.zeros_like(_returns)
        _signals[mask_long] = 1
        _signals[mask_short] = -1

        # Implement holding period
        _signals = self.__apply_holding_period(_signals)

        # Get the returns, compute martin score
        _returns = _signals * _returns

        martin_score = self.__compute_martin(_returns)

        self.verbose_output("Test Martin Score : ", martin_score)

        if plot_equity:
            plt.plot(np.cumsum(_returns))
            plt.show()

        ser = pd.Series(_returns)

        self.verbose_output("Profit Factor : ", qt.stats.profit_factor(ser))
        self.verbose_output("Risk of Ruin : ", qt.stats.risk_of_ruin(ser))
        self.verbose_output("Sharpe : ", qt.stats.sharpe(ser))
        self.verbose_output("Avg Win : ", qt.stats.avg_win(ser))
        self.verbose_output("Avg Loss : ", qt.stats.avg_loss(ser))
        self.verbose_output("Avg Return : ", qt.stats.avg_return(ser))

        return martin_score

    def transform(self, data: Union[List, np.ndarray]):
        """
        Generate labels for a full dataset.

        Args:
            data (Union[List, np.ndarray]): The input data to transform.

        Returns:
            np.ndarray: The generated labels.
        """
        # Preprocess Data
        data = self._preprocess_data(data, test_mode=True)

        # Generate data windows
        windows = self._generate_training_set(data)

        # Transform the Data
        _pivots, _unique_indices = self._transform_data(windows)

        # Generate the cluster labels
        _l = self._agent_cluster.predict(_pivots)
        indices = _unique_indices + self.n_lookback - 1
        _labels = self._assign_cluster_labels(data, _l, indices)

        return _labels

    def generate_signal(self, data: List[np.ndarray]):
        """
        Generate signal for one data window.

        Args:
            data (List[np.ndarray]): The input data window.

        Returns:
            Tuple: Containing the predicted signal and the list of pivot points.
        """
        data = np.atleast_2d(self._preprocess_data(data, test_mode=True))

        # Generate entry signals from a data window
        _pivots = self._agent_reduce.transform(data)

        pivots = self.__normalizer_standard(_pivots)
        _l = self._agent_cluster.predict(pivots)

        signal = (
            1
            if _l in self._cluster_labels_long
            else -1
            if _l in self._cluster_labels_short
            else 0
        )

        return signal, list(np.squeeze(_pivots))

    def save_model(self, path: Union[Path, str]):
        """
        Save the trained model to a file.

        Args:
            path (Union[Path, str]): The path to save the model.
        """
        # Convert path to Path object, and check if it exists
        if not isinstance(path, Path):
            path = Path(path)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

        # Save model
        with open(path, "wb") as f:
            try:
                pickle.dump(self, f)
                self.verbose_output(f"Model saved at {path}")
            except Exception:
                raise

    @staticmethod
    def load_model(path: Union[Path, str]):
        """
        Load a trained model from a file.

        Args:
            path (Union[Path, str]): The path to load the model from.

        Returns:
            Miner: The loaded Miner model.
        """
        # Convert path to Path object, and check if it exists
        if not isinstance(path, Path):
            path = Path(str)

            if not path.exists():
                raise FileNotFoundError(
                    f"File not found. Model does not exist at : {path}"
                )

        # Load model
        with open(path, "rb") as f:
            try:
                miner = pickle.load(f)
                print(f"OUTPUT : Model Loaded from : {path}")
            except Exception:
                raise

        return miner

    def apply_holding_period(
        self,
        labels,
        hold_period: Optional[int] = -1,
        selected_labels: Optional[List] = None,
    ):
        """
        Applies a holding period to selected cluster labels, ensuring that consecutive occurrences of the same label are separated by the specified time.

        Args:
            labels (numpy.ndarray): An array of cluster labels representing the labeling sequence.
            hold_period (int, optional): The minimum number of time steps required between consecutive
                                        occurrences of the same label. If -1, uses the value stored in
                                        `self.hold_period`. Defaults to -1.
            selected_labels (List, optional):  A list of specific cluster labels to apply the holding
                                            period to. If None, applies to all cluster labels.
                                            Defaults to None.

        Returns:
            numpy.ndarray: A new array of labels with the holding period applied. Unmodified labels are
                        assigned a value of -1.
        """
        n_clusters = self.n_cluster

        # If no holding period is passed, use self.hold_period
        if hold_period <= 0:
            hold_period = self.hold_period

        # If no specific labels are passed, apply the holding period to all cluster labels
        if (selected_labels is None) or (len(selected_labels) == 0):
            selected_labels = list(range(n_clusters))

        labels = np.array(labels)

        valid_indices = np.where(np.isin(labels, selected_labels))[0]
        new_labels = np.ones_like(labels) * -1

        prev_index = -hold_period - 1
        for index in valid_indices:
            label = labels[index]

            # Ensure no overlapping labels
            if index > prev_index + hold_period:
                prev_index = index

                start_index = index + 1
                end_index = min(index + hold_period + 1, len(labels))

                new_labels[start_index:end_index] = label

        return new_labels

    def _preprocess_data(self, data, **kwargs):
        """
        Perform series-level data transformations. These can include detrending, denoising/filtering, domain transformations.
        The parameters of these transformations are saved, to be used for new data.

        Args:
            data (np.ndarray): The input data to preprocess.

        Returns:
            np.ndarray: The preprocessed data.
        """
        if kwargs.get("test_mode", False):
            return np.arcsinh(data)

        self._data = np.arcsinh(data)

    def _generate_training_set(self, data: np.ndarray = None):
        """
        Generate the training input (X) and target (y) datasets. When running a test, the `data` parameter is to be processed.

        Args:
            data (np.ndarray, optional): The input data to generate the training set. Default is None.
        """
        # Clear stores for training data
        self._data_train_X.clear()
        self._data_train_y.clear()

        # Initialize variables
        test_mode = data is not None  # if data is passed, test_mode is active
        lookback = self.n_lookback
        windows = []

        if not test_mode:
            data = self._data

        # Assert there is data
        if len(data) > 0:
            raise ValueError("No data to generate training set")

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Iterate through the data and append the data windows
        for index in range(lookback, len(data)):
            try:
                start_index = index - lookback
                end_index = index

                windows.append(data[start_index:end_index])

            except Exception as e:
                self.verbose_output("Error Occurred : \n", e)
                continue

        # During tests, return the generated data windows
        if test_mode:
            return np.array(windows)

        # For training, assign windows to training data
        self._data_train_X = np.array(windows)

    def _transform_data(self, data: np.ndarray = None):
        """
        Transform the generated training set data for clustering. If data is passed, transform the data passed only.
        Transformations include scaling/normalization, dimensionality reduction, etc.

        Args:
            data (np.ndarray, optional): The input data to transform. Default is None.

        Returns:
            Tuple: Numpy array containing clustering-ready data, and unique indices if in test mode.
        """
        test_mode = data is not None

        if not test_mode:
            data = self._data_train_X

        # Ensure the training set has been generated
        if len(data) < 0:
            raise ValueError("No training data to transform")

        # Dimensionality Reduction
        if self._agent_reduce is None:
            raise ValueError(
                "Model has not been trained. self._agent_reduce has not been initialized."
            )

        pivots = self._agent_reduce.transform(data)

        # Only keep unique patterns
        null_pivots = [-1] * self.n_pivots

        # Compare each group with the previous one, excluding the first and last items
        is_duplicate = np.all(pivots[1:, 1:-1] == pivots[:-1, 1:-1], axis=1)

        # Replace duplicates with null_pivots
        pivots[1:][is_duplicate] = null_pivots

        # Remove null_pivots
        mask_unique = np.all(pivots != null_pivots, axis=1)
        pivots = pivots[mask_unique]

        # Data Scaling / Normalization ; Element-wise for each individual window
        data = np.apply_along_axis(self.__normalizer_standard, axis=1, arr=pivots)

        # For test mode
        if test_mode:
            return data, np.where(mask_unique)[0]

        # Get the indices where value is True
        self._data_train_X = data
        self._unique_pip_indices = np.where(mask_unique)[0]

    def _generate_clusters(self):
        """
        Cluster the training data. Default n_clusters
        """
        # SKTime (TSLearn) Kmeans
        if self._model_type == "ts":
            self._agent_cluster = TimeSeriesKMeansTslearn(
                n_clusters=self.n_cluster,
                metric="euclidean",
                n_jobs=-1,
                random_state=self._random_state,
            )

        elif self._model_type == "standard":
            self._agent_cluster = KMeans(
                n_clusters=self.n_cluster,
                n_init="auto",
                random_state=self._random_state,
            )

        elif self._model_type == "sequential":
            self._agent_cluster = SeqKMeans(
                n_clusters=self.n_cluster,
                learning_rate=0.5,
                centroid_update_threshold_std=3,
                verbose=self._verbose,
                fit_method="sequential",
                random_state=self._random_state,
            )

        self.verbose_output("Clustering data...")

        self._agent_cluster.fit(self._data_train_X)

        self.verbose_output("Clustering complete")

    def _assign_cluster_labels(
        self,
        data: np.ndarray = None,
        _labels: np.ndarray = None,
        indices: np.ndarray = None,
    ):
        """
        Assign clusters labels to each data point in the data.

        Args:
            data (np.ndarray, optional): The input data to assign labels. Default is None.
            _labels (np.ndarray, optional): The cluster labels. Default is None.
            indices (np.ndarray, optional): The unique indices. Default is None.
        """
        self._cluster_labels.clear()
        test_mode = data is not None

        if data is None:
            # Get raw data
            data = self._data

            # Get the labels from agent_cluster
            _labels = self._agent_cluster.labels_

            # Get the unique pips indices
            indices = self._unique_pip_indices + self.n_lookback - 1

        if _labels is None:
            raise ValueError("`_labels` array cannot be None.")
        if indices is None:
            raise ValueError("`indices` array cannot be None.")

        # Assign placeholder signals
        labels = np.ones(len(data)) * -1
        labels[indices] = _labels

        # For tests, return the labels
        if test_mode:
            return np.array(labels)

        self._cluster_labels = np.array(labels)

    def _assess_clusters(self):
        """
        Assess each cluster's performance, and selected the best performing clusters
        """
        self._cluster_labels_long.clear()
        self._cluster_labels_short.clear()

        # Store the cluster scores from each data
        cluster_scores = []
        baseline_long = []
        baseline_short = []

        # Compute the returns
        _returns = np.diff(self._price_data, prepend=self._price_data[0])

        # Baseline Metrics
        baseline_returns = pd.Series(_returns)
        baseline_profit_factor = max(qt.stats.profit_factor(baseline_returns), 0)
        baseline_sharpe_ratio = max(qt.stats.sharpe(baseline_returns), 0)
        baseline_upi = max(qt.stats.ulcer_performance_index(baseline_returns), 1)
        baseline_max_dd = qt.stats.max_drawdown(baseline_returns)

        # Get the cluster labels
        _labels = self._cluster_labels

        # Iterate through each cluster label
        for _label in range(self.n_cluster):
            # Create a mask for the label in the labels; everything else should be zero
            mask_label: np.ndarray = _labels == _label

            # Filter the labels
            _signals = mask_label.astype(int)

            # Implement Holding Period
            _signals = self.__apply_holding_period(_signals)

            # Get the returns, compute martin score
            _returns = _signals * _returns
            _martin = self.__compute_martin(_returns)

            if (_martin is np.inf) or np.isnan(_martin):
                cluster_scores.append(0)
            else:
                cluster_scores.append(_martin)

            if self._cluster_selection_mode == "baseline":
                for direction in [1, -1]:
                    # Compute the returns
                    _ret = pd.Series(_returns * direction)

                    # Compute the kpis
                    _pf = qt.stats.profit_factor(_ret)
                    _sharpe = qt.stats.rolling_sharpe(_ret).mean()
                    _upi = qt.stats.ulcer_performance_index(_ret)
                    _max_dd = qt.stats.to_drawdown_series(_ret).mean()

                    if (_upi is np.inf) or np.isnan(_upi):
                        continue

                    if (
                        (_pf > baseline_profit_factor)
                        and (_sharpe > baseline_sharpe_ratio)
                        and (_upi > baseline_upi)
                        and (_max_dd > baseline_max_dd)
                    ):
                        if direction > 0:
                            baseline_long.append(_label)
                        else:
                            baseline_short.append(_label)

        # Append the selected cluster labels
        if self._cluster_selection_mode == "baseline":
            self._cluster_labels_long = baseline_long
            self._cluster_labels_short = baseline_short

        else:
            self._cluster_labels_long.append(np.argmax(cluster_scores))
            self._cluster_labels_short.append(np.argmin(cluster_scores))

    def _compute_performance(self):
        """
        Test the performance of the selected clusters

        Returns:
            float: The computed Martin score for the selected clusters.
        """
        _returns = np.diff(self._price_data, prepend=self._price_data[0])

        # Get the full labels
        _labels = self._cluster_labels

        # Filter for only selected labels
        mask_long = np.isin(_labels, self._cluster_labels_long)
        mask_short = np.isin(_labels, self._cluster_labels_short)

        # Generate the signals
        _signals = np.zeros_like(_returns)
        _signals[mask_long] = 1
        _signals[mask_short] = -1

        # Implement the holding period
        _signals = self.__apply_holding_period(_signals)

        # Get the returns, compute martin score
        _returns = _signals * _returns

        return self.__compute_martin(_returns)

    def _cleanup(self):
        """
        Clear up memory used to store training data
        """
        # Clear Training Data
        self._data = []
        self._price_data = []
        self._data_train_X = []
        self._data_train_y = []

        # Clear Clusters Assignments
        self._unique_pip_indices = []
        self._cluster_labels = []

    def __apply_holding_period(self, signals):
        """
        Applies a holding period to a set of trading signals, enforcing a delay between
        consecutive non-zero signals.

        This function is likely used in trading strategies to prevent overly frequent
        trades, potentially reducing transaction costs and mitigating the effects of
        temporary market noise.

        Args:
            signals (numpy.ndarray): An array containing trading signals (e.g., buy/sell).

        Returns:
            numpy.ndarray: An array of signals with the holding period applied.
                        Non-zero signals will be followed by a specified number of
                        zero signals.
        """
        signals = np.array(signals)

        nonzero = np.where(signals != 0)[0]
        new_signals = np.zeros_like(signals)

        prev_index = -self.hold_period - 1
        for index in nonzero:
            signal = signals[index]

            if index > prev_index + self.hold_period:
                prev_index = index

                start_index = index + 1
                end_index = min(index + self.hold_period + 1, len(signals))

                new_signals[start_index:end_index] = signal

        return new_signals

    def __compute_martin(self, rets: np.array):
        """
        Compute the Martin ratio for the given returns.

        Args:
            rets (np.array): The returns array.

        Returns:
            float: The computed Martin ratio.
        """
        return qt.stats.ulcer_performance_index(pd.Series(rets))

    def __find_n_clusters(self):
        """
        Find the optimal number of clusters using the silhouette score.

        Returns:
            int: The optimal number of clusters.
        """
        n_clusters = range(2, 50)  # change this range according to your needs
        silhouette_scores = []
        X = self._data_train_X

        # Get the silhouette scores
        for n in n_clusters:
            birch = Birch(n_clusters=n)
            kmeans = KMeans(n_clusters=n, n_init="auto")

            birch.fit(X)
            kmeans.fit(X)

            silhouette_scores.append(
                np.mean(
                    [
                        silhouette_score(X, birch.labels_),
                        silhouette_score(X, kmeans.labels_),
                    ]
                )
            )

        kneedle = KneeLocator(
            range(2, 50),
            np.array(silhouette_scores),
            S=1.0,
            curve="convex",
            direction="decreasing",
        )
        optimal_n = kneedle.knee

        self.verbose_output("Optimal N: ", optimal_n)

        return optimal_n

    def __normalizer_standard(self, points):
        """
        Normalize the given points using standard normalization.

        Args:
            points (np.array): The input points to normalize.

        Returns:
            np.array: The normalized points.
        """
        points = np.array(points)

        points = (points - np.mean(points)) / (np.std(points) + 1e-10)
        return np.array(points)

    def __visualize_clusters(self, data, labels):
        """
        Visualize the clusters using Plotly.

        Args:
            data (np.ndarray): The input data.
            labels (np.ndarray): The cluster labels.
        """
        import plotly.graph_objects as go

        # Create a subplot for the time series data
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data, mode="lines", name="Data"))

        # Create a subplot for the labels
        fig.add_trace(go.Scatter(y=labels, mode="lines", name="Labels"))

        # Enable spike lines
        fig.update_layout(
            hovermode="x",
            spikedistance=-1,
            xaxis=dict(
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikedash="solid",
            ),
            yaxis=dict(
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikedash="solid",
            ),
        )

        # Show the plot
        fig.show()

    def __init_reducer(self, reducer: str, wavelet: str):
        """
        Initialize the dimensionality reduction agent.

        Args:
            reducer (str): The type of reducer to initialize.
            wavelet (str): The type of wavelet to use.

        Returns:
            Union[ReducerPIP, ReducerFFT, ReducerWavelet, ReducerFFTWavelet]: The initialized reducer.
        """
        if reducer == "FFT":
            return ReducerFFT(n_components=self.n_pivots)

        elif reducer == "Wavelet":
            return ReducerWavelet(n_coefficients=self.n_pivots, wavelet=wavelet)

        elif reducer == "FFTWavelet":
            return ReducerFFTWavelet(n_components=self.n_pivots)

        else:
            return ReducerPIP(n_pivots=self.n_pivots, dist_measure=self.dist_measure)

    def verbose_output(self, *args):
        """
        Print verbose output if verbosity is enabled.

        Args:
            *args: The arguments to print.
        """
        if not self._verbose:
            return

        for _ in args:
            print(_)


if __name__ == "__main__":
    parent_path = Path(__file__).parent
    btc_path = parent_path / "data/BTCUSDT_FULL.parquet"

    # Define your date range
    start_date = "2017-12-01"
    end_date = "2021-12-31"

    raw_data = pd.read_parquet(btc_path)

    # Filter the DataFrame
    train_data = raw_data[(raw_data.index >= start_date) & (raw_data.index <= end_date)]
    test_data = raw_data[
        (raw_data.index >= "2022-01-01") & (raw_data.index <= "2023-12-31")
    ]

    train_data = train_data["close"].dropna(axis=0)
    train_data = train_data.to_numpy()

    test_data = test_data["close"].dropna(axis=0)
    test_data = test_data.to_numpy()

    miner = Miner(25, 4, 16, 3, cluster_selection_mode="")

    miner.fit(train_data)
    print(miner._cluster_labels_long)
    print(miner._cluster_labels_short)
