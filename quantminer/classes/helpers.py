import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import quantstats as qt
import seaborn as sns
from dtaidistance import dtw
from kneed import KneeLocator
from plotly.subplots import make_subplots
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import TimeSeriesSplit
from sktime.clustering.k_means import TimeSeriesKMeansTslearn

from . import (
    BaseReducer,
    DistanceMeasure,
    ReducerFFT,
    ReducerFFTWavelet,
    ReducerPIP,
    ReducerWavelet,
    SeqKMeans,
)


class ModelTester:
    """
    Handles model testing, cross-validation, and performance evaluation.

    This class centralizes all testing and evaluation functionality, providing methods
    for model validation, performance assessment, and reporting. It works with the
    core components to evaluate the model's effectiveness.

    Attributes:
        verbose (bool): Controls logging verbosity
        hold_period (int): Number of periods to hold positions
        preprocessor: DataPreprocessor instance
        mining_core: MiningCore instance
        cluster_evaluator: ClusterEvaluator instance
    """

    def __init__(
        self,
        preprocessor,  # DataPreprocessor instance
        mining_core,  # MiningCore instance
        cluster_evaluator,  # ClusterEvaluator instance
        hold_period: int = 6,
        verbose: bool = False,
    ):
        """
        Initialize the ModelTester.

        Args:
            preprocessor: DataPreprocessor instance for data preparation
            mining_core: MiningCore instance for clustering and signal generation
            cluster_evaluator: ClusterEvaluator instance for performance evaluation
            hold_period (int): Number of periods to hold positions
            verbose (bool): Enable detailed logging output
        """
        self.preprocessor = preprocessor
        self.mining_core = mining_core
        self.cluster_evaluator = cluster_evaluator
        self.hold_period = hold_period
        self.verbose = verbose

        # Setup logging
        if self.verbose:
            logging.basicConfig(level=logging.INFO)

    def test_model(
        self,
        data: np.ndarray,
        price_data: Optional[np.ndarray] = None,
        returns_series: Optional[pd.Series] = None,  # New parameter
        plot_equity: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Test the model on new data and compute performance metrics.

        Args:
            data (np.ndarray): Input data for testing
            price_data (np.ndarray, optional): Price data for returns computation
            returns_series (pd.Series, optional): Pre-calculated returns with datetime index
            plot_equity (bool): Whether to plot the equity curve
        """
        if price_data is None:
            price_data = np.copy(data)

        if len(price_data) != len(data):
            raise ValueError(
                f"Length mismatch: price_data is {len(price_data)}, data is {len(data)}"
            )

        # Preprocess data
        processed_data = self.preprocessor.preprocess_data(data, test_mode=True)
        processed_price = self.preprocessor.preprocess_data(price_data, test_mode=True)

        # Use provided returns_series if available, otherwise calculate returns
        if returns_series is not None:
            returns = returns_series.values
            returns_index = returns_series.index
        else:
            returns = np.diff(processed_price, prepend=processed_price[0])
            returns_index = None

        # Generate windows and transform data
        windows = self.preprocessor.generate_training_set(
            processed_data, self.mining_core.n_lookback, test_mode=True
        )
        pivots, unique_indices = self.preprocessor.transform_data(
            windows, self.mining_core.n_pivots, self.mining_core.reducer, test_mode=True
        )

        # Generate cluster labels and signals
        cluster_labels = self.mining_core.predict(pivots)
        indices = unique_indices + self.mining_core.n_lookback - 1
        labels = self._assign_labels(processed_data, cluster_labels, indices)
        signals = self._generate_signals(labels, returns)

        # Calculate performance metrics
        strategy_returns = signals * returns
        if returns_index is not None:
            strategy_returns = pd.Series(strategy_returns, index=returns_index)

        metrics = self._calculate_metrics(strategy_returns)

        if plot_equity:
            self._plot_equity_curve(strategy_returns)

        return metrics["martin_ratio"], metrics

    def cross_validate(
        self,
        data: np.ndarray,
        price_data: Optional[np.ndarray] = None,
        n_splits: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.

        Args:
            data (np.ndarray): Input data
            price_data (np.ndarray, optional): Price data
            n_splits (int): Number of cross-validation splits

        Returns:
            Dict[str, List[float]]: Dictionary of performance metrics for each fold
        """
        if price_data is None:
            price_data = np.copy(data)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_metrics = {
            "martin_ratio": [],
            "sharpe_ratio": [],
            "profit_factor": [],
            "avg_return": [],
            "max_drawdown": [],
        }

        for train_idx, test_idx in tscv.split(data):
            # Split data
            train_data = data[train_idx]
            test_data = data[test_idx]
            test_price = price_data[test_idx]

            # Train model
            self._train_model(train_data)

            # Test model
            martin_ratio, metrics = self.test_model(test_data, test_price)

            # Store metrics
            for key in cv_metrics:
                cv_metrics[key].append(metrics[key])

        if self.verbose:
            self._log_cv_results(cv_metrics)

        return cv_metrics

    def generate_performance_report(
        self, returns: np.ndarray, output_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Generate comprehensive performance report.

        Args:
            returns (np.ndarray): Strategy returns
            output_path (Path, optional): Path to save report

        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        series_returns = pd.Series(returns)

        metrics = {
            "total_return": float(np.sum(returns)),
            "annual_return": float(np.mean(returns) * 252),
            "volatility": float(np.std(returns) * np.sqrt(252)),
            "sharpe_ratio": float(qt.stats.sharpe(series_returns)),
            "sortino_ratio": float(qt.stats.sortino(series_returns)),
            "martin_ratio": float(qt.stats.ulcer_performance_index(series_returns)),
            "profit_factor": float(qt.stats.profit_factor(series_returns)),
            "avg_return": float(qt.stats.avg_return(series_returns)),
            "avg_win": float(qt.stats.avg_win(series_returns)),
            "avg_loss": float(qt.stats.avg_loss(series_returns)),
            "win_rate": float(qt.stats.win_rate(series_returns)),
            "risk_of_ruin": float(qt.stats.risk_of_ruin(series_returns)),
            "max_drawdown": float(qt.stats.max_drawdown(series_returns)),
            "avg_drawdown": float(qt.stats.to_drawdown_series(series_returns).mean()),
        }

        if output_path is not None:
            self._save_report(metrics, output_path)

        return metrics

    def _train_model(self, train_data: np.ndarray):
        """
        Train model components on data.

        Args:
            train_data (np.ndarray): Training data
        """
        processed_data = self.preprocessor.preprocess_data(train_data)
        windows = self.preprocessor.generate_training_set(
            processed_data, self.mining_core.n_lookback
        )
        transformed_data = self.preprocessor.transform_data(
            windows, self.mining_core.n_pivots, self.mining_core.reducer
        )
        self.mining_core.fit(transformed_data)

    def _assign_labels(
        self, data: np.ndarray, labels: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """
        Assign cluster labels to full dataset.

        Args:
            data (np.ndarray): Full dataset
            labels (np.ndarray): Cluster labels
            indices (np.ndarray): Indices for label assignment

        Returns:
            np.ndarray: Array of assigned labels
        """
        full_labels = np.ones(len(data)) * -1
        full_labels[indices] = labels
        return full_labels

    def _generate_signals(self, labels: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Generate trading signals from labels.

        Args:
            labels (np.ndarray): Cluster labels
            returns (np.ndarray): Returns data

        Returns:
            np.ndarray: Trading signals
        """
        # Create signal masks
        mask_long = np.isin(labels, self.cluster_evaluator.cluster_labels_long)
        mask_short = np.isin(labels, self.cluster_evaluator.cluster_labels_short)

        # Generate signals
        signals = np.zeros_like(returns)
        signals[mask_long] = 1
        signals[mask_short] = -1

        # Apply holding period
        return self.cluster_evaluator._apply_holding_period(signals)

    def _calculate_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate key performance metrics.

        Args:
            returns (np.ndarray): Strategy returns

        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        series_returns = pd.Series(returns)

        return {
            "martin_ratio": float(qt.stats.ulcer_performance_index(series_returns)),
            "sharpe_ratio": float(qt.stats.sharpe(series_returns)),
            "profit_factor": float(qt.stats.profit_factor(series_returns)),
            "avg_return": float(qt.stats.avg_return(series_returns)),
            "max_drawdown": float(qt.stats.max_drawdown(series_returns)),
        }

    def _plot_equity_curve(self, returns: np.ndarray):
        """
        Plot cumulative returns curve.

        Args:
            returns (np.ndarray): Strategy returns
        """
        plt.figure(figsize=(12, 6))
        plt.plot(np.cumsum(returns))
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Returns")
        plt.grid(True)
        plt.show()

    def _log_cv_results(self, metrics: Dict[str, List[float]]):
        """
        Log cross-validation results.

        Args:
            metrics (Dict[str, List[float]]): Cross-validation metrics
        """
        for metric, values in metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            logging.info(f"{metric}: mean = {mean_value:.4f}, std = {std_value:.4f}")

    def _save_report(self, metrics: Dict[str, float], path: Path):
        """
        Save performance report to file.

        Args:
            metrics (Dict[str, float]): Performance metrics
            path (Path): Output file path
        """
        with open(path, "w") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")


class MiningCore:
    """
    Handles core clustering and signal generation operations.

    This class centralizes the core mining operations including clustering, signal generation,
    and model management. It works in conjunction with DataPreprocessor and ClusterEvaluator.

    Attributes:
        n_clusters (int): Number of clusters
        model_type (str): Type of clustering model ('standard', 'ts', or 'sequential')
        random_state (int): Random state for reproducibility
        verbose (bool): Controls logging verbosity
        reducer_type (str): Type of dimensionality reducer
        wavelet (str): Type of wavelet for wavelet-based reduction
        cluster_model: The clustering model instance
        reducer: The dimensionality reducer instance
    """

    def __init__(
        self,
        n_clusters: int = 8,
        n_pivots: int = 4,
        n_lookback: int = 25,
        model_type: Literal["standard", "ts", "sequential"] = "standard",
        reducer_type: Literal["FFT", "PIP", "Wavelet", "FFTWavelet"] = "PIP",
        wavelet: str = "coif1",
        random_state: int = 14,
        verbose: bool = False,
    ):
        """
        Initialize the MiningCore.

        Args:
            n_clusters (int): Number of clusters to create
            model_type (str): Type of clustering model
            reducer_type (str): Type of dimensionality reducer
            wavelet (str): Type of wavelet for wavelet-based reduction
            random_state (int): Random state for reproducibility
            verbose (bool): Enable detailed logging output
        """
        self.n_clusters = n_clusters
        self.n_pivots = n_pivots
        self.n_lookback = n_lookback
        self.model_type = model_type
        self.random_state = random_state
        self.verbose = verbose

        # Initialize components
        self.cluster_model = None
        self.reducer = self._init_reducer(reducer_type, wavelet)

        # Setup logging
        if self.verbose:
            logging.basicConfig(level=logging.INFO)

    def fit(self, data: np.ndarray):
        """
        Fit the clustering model to the provided data.

        Args:
            data (np.ndarray): Training data for clustering

        Raises:
            ValueError: If the data is invalid or empty
        """
        if data is None or len(data) == 0:
            raise ValueError("Invalid or empty training data")

        # Initialize clustering model if not already done
        if self.cluster_model is None:
            self._init_cluster_model()

        if self.verbose:
            logging.info("Fitting clustering model...")

        # Fit the model
        self.cluster_model.fit(data)

        if self.verbose:
            logging.info("Model fitting completed")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Generate cluster predictions for new data.

        Args:
            data (np.ndarray): Data to predict clusters for

        Returns:
            np.ndarray: Predicted cluster labels

        Raises:
            ValueError: If model hasn't been fitted
        """
        if self.cluster_model is None:
            raise ValueError("Model must be fitted before prediction")

        return self.cluster_model.predict(data)

    def generate_signal(
        self,
        data: np.ndarray,
        cluster_labels_long: List[int],
        cluster_labels_short: List[int],
    ) -> Tuple[int, float, List[float]]:
        """
        Generate trading signals with confidence scores.

        Args:
            data (np.ndarray): Input data for signal generation
            cluster_labels_long (List[int]): Cluster labels for long positions
            cluster_labels_short (List[int]): Cluster labels for short positions

        Returns:
            Tuple[int, float, List[float]]:
                - Trading signal (-1, 0, 1)
                - Confidence score
                - List of pivot points
        """
        try:
            # Transform data to pivot points
            pivots = self.reducer.transform(np.atleast_2d(data))

            # Normalize pivots
            pivots_norm = self._normalizer_standard(pivots)

            # Calculate distances to centroids
            distances = self._calculate_cluster_distances(pivots_norm)

            # Convert distances to probabilities
            probs = self._calculate_signal_probabilities(distances)

            # Generate signal and confidence
            signal, confidence = self._get_signal_with_confidence(
                probs, cluster_labels_long, cluster_labels_short
            )

            if self.verbose:
                logging.info(
                    f"Generated signal: {signal} with confidence: {confidence:.2f}"
                )

        except Exception as e:
            logging.error(f"Error generating signal: {e!s}")
            return 0, 0.0, list(np.squeeze(pivots))

        return signal, confidence, list(np.squeeze(pivots))

    def _init_cluster_model(self):
        """
        Initialize the appropriate clustering model based on model_type.
        """
        if self.model_type == "ts":
            self.cluster_model = TimeSeriesKMeansTslearn(
                n_clusters=self.n_clusters,
                metric="euclidean",
                n_jobs=-1,
                random_state=self.random_state,
            )
        elif self.model_type == "standard":
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                n_init="auto",
                random_state=self.random_state,
            )
        else:  # sequential
            self.cluster_model = SeqKMeans(
                n_clusters=self.n_clusters,
                learning_rate=0.5,
                centroid_update_threshold_std=3,
                verbose=self.verbose,
                fit_method="sequential",
                random_state=self.random_state,
            )

        if self.verbose:
            logging.info(f"Initialized {self.model_type} clustering model")

    def _init_reducer(
        self, reducer_type: str, wavelet: str
    ) -> Union[ReducerPIP, ReducerFFT, ReducerWavelet, ReducerFFTWavelet]:
        """
        Initialize the dimensionality reduction component.

        Args:
            reducer_type (str): Type of reducer to initialize
            wavelet (str): Type of wavelet to use

        Returns:
            Union[ReducerPIP, ReducerFFT, ReducerWavelet, ReducerFFTWavelet]:
                Initialized reducer
        """
        if reducer_type == "FFT":
            return ReducerFFT(n_components=self.n_pivots)
        elif reducer_type == "Wavelet":
            return ReducerWavelet(n_coefficients=self.n_pivots, wavelet=wavelet)
        elif reducer_type == "FFTWavelet":
            return ReducerFFTWavelet(n_components=self.n_pivots, wavelet=wavelet)
        else:
            return ReducerPIP(
                n_pivots=self.n_pivots, dist_measure=DistanceMeasure.EUCLIDEAN
            )

    def _calculate_cluster_distances(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate distances between a point and all cluster centroids.

        Args:
            point (np.ndarray): Input data point

        Returns:
            np.ndarray: Distances to each centroid
        """
        distances = np.zeros(self.n_clusters)

        for i in range(self.n_clusters):
            if self.model_type == "ts":
                distances[i] = dtw.distance_fast(
                    point, self.cluster_model.cluster_centers_[i]
                )
            else:
                distances[i] = np.linalg.norm(
                    point - self.cluster_model.cluster_centers_[i]
                )

        return distances

    def _calculate_signal_probabilities(self, distances: np.ndarray) -> np.ndarray:
        """
        Convert distances to probabilities using softmax.

        Args:
            distances (np.ndarray): Array of distances to centroids

        Returns:
            np.ndarray: Array of probabilities for each cluster
        """
        # Convert distances to similarities
        similarities = -distances

        # Apply softmax
        exp_similarities = np.exp(similarities - np.max(similarities))
        return exp_similarities / exp_similarities.sum()

    def _get_signal_with_confidence(
        self,
        probabilities: np.ndarray,
        cluster_labels_long: List[int],
        cluster_labels_short: List[int],
    ) -> Tuple[int, float]:
        """
        Generate trading signal and confidence score from cluster probabilities.

        Args:
            probabilities (np.ndarray): Array of cluster probabilities
            cluster_labels_long (List[int]): Clusters for long positions
            cluster_labels_short (List[int]): Clusters for short positions

        Returns:
            Tuple[int, float]:
                - Trading signal (-1, 0, 1)
                - Confidence score
        """
        # Calculate directional probabilities
        long_prob = sum(probabilities[i] for i in cluster_labels_long)
        short_prob = sum(probabilities[i] for i in cluster_labels_short)

        # Signal threshold
        SIGNAL_THRESHOLD = 0.6

        # Determine signal
        if long_prob > SIGNAL_THRESHOLD and long_prob > short_prob:
            return 1, long_prob
        elif short_prob > SIGNAL_THRESHOLD and short_prob > long_prob:
            return -1, short_prob
        else:
            return 0, max(1 - (long_prob + short_prob), 0)

    def _normalizer_standard(self, points: np.ndarray) -> np.ndarray:
        """
        Apply standard normalization to points.

        Args:
            points (np.ndarray): Points to normalize

        Returns:
            np.ndarray: Normalized points
        """
        points = np.array(points)
        return (points - np.mean(points)) / (np.std(points) + 1e-10)

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Get cluster centers."""
        return self.cluster_model.cluster_centers_

    @property
    def labels_(self) -> np.ndarray:
        """Get cluster labels."""
        return self.cluster_model.labels_


class ClusterEvaluator:
    """
    Handles cluster performance assessment and model evaluation.

    This class centralizes all cluster evaluation operations, including performance metrics
    calculation, cluster assessment, and optimal cluster number determination.

    Attributes:
        selection_mode (str): Mode for selecting clusters ('best' or 'baseline')
        verbose (bool): Controls logging verbosity
        hold_period (int): Number of periods to hold positions
        cluster_labels_long (List[int]): Selected cluster labels for long positions
        cluster_labels_short (List[int]): Selected cluster labels for short positions
    """

    def __init__(
        self, selection_mode: str = "best", hold_period: int = 6, verbose: bool = False
    ):
        """
        Initialize the ClusterEvaluator.

        Args:
            selection_mode (str): Strategy for selecting clusters ('best' or 'baseline')
            hold_period (int): Number of periods to hold positions
            verbose (bool): Enable detailed logging output
        """
        self.selection_mode = selection_mode
        self.verbose = verbose
        self.hold_period = hold_period

        # Initialize cluster label storage
        self.cluster_labels_long: List[int] = []
        self.cluster_labels_short: List[int] = []

        # Setup logging
        if self.verbose:
            logging.basicConfig(level=logging.INFO)

    def assess_clusters(
        self, labels: np.ndarray, returns: Union[np.ndarray, pd.Series], n_clusters: int
    ) -> Tuple[List[int], List[int]]:
        """
        Assess each cluster's performance and select the best performing clusters.

        Args:
            labels (np.ndarray): Cluster labels for each data point
            returns (Union[np.ndarray, pd.Series]): Returns data
            n_clusters (int): Total number of clusters
        """
        self.cluster_labels_long.clear()
        self.cluster_labels_short.clear()

        # Convert numpy array to pandas Series if needed
        if isinstance(returns, np.ndarray):
            returns = pd.Series(
                returns,
                index=pd.date_range(start="2000-01-01", periods=len(returns), freq="D"),
            )

        # Adjust lengths to match
        min_length = min(len(labels), len(returns))
        labels = labels[:min_length]
        returns = returns[:min_length]

        # Store cluster scores
        cluster_scores = []
        baseline_long = []
        baseline_short = []

        # Calculate baseline metrics with adjusted lengths
        baseline_metrics = self._compute_baseline_metrics(returns)

        # Evaluate each cluster
        for label in range(n_clusters):
            score, long_metrics, short_metrics = self._evaluate_single_cluster(
                label, labels, returns
            )

            cluster_scores.append(score)

            # Handle baseline selection mode
            if self.selection_mode == "baseline":
                self._assess_baseline_performance(
                    label,
                    long_metrics,
                    short_metrics,
                    baseline_metrics,
                    baseline_long,
                    baseline_short,
                )

        # Select best performing clusters
        if self.selection_mode == "baseline":
            self.cluster_labels_long = baseline_long
            self.cluster_labels_short = baseline_short
        else:
            self._select_best_clusters(cluster_scores)

        if self.verbose:
            logging.info(f"Selected long clusters: {self.cluster_labels_long}")
            logging.info(f"Selected short clusters: {self.cluster_labels_short}")

        return self.cluster_labels_long, self.cluster_labels_short

    def compute_performance(
        self, labels: np.ndarray, returns: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Compute overall performance metrics for selected clusters.

        Args:
            labels (np.ndarray): Cluster labels for each data point
            returns (Union[np.ndarray, pd.Series]): Returns data
        """
        # Adjust lengths to match
        min_length = min(len(labels), len(returns))
        labels = labels[:min_length]

        if isinstance(returns, pd.Series):
            returns = returns[:min_length]
        else:
            returns = returns[:min_length]

        # Generate signals from cluster labels
        signals = self._generate_signals(labels)

        # Apply holding period
        signals = self._apply_holding_period(signals)

        # Calculate returns
        if isinstance(returns, pd.Series):
            strategy_returns = pd.Series(signals * returns.values, index=returns.index)
        else:
            strategy_returns = signals * returns

        return self.compute_martin(strategy_returns)

    def compute_martin(self, returns: Union[np.ndarray, pd.Series]) -> float:
        """Modified to ensure returns has datetime index"""
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.date_range(
                start="2000-01-01", periods=len(returns), freq="D"
            )

        return qt.stats.ulcer_performance_index(returns)

    def find_optimal_clusters(
        self, data: np.ndarray, min_clusters: int = 2, max_clusters: int = 50
    ) -> int:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            data (np.ndarray): Input data for clustering
            min_clusters (int): Minimum number of clusters to try
            max_clusters (int): Maximum number of clusters to try

        Returns:
            int: Optimal number of clusters
        """
        n_clusters_range = range(min_clusters, max_clusters)
        silhouette_scores = []

        for n in n_clusters_range:
            score = self._compute_clustering_score(data, n)
            silhouette_scores.append(score)

        # Find the elbow point
        kneedle = KneeLocator(
            n_clusters_range,
            np.array(silhouette_scores),
            S=1.0,
            curve="convex",
            direction="decreasing",
        )

        optimal_n = kneedle.knee

        if self.verbose:
            logging.info(f"Optimal number of clusters: {optimal_n}")

        return optimal_n

    def _compute_baseline_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Compute baseline performance metrics for comparison.

        Args:
            returns (pd.Series): Returns data

        Returns:
            Dict[str, float]: Dictionary of baseline metrics
        """
        return {
            "profit_factor": max(qt.stats.profit_factor(returns), 0),
            "sharpe_ratio": max(qt.stats.sharpe(returns), 0),
            "upi": max(qt.stats.ulcer_performance_index(returns), 1),
            "max_dd": qt.stats.max_drawdown(returns),
        }

    def _evaluate_single_cluster(
        self, label: int, labels: np.ndarray, returns: Union[np.ndarray, pd.Series]
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Evaluate performance metrics for a single cluster.

        Args:
            label (int): Cluster label to evaluate
            labels (np.ndarray): All cluster labels
            returns (Union[np.ndarray, pd.Series]): Returns data

        Returns:
            Tuple[float, Dict[str, float], Dict[str, float]]:
                Martin score, long metrics, and short metrics
        """
        # Create mask for the label
        mask_label = labels == label

        # Generate signals
        signals = mask_label.astype(int)
        signals = self._apply_holding_period(signals)

        # Handle returns data
        if isinstance(returns, pd.Series):
            returns_index = returns.index
            returns_values = returns.values
        else:
            returns_values = returns
            returns_index = pd.date_range(
                start="2000-01-01", periods=len(returns), freq="D"
            )

        # Adjust lengths to match
        min_length = min(len(signals), len(returns_values))
        signals = signals[:min_length]
        returns_values = returns_values[:min_length]
        returns_index = returns_index[:min_length]

        # Calculate cluster returns with adjusted lengths
        cluster_returns = pd.Series(signals * returns_values, index=returns_index)
        martin_score = self.compute_martin(cluster_returns)

        # Calculate metrics for both directions using the length-adjusted data
        long_metrics = self._calculate_direction_metrics(
            pd.Series(returns_values, index=returns_index), 1
        )
        short_metrics = self._calculate_direction_metrics(
            pd.Series(returns_values, index=returns_index), -1
        )

        return (
            0 if np.isinf(martin_score) or np.isnan(martin_score) else martin_score,
            long_metrics,
            short_metrics,
        )

    def _calculate_direction_metrics(
        self, returns: np.ndarray, direction: int
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for a specific trading direction.

        Args:
            returns (np.ndarray): Returns data
            direction (int): Trading direction (1 for long, -1 for short)

        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        ret = pd.Series(returns * direction)
        return {
            "profit_factor": qt.stats.profit_factor(ret),
            "sharpe": qt.stats.rolling_sharpe(ret).mean(),
            "upi": qt.stats.ulcer_performance_index(ret),
            "max_dd": qt.stats.to_drawdown_series(ret).mean(),
        }

    def _assess_baseline_performance(
        self,
        label: int,
        long_metrics: Dict[str, float],
        short_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        baseline_long: List[int],
        baseline_short: List[int],
    ):
        """
        Assess cluster performance against baseline metrics.

        Args:
            label (int): Cluster label
            long_metrics (Dict[str, float]): Long position metrics
            short_metrics (Dict[str, float]): Short position metrics
            baseline_metrics (Dict[str, float]): Baseline metrics
            baseline_long (List[int]): List to store long cluster labels
            baseline_short (List[int]): List to store short cluster labels
        """
        for metrics, direction_list in [
            (long_metrics, baseline_long),
            (short_metrics, baseline_short),
        ]:
            if (
                not np.isinf(metrics["upi"])
                and not np.isnan(metrics["upi"])
                and metrics["profit_factor"] > baseline_metrics["profit_factor"]
                and metrics["sharpe"] > baseline_metrics["sharpe_ratio"]
                and metrics["upi"] > baseline_metrics["upi"]
                and metrics["max_dd"] > baseline_metrics["max_dd"]
            ):
                direction_list.append(label)

    def _select_best_clusters(self, scores: List[float]):
        """
        Select best performing clusters based on scores.

        Args:
            scores (List[float]): List of cluster performance scores
        """
        self.cluster_labels_long.append(np.argmax(scores))
        self.cluster_labels_short.append(np.argmin(scores))

    def _generate_signals(self, labels: np.ndarray) -> np.ndarray:
        """
        Generate trading signals from cluster labels.

        Args:
            labels (np.ndarray): Cluster labels

        Returns:
            np.ndarray: Generated trading signals
        """
        signals = np.zeros_like(labels, dtype=float)
        signals[np.isin(labels, self.cluster_labels_long)] = 1
        signals[np.isin(labels, self.cluster_labels_short)] = -1
        return signals

    def _apply_holding_period(self, signals: np.ndarray) -> np.ndarray:
        """
        Apply holding period to trading signals.

        Args:
            signals (np.ndarray): Trading signals

        Returns:
            np.ndarray: Signals with holding period applied
        """
        new_signals = np.zeros_like(signals)
        nonzero = np.where(signals != 0)[0]
        prev_index = -self.hold_period - 1

        for index in nonzero:
            signal = signals[index]
            if index > prev_index + self.hold_period:
                prev_index = index
                start_index = index + 1
                end_index = min(index + self.hold_period + 1, len(signals))
                new_signals[start_index:end_index] = signal

        return new_signals

    def _compute_clustering_score(self, data: np.ndarray, n_clusters: int) -> float:
        """
        Compute clustering score using multiple clustering algorithms.

        Args:
            data (np.ndarray): Input data
            n_clusters (int): Number of clusters to evaluate

        Returns:
            float: Average silhouette score
        """
        birch = Birch(n_clusters=n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto")

        birch.fit(data)
        kmeans.fit(data)

        return np.mean(
            [
                silhouette_score(data, birch.labels_),
                silhouette_score(data, kmeans.labels_),
            ]
        )


class DataPreprocessor:
    """
    Handles data preprocessing operations for time series data including normalization,
    training set generation, and dimensionality reduction.

    This class centralizes all data preprocessing operations that were previously part of
    the Miner class, providing a cleaner interface for data transformations while maintaining
    the same core functionality.

    Attributes:
        verbose (bool): Controls logging verbosity
        _data_min (float): Minimum value from training data for scaling
        _data_max (float): Maximum value from training data for scaling
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the DataPreprocessor.

        Args:
            verbose (bool): Enable detailed logging output
        """
        self.verbose = verbose
        self._data_min: Optional[float] = None
        self._data_max: Optional[float] = None

        # Setup logging
        if self.verbose:
            logging.basicConfig(level=logging.INFO)

    def preprocess_data(self, data: np.ndarray, test_mode: bool = False) -> np.ndarray:
        """
        Perform series-level data transformations including handling NaN/Inf values,
        scaling, and normalization.

        Args:
            data (np.ndarray): Input data array to preprocess
            test_mode (bool): Whether preprocessing is for testing data

        Returns:
            np.ndarray: Preprocessed data

        Raises:
            ValueError: If data contains NaN or infinite values
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Handle NaN/Inf values
        if np.any(~np.isfinite(data)):
            raise ValueError("Data contains NaN or infinite values")

        # Store scaling parameters during training
        if not test_mode:
            self._store_scaling_parameters(data)

        # Apply transformations
        transformed = self._transform_and_scale(data)

        if self.verbose:
            logging.info(f"Preprocessed data shape: {transformed.shape}")

        return transformed

    def generate_training_set(
        self, data: np.ndarray, n_lookback: int, test_mode: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate training input (X) and optionally target (y) datasets using sliding windows.

        Args:
            data (np.ndarray): Input time series data
            n_lookback (int): Number of lookback periods for each window
            test_mode (bool): Whether generating data for testing

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                - If test_mode: returns windows only
                - If not test_mode: returns (windows, targets)

        Raises:
            ValueError: If data length is less than lookback period
        """
        # Validate inputs
        if len(data) < n_lookback:
            raise ValueError(
                f"Data length ({len(data)}) must be >= lookback period ({n_lookback})"
            )

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Pre-allocate array for efficiency
        n_windows = len(data) - n_lookback + 1
        windows = np.zeros((n_windows, n_lookback))

        # Generate sliding windows
        for i in range(n_windows):
            windows[i] = data[i : i + n_lookback]

        if test_mode:
            return windows

        # For training, also generate target values
        targets = data[n_lookback:]

        if self.verbose:
            logging.info(f"Generated {n_windows} windows of size {n_lookback}")

        return windows, targets

    def transform_data(
        self,
        data: np.ndarray,
        n_pivots: int,
        reducer: BaseReducer,
        test_mode: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Transform the data through dimensionality reduction and normalization.

        Args:
            data (np.ndarray): Input data windows
            n_pivots (int): Number of pivot points for dimensionality reduction
            reducer: The reducer object for dimensionality reduction
            test_mode (bool): Whether transforming data for testing

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                - If test_mode: returns (transformed_data, unique_indices)
                - If not test_mode: returns transformed_data only
        """
        if reducer is None:
            raise ValueError("Reducer not initialized")

        if n_pivots < 1:
            raise ValueError("n_pivots must be positive")

        # Apply dimensionality reduction
        try:
            pivots = reducer.transform(data)
        except ValueError as e:
            if test_mode:
                # Handle validation failures gracefully in test mode by padding
                if len(data) < n_pivots:
                    # Pad the data to reach n_pivots length
                    pad_length = n_pivots - len(data)
                    padded_data = np.pad(
                        data,
                        ((0, pad_length), (0, 0)) if data.ndim > 1 else (0, pad_length),
                        mode="edge",
                    )  # Use edge padding to extend the last value
                    pivots = reducer.transform(padded_data)
                else:
                    return np.zeros((1, n_pivots)), np.array([0])
            else:
                raise e

        # Remove duplicate patterns
        pivots, mask_unique = self.remove_duplicates(pivots, n_pivots)

        # Normalize each window
        normalized_data = np.apply_along_axis(
            self.normalize_standard, axis=1, arr=pivots
        )

        if self.verbose:
            logging.info(f"Transformed data shape: {normalized_data.shape}")

        if test_mode:
            return normalized_data, np.where(mask_unique)[0]

        return normalized_data

    def remove_duplicates(
        self, pivots: np.ndarray, n_pivots: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove duplicate pivot patterns from the data.

        Args:
            pivots (np.ndarray): Array of pivot points
            n_pivots (int): Number of pivot points

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Filtered pivot points
                - Boolean mask of unique patterns
        """
        null_pivots = [-1] * n_pivots

        # Compare each group with the previous one, excluding first and last items
        is_duplicate = np.all(pivots[1:, 1:-1] == pivots[:-1, 1:-1], axis=1)

        # Replace duplicates with null_pivots
        pivots[1:][is_duplicate] = null_pivots

        # Remove null_pivots
        mask_unique = np.all(pivots != null_pivots, axis=1)
        pivots = pivots[mask_unique]

        if self.verbose:
            logging.info(f"Removed {np.sum(~mask_unique)} duplicate patterns")

        return pivots, mask_unique

    def normalize_standard(self, points: np.ndarray) -> np.ndarray:
        """
        Apply standard normalization to a set of points.

        Args:
            points (np.ndarray): Input points to normalize

        Returns:
            np.ndarray: Normalized points
        """
        points = np.array(points)
        return (points - np.mean(points)) / (np.std(points) + 1e-10)

    def _store_scaling_parameters(self, data: np.ndarray):
        """
        Store min and max values from training data for consistent scaling.

        Args:
            data (np.ndarray): Training data to derive scaling parameters
        """
        self._data_min = np.min(data)
        self._data_max = np.max(data)

    def _transform_and_scale(self, data: np.ndarray) -> np.ndarray:
        """
        Apply transformation and scaling to the data.

        Args:
            data (np.ndarray): Input data to transform and scale

        Returns:
            np.ndarray: Transformed and scaled data
        """
        # Apply inverse hyperbolic sine transformation
        transformed = np.arcsinh(data)

        # Apply scaling if parameters exist
        if self._data_min is not None and self._data_max is not None:
            transformed = (transformed - self._data_min) / (
                self._data_max - self._data_min
            )

        return transformed


class Visualizer:
    """
    Handles all visualization functionality for the mining framework.

    This class centralizes all plotting and visualization operations, providing methods
    for cluster visualization, performance analysis, signal distribution, and interactive
    dashboards.

    Attributes:
        style (str): Matplotlib style to use
        figsize (Tuple[int, int]): Default figure size
        verbose (bool): Controls logging verbosity
        save_path (Optional[Path]): Default path for saving plots
    """

    def __init__(
        self,
        style: str = "default",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None,
        verbose: bool = False,
    ):
        """
        Initialize the Visualizer.

        Args:
            style (str): Matplotlib style to use
            figsize (Tuple[int, int]): Default figure size
            save_path (Optional[Path]): Default path for saving plots
            verbose (bool): Enable detailed logging output
        """
        self.style = style
        self.figsize = figsize
        self.save_path = save_path
        self.verbose = verbose

        # Import required libraries
        import seaborn as sns

        sns.set_style("darkgrid")  # Set seaborn style directly

        # Set style
        plt.style.use(self.style)

        # Setup logging
        if self.verbose:
            logging.basicConfig(level=logging.INFO)

    def plot_clusters(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        title: str = "Cluster Analysis",
        interactive: bool = True,
        save: bool = False,
    ) -> None:
        """
        Visualize clusters and their distribution in the data.

        Args:
            data (np.ndarray): Time series data
            labels (np.ndarray): Cluster labels
            title (str): Plot title
            interactive (bool): Use interactive plotly plot
            save (bool): Whether to save the plot
        """
        if interactive:
            self._plot_clusters_interactive(data, labels, title)
        else:
            self._plot_clusters_static(data, labels, title)

        if save:
            self._save_plot(title)

    def plot_equity_curve(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        title: str = "Equity Curve",
        include_metrics: bool = True,
        save: bool = False,
    ) -> None:
        """
        Plot cumulative returns and optionally compare with benchmark.

        Args:
            returns (np.ndarray): Strategy returns
            benchmark_returns (np.ndarray, optional): Benchmark returns
            title (str): Plot title
            include_metrics (bool): Include performance metrics
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=self.figsize)

        # Plot strategy returns
        cum_returns = np.cumsum(returns)
        plt.plot(cum_returns, label="Strategy", linewidth=2)

        # Plot benchmark if provided
        if benchmark_returns is not None:
            cum_benchmark = np.cumsum(benchmark_returns)
            plt.plot(cum_benchmark, label="Benchmark", linewidth=2, alpha=0.7)

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Cumulative Returns")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if include_metrics:
            self._add_performance_metrics(returns, benchmark_returns)

        if save:
            self._save_plot(title)

        plt.show()

    def plot_signal_distribution(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        title: str = "Signal Distribution",
        save: bool = False,
    ) -> None:
        """
        Visualize the distribution of trading signals across clusters.

        Args:
            signals (np.ndarray): Trading signals
            labels (np.ndarray): Cluster labels
            title (str): Plot title
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=self.figsize)

        # Create signal distribution plot
        unique_labels = np.unique(labels[labels != -1])
        signal_dist = []

        for label in unique_labels:
            mask = labels == label
            label_signals = signals[mask]
            signal_dist.append(
                [
                    np.sum(label_signals == 1) / len(label_signals),
                    np.sum(label_signals == 0) / len(label_signals),
                    np.sum(label_signals == -1) / len(label_signals),
                ]
            )

        signal_dist = np.array(signal_dist)

        # Plot stacked bar chart
        bottom = np.zeros(len(unique_labels))

        for i, signal_type in enumerate(["Long", "Neutral", "Short"]):
            plt.bar(
                unique_labels,
                signal_dist[:, i],
                bottom=bottom,
                label=signal_type,
                alpha=0.7,
            )
            bottom += signal_dist[:, i]

        plt.title(title)
        plt.xlabel("Cluster")
        plt.ylabel("Signal Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save:
            self._save_plot(title)

        plt.show()

    def plot_cluster_performance(
        self,
        cluster_metrics: Dict[int, Dict[str, float]],
        title: str = "Cluster Performance",
        metrics: Optional[List[str]] = None,
        save: bool = False,
    ) -> None:
        """
        Visualize performance metrics for each cluster.

        Args:
            cluster_metrics (Dict[int, Dict[str, float]]): Performance metrics by cluster
            title (str): Plot title
            metrics (List[str], optional): Specific metrics to plot
            save (bool): Whether to save the plot
        """
        if metrics is None:
            metrics = ["martin_ratio", "sharpe_ratio", "profit_factor"]

        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))

        # Convert axes to array if single metric
        if n_metrics == 1:
            axes = np.array([axes])

        for i, metric in enumerate(metrics):
            values = [m[metric] for m in cluster_metrics.values()]
            ax = axes[i]

            sns.barplot(x=list(cluster_metrics.keys()), y=values, ax=ax)
            ax.set_title(f"{metric} by Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            self._save_plot(title)

        plt.show()

    def create_performance_dashboard(
        self,
        returns: np.ndarray,
        signals: np.ndarray,
        metrics: Dict[str, float],
        title: str = "Performance Dashboard",
        save: bool = False,
    ) -> None:
        """
        Create an interactive dashboard of performance metrics.

        Args:
            returns (np.ndarray): Strategy returns
            signals (np.ndarray): Trading signals
            metrics (Dict[str, float]): Performance metrics
            title (str): Dashboard title
            save (bool): Whether to save the dashboard
        """
        # Create subplot figure with proper specs for pie chart
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Equity Curve",
                "Rolling Sharpe Ratio",
                "Returns Distribution",
                "Signal Distribution",
                "Drawdown Analysis",
                "Rolling Volatility",
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "domain"}],  # "domain" type for pie chart
                [{"type": "xy"}, {"type": "xy"}],
            ],
        )

        # Equity Curve
        fig.add_trace(go.Scatter(y=np.cumsum(returns), name="Equity"), row=1, col=1)

        # Rolling Sharpe
        window = 252
        rolling_sharpe = self._calculate_rolling_sharpe(returns, window)
        fig.add_trace(go.Scatter(y=rolling_sharpe, name="Rolling Sharpe"), row=1, col=2)

        # Returns Distribution
        fig.add_trace(go.Histogram(x=returns, name="Returns Dist"), row=2, col=1)

        # Signal Distribution (pie chart)
        signal_counts = np.array(
            [np.sum(signals == 1), np.sum(signals == 0), np.sum(signals == -1)]
        )
        fig.add_trace(
            go.Pie(
                labels=["Long", "Neutral", "Short"],
                values=signal_counts,
                name="Signals",
            ),
            row=2,
            col=2,  # This will now work with the "domain" type subplot
        )

        # Drawdown Analysis
        drawdowns = self._calculate_drawdowns(returns)
        fig.add_trace(go.Scatter(y=drawdowns, name="Drawdown"), row=3, col=1)

        # Rolling Volatility
        rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(y=rolling_vol, name="Volatility"), row=3, col=2)

        # Update layout
        fig.update_layout(height=900, showlegend=True, title_text=title)

        # Add metrics annotations
        self._add_metrics_annotations(fig, metrics)

        if save and self.save_path:
            fig.write_html(self.save_path / f"{title}.html")

        fig.show()

    def _plot_clusters_interactive(
        self, data: np.ndarray, labels: np.ndarray, title: str
    ) -> None:
        """
        Create interactive cluster visualization using plotly.

        Args:
            data (np.ndarray): Time series data
            labels (np.ndarray): Cluster labels
            title (str): Plot title
        """
        fig = go.Figure()

        # Plot data
        fig.add_trace(
            go.Scatter(
                y=data, mode="lines", name="Data", line=dict(color="blue", width=1)
            )
        )

        # Plot cluster labels
        fig.add_trace(
            go.Scatter(
                y=labels, mode="lines", name="Clusters", line=dict(color="red", width=1)
            )
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified",
            showlegend=True,
        )

        fig.show()

    def _plot_clusters_static(
        self, data: np.ndarray, labels: np.ndarray, title: str
    ) -> None:
        """
        Create static cluster visualization using matplotlib.

        Args:
            data (np.ndarray): Time series data
            labels (np.ndarray): Cluster labels
            title (str): Plot title
        """
        plt.figure(figsize=self.figsize)

        plt.plot(data, label="Data", alpha=0.7)
        plt.plot(labels, label="Clusters", alpha=0.5)

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()

    def _add_performance_metrics(
        self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray]
    ) -> None:
        """
        Add performance metrics to the current plot.

        Args:
            returns (np.ndarray): Strategy returns
            benchmark_returns (np.ndarray, optional): Benchmark returns
        """
        metrics = [
            f"Sharpe: {self._calculate_sharpe(returns):.2f}",
            f"Max DD: {self._calculate_max_drawdown(returns):.2%}",
            f"Ann. Return: {self._calculate_annual_return(returns):.2%}",
        ]

        if benchmark_returns is not None:
            metrics.append(
                f"Info Ratio: {self._calculate_information_ratio(returns, benchmark_returns):.2f}"
            )

        plt.figtext(0.02, 0.98, "\n".join(metrics), fontsize=10, va="top")

    def _calculate_rolling_sharpe(self, returns: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate rolling Sharpe ratio.

        Args:
            returns (np.ndarray): Returns array
            window (int): Rolling window size

        Returns:
            np.ndarray: Rolling Sharpe ratio
        """
        series = pd.Series(returns)
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        return (rolling_mean / rolling_std) * np.sqrt(252)

    def _calculate_drawdowns(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown series.

        Args:
            returns (np.ndarray): Returns array

        Returns:
            np.ndarray: Drawdown series
        """
        cum_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns - running_max
        return drawdowns

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        drawdowns = self._calculate_drawdowns(returns)
        return np.min(drawdowns)

    def _calculate_annual_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return."""
        return np.mean(returns) * 252

    def _calculate_information_ratio(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        """Calculate information ratio."""
        active_returns = returns - benchmark_returns
        return np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)

    def _add_metrics_annotations(
        self, fig: go.Figure, metrics: Dict[str, float]
    ) -> None:
        """
        Add metrics annotations to plotly figure.

        Args:
            fig (go.Figure): Plotly figure
            metrics (Dict[str, float]): Performance metrics
        """
        annotation_text = "<br>".join(
            [f"{key}: {value:.4f}" for key, value in metrics.items()]
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0,
            y=1.05,
            text=annotation_text,
            showarrow=False,
            font=dict(size=10),
            align="left",
        )

    def _save_plot(self, title: str) -> None:
        """
        Save the current plot to file.

        Args:
            title (str): Plot title/filename

        Raises:
            ValueError: If save_path is not set
        """
        if self.save_path is None:
            raise ValueError("save_path must be set to save plots")

        # Create directory if it doesn't exist
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Clean filename
        filename = "".join(
            c for c in title if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()

        # Save plot
        plt.savefig(self.save_path / f"{filename}.png", dpi=300, bbox_inches="tight")

        if self.verbose:
            logging.info(f"Plot saved to {self.save_path / f'{filename}.png'}")

    def plot_cluster_transitions(
        self, labels: np.ndarray, title: str = "Cluster Transitions", save: bool = False
    ) -> None:
        """
        Visualize transitions between clusters over time.

        Args:
            labels (np.ndarray): Cluster labels sequence
            title (str): Plot title
            save (bool): Whether to save the plot
        """
        # Remove placeholder labels (-1)
        valid_labels = labels[labels != -1]

        # Calculate transition matrix
        unique_labels = np.unique(valid_labels)
        n_clusters = len(unique_labels)
        transitions = np.zeros((n_clusters, n_clusters))

        for i in range(len(valid_labels) - 1):
            from_cluster = np.where(unique_labels == valid_labels[i])[0][0]
            to_cluster = np.where(unique_labels == valid_labels[i + 1])[0][0]
            transitions[from_cluster, to_cluster] += 1

        # Normalize transitions
        row_sums = transitions.sum(axis=1)
        transition_probs = transitions / row_sums[:, np.newaxis]

        # Create heatmap
        plt.figure(figsize=self.figsize)
        sns.heatmap(
            transition_probs,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=unique_labels,
            yticklabels=unique_labels,
        )

        plt.title(title)
        plt.xlabel("To Cluster")
        plt.ylabel("From Cluster")

        if save:
            self._save_plot(title)

        plt.show()

    def plot_cluster_characteristics(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        title: str = "Cluster Characteristics",
        save: bool = False,
    ) -> None:
        """
        Visualize characteristic patterns for each cluster.

        Args:
            data (np.ndarray): Time series data windows
            labels (np.ndarray): Cluster labels
            title (str): Plot title
            save (bool): Whether to save the plot
        """
        unique_labels = np.unique(labels[labels != -1])
        n_clusters = len(unique_labels)

        # Create subplot grid
        fig, axes = plt.subplots(
            int(np.ceil(n_clusters / 2)), 2, figsize=(15, 5 * np.ceil(n_clusters / 2))
        )
        axes = axes.ravel()

        for i, label in enumerate(unique_labels):
            # Get all sequences for this cluster
            cluster_sequences = data[labels == label]

            if len(cluster_sequences) > 0:
                # Plot mean pattern
                mean_pattern = np.mean(cluster_sequences, axis=0)
                std_pattern = np.std(cluster_sequences, axis=0)

                axes[i].plot(mean_pattern, "b-", label="Mean Pattern")
                axes[i].fill_between(
                    range(len(mean_pattern)),
                    mean_pattern - std_pattern,
                    mean_pattern + std_pattern,
                    alpha=0.2,
                    color="b",
                    label="±1 STD",
                )

                axes[i].set_title(f"Cluster {label}")
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()

        plt.tight_layout()

        if save:
            self._save_plot(title)

        plt.show()

    def plot_rolling_metrics(
        self,
        returns: np.ndarray,
        window: int = 252,
        metrics: Optional[List[str]] = None,
        title: str = "Rolling Performance Metrics",
        save: bool = False,
    ) -> None:
        """
        Plot rolling performance metrics.

        Args:
            returns (np.ndarray): Strategy returns
            window (int): Rolling window size
            metrics (List[str], optional): Metrics to plot
            title (str): Plot title
            save (bool): Whether to save the plot
        """
        if metrics is None:
            metrics = ["sharpe", "volatility", "returns"]

        series = pd.Series(returns)
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        axes = np.array([axes]) if len(metrics) == 1 else axes

        for i, metric in enumerate(metrics):
            if metric == "sharpe":
                rolling_data = self._calculate_rolling_sharpe(returns, window)
                label = "Rolling Sharpe Ratio"
            elif metric == "volatility":
                rolling_data = series.rolling(window).std() * np.sqrt(252)
                label = "Rolling Volatility"
            else:  # returns
                rolling_data = series.rolling(window).mean() * 252
                label = "Rolling Returns"

            axes[i].plot(rolling_data, label=label)
            axes[i].set_title(label)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        plt.tight_layout()

        if save:
            self._save_plot(title)

        plt.show()


class ModelManager:
    """
    Handles model persistence and state management for the mining framework.

    This class provides functionality for saving and loading model states, cleaning up
    resources, and managing model configurations. It ensures proper serialization and
    state management across the framework components.

    Attributes:
        verbose (bool): Controls logging verbosity
        model_state (Dict): Current model state
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the ModelManager.

        Args:
            verbose (bool): Enable detailed logging output
        """
        self.verbose = verbose
        self.model_state: Dict[str, Any] = {}

        # Setup logging
        if self.verbose:
            logging.basicConfig(level=logging.INFO)

    def save_model(self, model: Any, path: Path) -> None:
        """
        Save the model state to a file.

        Args:
            model: Model instance to save
            path (Path): Path to save the model

        Raises:
            ValueError: If path is invalid
            IOError: If save operation fails
        """
        # Validate path
        if not isinstance(path, Path):
            path = Path(path)

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Get current model state
            self.model_state = self._extract_model_state(model)

            # Save model state
            with open(path, "wb") as f:
                pickle.dump(self.model_state, f)

            if self.verbose:
                logging.info(f"Model saved successfully to {path}")

        except Exception as e:
            logging.error(f"Error saving model: {e!s}")
            raise IOError(f"Failed to save model to {path}: {e!s}")

    def load_model(self, path: Path) -> Dict[str, Any]:
        """
        Load a model state from a file.

        Args:
            path (Path): Path to load the model from

        Returns:
            Dict[str, Any]: Loaded model state

        Raises:
            FileNotFoundError: If model file doesn't exist
            IOError: If load operation fails
        """
        # Validate path
        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        try:
            # Load model state
            with open(path, "rb") as f:
                self.model_state = pickle.load(f)

            if self.verbose:
                logging.info(f"Model loaded successfully from {path}")

            return self.model_state

        except Exception as e:
            logging.error(f"Error loading model: {e!s}")
            raise IOError(f"Failed to load model from {path}: {e!s}")

    def cleanup_resources(self) -> None:
        """
        Clean up model resources and temporary data.

        This method ensures proper cleanup of memory and resources:
        - Clears training data
        - Resets temporary variables
        - Cleans up any cached computations
        """
        try:
            # Clear training data
            if "training_data" in self.model_state:
                self.model_state["training_data"] = None

            # Clear temporary variables
            temp_keys = [k for k in self.model_state if k.startswith("temp_")]
            for k in temp_keys:
                del self.model_state[k]

            # Clear numpy arrays
            for k, v in self.model_state.items():
                if isinstance(v, np.ndarray):
                    self.model_state[k] = None

            if self.verbose:
                logging.info("Model resources cleaned up successfully")

        except Exception as e:
            logging.error(f"Error during cleanup: {e!s}")

    def get_model_state(self) -> Dict[str, Any]:
        """
        Get the current model state.

        Returns:
            Dict[str, Any]: Current model state dictionary
        """
        return self.model_state

    def set_model_state(self, state: Dict[str, Any]) -> None:
        """
        Set the model state.

        Args:
            state (Dict[str, Any]): Model state to set

        Raises:
            ValueError: If state is invalid
        """
        if not isinstance(state, dict):
            raise ValueError("Model state must be a dictionary")

        try:
            # Validate state format
            self._validate_state(state)

            # Update model state
            self.model_state = state.copy()

            if self.verbose:
                logging.info("Model state updated successfully")

        except Exception as e:
            logging.error(f"Error setting model state: {e!s}")
            raise

    def _extract_model_state(self, model: Any) -> Dict[str, Any]:
        """
        Extract state from model components.

        Args:
            model: Model instance

        Returns:
            Dict[str, Any]: Extracted model state
        """
        state = {
            "model_params": {
                "n_clusters": getattr(model, "n_clusters", None),
                "n_lookback": getattr(model, "n_lookback", None),
                "n_pivots": getattr(model, "n_pivots", None),
                "hold_period": getattr(model, "hold_period", None),
                "model_type": getattr(model, "model_type", None),
                "reducer_type": getattr(model, "reducer_type", None),
                "wavelet": getattr(model, "wavelet", None),
                "cluster_selection_mode": getattr(
                    model, "cluster_selection_mode", None
                ),
            },
            "preprocessing_params": {
                "data_min": getattr(model, "_data_min", None),
                "data_max": getattr(model, "_data_max", None),
            },
            "cluster_labels": {
                "long": getattr(model, "_cluster_labels_long", []),
                "short": getattr(model, "_cluster_labels_short", []),
            },
            "model_components": {
                "reducer": getattr(model, "reducer", None),
                "cluster_model": getattr(model, "cluster_model", None),
            },
        }
        return state

    def _validate_state(self, state: Dict[str, Any]) -> None:
        """
        Validate model state format and required components.

        Args:
            state (Dict[str, Any]): State to validate

        Raises:
            ValueError: If state format is invalid
        """
        required_keys = [
            "model_params",
            "preprocessing_params",
            "cluster_labels",
            "model_components",
        ]

        if not all(k in state for k in required_keys):
            missing = [k for k in required_keys if k not in state]
            raise ValueError(f"Missing required state components: {missing}")

        # Validate model parameters
        if not isinstance(state["model_params"], dict):
            raise ValueError("Invalid model_params format")

        # Validate preprocessing parameters
        if not isinstance(state["preprocessing_params"], dict):
            raise ValueError("Invalid preprocessing_params format")

        # Validate cluster labels
        if not all(k in state["cluster_labels"] for k in ["long", "short"]):
            raise ValueError("Invalid cluster_labels format")

        # Validate model components
        if not isinstance(state["model_components"], dict):
            raise ValueError("Invalid model_components format")
