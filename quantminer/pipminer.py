import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .classes.evaluators import (
    EvaluatorType,
    HoldingPeriodEvaluator,
    VectorPatternEvaluator,
)
from .classes.helpers import (
    DataPreprocessor,
    MiningCore,
    ModelManager,
    ModelTester,
    Visualizer,
)

# Ignore runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Miner:
    """
    Main orchestrator class that integrates all components of the mining framework.

    This class provides a high-level interface for the entire mining system,
    coordinating data preprocessing, clustering, evaluation, testing, visualization,
    and model management.

    Attributes:
        config (Dict): Configuration parameters
        preprocessor (DataPreprocessor): Handles data preprocessing
        core (MiningCore): Handles clustering and signal generation
        evaluator (ClusterEvaluator): Handles performance evaluation
        tester (ModelTester): Handles model testing
        visualizer (Visualizer): Handles visualization
        model_manager (ModelManager): Handles model persistence
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Miner with configuration.

        Args:
            config: Dictionary containing configuration parameters:
                - n_lookback: Number of lookback periods
                - n_pivots: Number of pivot points
                - n_clusters: Number of clusters
                - hold_period: Position holding period
                - model_type: Type of clustering model
                - reducer_type: Type of dimensionality reducer
                - wavelet: Type of wavelet
                - verbose: Enable logging
        """
        self.config = config
        self.verbose = config.get("verbose", False)

        # Setup logging
        if self.verbose:
            logging.basicConfig(level=logging.INFO)

        # Initialize components
        self.preprocessor = DataPreprocessor(verbose=self.verbose)

        self.core = MiningCore(
            n_clusters=config["n_clusters"],
            n_pivots=config["n_pivots"],
            n_lookback=config["n_lookback"],
            model_type=config["model_type"],
            reducer_type=config["reducer_type"],
            wavelet=config.get("wavelet", "coif1"),
            verbose=self.verbose,
        )

        self.evaluator = self._init_evaluator(config)

        self.tester = ModelTester(
            self.preprocessor,
            self.core,
            self.evaluator,
            hold_period=config["hold_period"],
            verbose=self.verbose,
        )

        self.visualizer = Visualizer(verbose=self.verbose)
        self.model_manager = ModelManager(verbose=self.verbose)

        # Add storage for training data and labels
        self._training_data: Optional[np.ndarray] = None
        self._training_labels: Optional[np.ndarray] = None
        self._processed_training_data: Optional[np.ndarray] = None

    def fit_old(
        self,
        data: np.ndarray,
        price_data: Optional[np.ndarray] = None,
        returns_series: Optional[pd.Series] = None,
    ) -> float:
        """
        Fit the model to training data.

        Args:
            data: Input training data
            price_data: Optional price data for returns calculation
            returns_series: Optional pre-calculated returns with datetime index

        Returns:
            float: Training performance score

        Raises:
            ValueError: If data validation fails
        """
        if price_data is None:
            price_data = np.copy(data)

        try:
            # Preprocess data
            processed_data = self.preprocessor.preprocess_data(data)
            processed_price = self.preprocessor.preprocess_data(price_data)

            # Handle returns calculation
            if returns_series is not None:
                returns = returns_series
            else:
                returns = pd.Series(
                    np.diff(processed_price, prepend=processed_price[0]),
                    index=pd.date_range(
                        start="2000-01-01", periods=len(processed_price), freq="D"
                    ),
                )

            # Generate training windows
            windows, _ = self.preprocessor.generate_training_set(
                processed_data, self.config["n_lookback"]
            )

            # Transform data
            transformed_data = self.preprocessor.transform_data(
                windows, self.config["n_pivots"], self.core.reducer
            )

            # Fit clustering model
            self.core.fit(transformed_data)

            # Store training data and labels
            self._training_data = data
            self._processed_training_data = processed_data
            self._training_labels = self.core.labels_

            # Evaluate clusters using new evaluator
            self.evaluator.evaluate_all_clusters(
                data=transformed_data, labels=self.core.labels_, returns=returns
            )

            # Calculate overall performance
            performance = self.evaluator.compute_performance(self.core.labels_, returns)

            if self.verbose:
                logging.info(f"Training completed with performance: {performance:.4f}")

            return performance

        except Exception as e:
            logging.error(f"Error during training: {e!s}")
            raise

    def fit(
        self,
        data: np.ndarray,
        price_data: Optional[np.ndarray] = None,
        returns_series: Optional[pd.Series] = None,
    ) -> float:
        """
        Fit the model to training data.

        Args:
            data: Input training data
            price_data: Optional price data for returns calculation
            returns_series: Optional pre-calculated returns with datetime index

        Returns:
            float: Training performance score
        """
        if price_data is None:
            price_data = np.copy(data)

        try:
            # Preprocess data
            processed_data = self.preprocessor.preprocess_data(data)
            processed_price = self.preprocessor.preprocess_data(price_data)

            # Handle returns calculation
            if returns_series is not None:
                returns = returns_series
            else:
                returns = pd.Series(
                    np.diff(processed_price, prepend=processed_price[0]),
                    index=pd.date_range(
                        start="2000-01-01", periods=len(processed_price), freq="D"
                    ),
                )

            # Generate training windows
            windows, _ = self.preprocessor.generate_training_set(
                processed_data, self.config["n_lookback"]
            )

            # Transform data
            transformed_data = self.preprocessor.transform_data(
                windows, self.config["n_pivots"], self.core.reducer
            )

            # Fit clustering model
            self.core.fit(transformed_data)

            # Store training data and labels
            self._training_data = data
            self._processed_training_data = processed_data
            self._training_labels = self.core.labels_

            # Align returns with the transformed data windows
            # Each window label corresponds to the return following that window
            aligned_returns = returns[
                self.config["n_lookback"] : len(self.core.labels_)
                + self.config["n_lookback"]
            ]

            # Evaluate clusters using new evaluator
            self.evaluator.evaluate_all_clusters(
                data=transformed_data, labels=self.core.labels_, returns=aligned_returns
            )

            # Calculate overall performance
            performance = self.evaluator.compute_performance(
                self.core.labels_, aligned_returns
            )

            if self.verbose:
                logging.info(f"Training completed with performance: {performance:.4f}")

            return performance

        except Exception as e:
            logging.error(f"Error during training: {e!s}")
            raise

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Generate predictions for new data.

        Args:
            data: Input data for prediction

        Returns:
            np.ndarray: Predicted labels/signals

        Raises:
            ValueError: If model hasn't been fitted
        """
        try:
            # Preprocess data
            processed_data = self.preprocessor.preprocess_data(data, test_mode=True)

            # Generate windows
            windows = self.preprocessor.generate_training_set(
                processed_data, self.config["n_lookback"], test_mode=True
            )

            # Transform data
            transformed_data, indices = self.preprocessor.transform_data(
                windows, self.config["n_pivots"], self.core.reducer, test_mode=True
            )

            # Generate predictions
            predictions = self.core.predict(transformed_data)

            return predictions

        except Exception as e:
            logging.error(f"Error during prediction: {e!s}")
            raise

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using the fitted model.

        Args:
            data: Input data to transform

        Returns:
            np.ndarray: Transformed data
        """
        processed_data = self.preprocessor.preprocess_data(data, test_mode=True)
        windows = self.preprocessor.generate_training_set(
            processed_data, self.config["n_lookback"], test_mode=True
        )
        transformed_data = self.preprocessor.transform_data(
            windows, self.config["n_pivots"], self.core.reducer, test_mode=True
        )
        return transformed_data

    def evaluate(
        self,
        data: np.ndarray,
        price_data: Optional[np.ndarray] = None,
        returns_series: Optional[pd.Series] = None,
        plot_results: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            data: Test data
            price_data: Optional price data for returns calculation
            returns_series: Optional pre-calculated returns with datetime index
            plot_results: Whether to plot performance visualizations

        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        if price_data is None:
            price_data = np.copy(data)

        try:
            # Preprocess data
            processed_data = self.preprocessor.preprocess_data(data, test_mode=True)
            processed_price = self.preprocessor.preprocess_data(
                price_data, test_mode=True
            )

            # Generate windows and transform data
            windows = self.preprocessor.generate_training_set(
                processed_data, self.config["n_lookback"], test_mode=True
            )
            transformed_data, valid_indices = self.preprocessor.transform_data(
                windows, self.config["n_pivots"], self.core.reducer, test_mode=True
            )

            # Generate predictions
            predictions = self.core.predict(transformed_data)

            # Handle returns alignment
            if returns_series is not None:
                returns = returns_series
            else:
                returns = pd.Series(
                    np.diff(processed_price, prepend=processed_price[0]),
                    index=pd.date_range(
                        start="2000-01-01", periods=len(processed_price), freq="D"
                    ),
                )

            # Adjust indices for lookback and transform window
            valid_indices = valid_indices + self.config["n_lookback"] - 1
            aligned_returns = returns.iloc[valid_indices]

            # Ensure predictions and returns match
            assert len(predictions) == len(aligned_returns), (
                f"Length mismatch: predictions={len(predictions)}, "
                f"returns={len(aligned_returns)}"
            )

            # Evaluate using aligned data
            if isinstance(self.evaluator, VectorPatternEvaluator):
                metrics = self.evaluator.evaluate_all_clusters(
                    data=transformed_data, labels=predictions
                )
            else:
                metrics = self.evaluator.evaluate_all_clusters(
                    data=transformed_data, labels=predictions, returns=aligned_returns
                )

            # Format metrics for visualization
            display_metrics = {}
            for cluster_id, stats in metrics.items():
                if hasattr(stats, "mean_score"):
                    # Include only numeric metrics for visualization
                    display_metrics[f"Cluster_{cluster_id}"] = {
                        "mean_score": float(stats.mean_score),
                        "score_std": float(stats.score_std),
                        "sample_size": int(stats.sample_size),
                    }

            if plot_results:
                self._create_evaluation_plots(
                    transformed_data,
                    predictions,
                    aligned_returns,
                    valid_indices,
                    display_metrics,
                )

            return metrics

        except Exception as e:
            logging.error(f"Error during evaluation: {e!s}")
            raise

    def save(self, path: Path) -> None:
        """
        Save model state to disk.

        Args:
            path: Path to save the model
        """
        try:
            # Get evaluator state
            evaluator_state = {
                "type": type(self.evaluator).__name__,
                "params": self.evaluator.__dict__,
                "cluster_stats": self.evaluator.cluster_stats,
                "selected_long": self.evaluator.selected_long,
                "selected_short": self.evaluator.selected_short,
            }

            # Create model state
            model_state = {
                "config": self.config,
                "evaluator_state": evaluator_state,
                "preprocessor_state": {
                    "data_min": self.preprocessor._data_min,
                    "data_max": self.preprocessor._data_max,
                },
                "core_state": {
                    "cluster_model": self.core.cluster_model,
                    "reducer": self.core.reducer,
                },
                "training_data": {
                    "original": self._training_data,
                    "processed": self._processed_training_data,
                    "labels": self._training_labels,
                },
            }

            # Save state
            self.model_manager.save_model(model_state, path)

        except Exception as e:
            logging.error(f"Error saving model: {e!s}")
            raise

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Miner":
        """
        Load a saved model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Miner: Loaded model instance

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        try:
            # Load model state
            model_manager = ModelManager()
            model_state = model_manager.load_model(path)

            # Create new instance with saved config
            instance = cls(model_state["config"])

            # Restore preprocessor state
            instance.preprocessor._data_min = model_state["preprocessor_state"][
                "data_min"
            ]
            instance.preprocessor._data_max = model_state["preprocessor_state"][
                "data_max"
            ]

            # Restore core state
            instance.core.cluster_model = model_state["core_state"]["cluster_model"]
            instance.core.reducer = model_state["core_state"]["reducer"]

            # Restore training data
            instance._training_data = model_state["training_data"]["original"]
            instance._processed_training_data = model_state["training_data"][
                "processed"
            ]
            instance._training_labels = model_state["training_data"]["labels"]

            # Restore evaluator state
            evaluator_state = model_state["evaluator_state"]
            instance.evaluator = instance._init_evaluator(instance.config)
            instance.evaluator.__dict__.update(evaluator_state["params"])
            instance.evaluator.cluster_stats = evaluator_state["cluster_stats"]
            instance.evaluator.selected_long = evaluator_state["selected_long"]
            instance.evaluator.selected_short = evaluator_state["selected_short"]

            return instance

        except Exception as e:
            logging.error(f"Error loading model: {e!s}")
            raise

    def get_training_data(self, processed: bool = False) -> Optional[np.ndarray]:
        """Retrieve the original training data."""
        if processed:
            return self._processed_training_data
        return self._training_data

    def _create_evaluation_plots(
        self,
        data: np.ndarray,
        predictions: np.ndarray,
        returns: pd.Series,
        valid_indices: np.ndarray,
        metrics: Dict[str, float],
    ) -> None:
        """
        Create evaluation plots using the visualizer.

        Args:
            data: Transformed data
            predictions: Model predictions
            returns: Returns series
            valid_indices: Valid indices for signals
            metrics: Performance metrics
        """
        # Create performance dashboard
        self.visualizer.create_performance_dashboard(
            returns.values, predictions, metrics
        )

        # Format cluster metrics for plotting
        cluster_metrics = {}
        for cluster_id in range(self.config["n_clusters"]):
            if cluster_id in self.evaluator.cluster_stats:
                stats = self.evaluator.cluster_stats[cluster_id]
                cluster_metrics[cluster_id] = {
                    "mean_score": float(stats.mean_score),
                    "std_dev": float(stats.score_std),
                    "sample_size": int(stats.sample_size),
                }
                # Add confidence interval if available
                if hasattr(stats, "confidence_interval"):
                    cluster_metrics[cluster_id]["conf_lower"] = float(
                        stats.confidence_interval[0]
                    )
                    cluster_metrics[cluster_id]["conf_upper"] = float(
                        stats.confidence_interval[1]
                    )

        if cluster_metrics:
            self.visualizer.plot_cluster_performance(
                cluster_metrics,
                metrics=[
                    "mean_score",
                    "std_dev",
                    "sample_size",
                ],  # Specify metrics to plot
            )

    def _init_evaluator(self, config: Dict[str, Any]):
        """
        Initialize appropriate evaluator based on configuration.

        Args:
            config: Validated configuration dictionary

        Returns:
            BaseClusterEvaluator: Initialized evaluator instance
        """
        evaluator_type = config.get("evaluator_type", EvaluatorType.HOLDING_PERIOD)
        evaluator_params = config.get("evaluator_params", {})

        if evaluator_type == EvaluatorType.HOLDING_PERIOD:
            return HoldingPeriodEvaluator(
                n_clusters=config["n_clusters"],
                hold_period=config["hold_period"],
                min_samples=evaluator_params.get("min_samples", 50),
                baseline_threshold=evaluator_params.get("baseline_threshold", 0.0),
            )
        else:  # VECTOR_PATTERN
            return VectorPatternEvaluator(
                n_clusters=config["n_clusters"],
                n_future_vectors=evaluator_params.get("n_future_vectors", 5),
                n_features=evaluator_params.get("n_features", 8),
                min_samples=evaluator_params.get("min_samples", 50),
                confidence_level=evaluator_params.get("confidence_level", 0.95),
                min_movement=evaluator_params.get("min_movement", 0.001),
            )

    @property
    def labels(self) -> Optional[np.ndarray]:
        """Retrieve the training labels."""
        return self._training_labels
