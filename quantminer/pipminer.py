import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .classes.helpers import (
    ClusterEvaluator,
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
                - cluster_selection_mode: Mode for selecting clusters
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

        self.evaluator = ClusterEvaluator(
            selection_mode=config["cluster_selection_mode"],
            hold_period=config["hold_period"],
            verbose=self.verbose,
        )

        self.tester = ModelTester(
            self.preprocessor,
            self.core,
            self.evaluator,
            hold_period=config["hold_period"],
            verbose=self.verbose,
        )

        self.visualizer = Visualizer(verbose=self.verbose)
        self.model_manager = ModelManager(verbose=self.verbose)

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

        Raises:
            ValueError: If data validation fails
        """
        if price_data is None:
            price_data = np.copy(data)

        try:
            # Preprocess data
            processed_data = self.preprocessor.preprocess_data(data)
            processed_price = self.preprocessor.preprocess_data(price_data)

            # Use provided returns_series if available, otherwise calculate returns
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

            # Get cluster labels
            labels = self.core.labels_

            # Assess clusters and select best performers
            self.evaluator.assess_clusters(labels, returns, self.config["n_clusters"])

            # Calculate performance
            performance = self.evaluator.compute_performance(labels, returns)

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

        # Test model
        martin_ratio, metrics = self.tester.test_model(
            data, price_data, returns_series=returns_series, plot_equity=plot_results
        )

        if plot_results:
            # Get predictions and returns for visualization
            predictions = self.predict(data)
            if returns_series is not None:
                returns = returns_series.values
            else:
                returns = np.diff(price_data, prepend=price_data[0])

            # Create visualizations
            # Convert metrics to per-cluster format for visualization
            cluster_metrics = {}
            for i in range(self.config["n_clusters"]):
                cluster_metrics[i] = {
                    "martin_ratio": metrics.get("martin_ratio", 0),
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                    "profit_factor": metrics.get("profit_factor", 0),
                }

            self.visualizer.plot_cluster_performance(cluster_metrics)
            self.visualizer.create_performance_dashboard(returns, predictions, metrics)

        return metrics

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model

        Raises:
            IOError: If save operation fails
        """
        try:
            self.model_manager.save_model(self, path)
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
            model_state = ModelManager().load_model(path)

            # Recreate model with saved configuration
            model = cls(model_state["model_params"])

            # Restore component states
            model.preprocessor._data_min = model_state["preprocessing_params"][
                "data_min"
            ]
            model.preprocessor._data_max = model_state["preprocessing_params"][
                "data_max"
            ]

            model.evaluator.cluster_labels_long = model_state["cluster_labels"]["long"]
            model.evaluator.cluster_labels_short = model_state["cluster_labels"][
                "short"
            ]

            model.core.reducer = model_state["model_components"]["reducer"]
            model.core.cluster_model = model_state["model_components"]["cluster_model"]

            return model

        except Exception as e:
            logging.error(f"Error loading model: {e!s}")
            raise


class ErrorType(Enum):
    """Enumeration of different error types for error handling."""

    PREPROCESSING = "preprocessing"
    CLUSTERING = "clustering"
    DATA_VALIDATION = "data_validation"
    CONFIGURATION = "configuration"


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    required_params = {
        "n_lookback": (int, lambda x: x > 0),
        "n_pivots": (int, lambda x: x > 0),
        "n_clusters": (int, lambda x: x > 1),
        "hold_period": (int, lambda x: x >= 0),
        "model_type": (str, lambda x: x in ["standard", "ts", "sequential"]),
        "reducer_type": (str, lambda x: x in ["FFT", "PIP", "Wavelet", "FFTWavelet"]),
        "cluster_selection_mode": (str, lambda x: x in ["best", "baseline"]),
    }

    optional_params = {
        "wavelet": (str, lambda x: x in ["db1", "db2", "coif1", "haar", "sym5"]),
        "verbose": (bool, lambda x: isinstance(x, bool)),
    }

    # Check required parameters
    for param, (param_type, validator) in required_params.items():
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
        if not isinstance(config[param], param_type):
            raise ValueError(f"Invalid type for {param}: expected {param_type}")
        if not validator(config[param]):
            raise ValueError(f"Invalid value for {param}: {config[param]}")

    # Check optional parameters if present
    for param, (param_type, validator) in optional_params.items():
        if param in config:
            if not isinstance(config[param], param_type):
                raise ValueError(f"Invalid type for {param}: expected {param_type}")
            if not validator(config[param]):
                raise ValueError(f"Invalid value for {param}: {config[param]}")


def validate_data_format(data: np.ndarray) -> None:
    """
    Validate input data format and structure.

    Args:
        data: Input data array to validate

    Raises:
        ValueError: If data format is invalid
    """
    if not isinstance(data, (np.ndarray, list)):
        raise ValueError("Data must be numpy array or list")

    data = np.asarray(data)

    if data.size == 0:
        raise ValueError("Empty data array")

    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Data must contain numeric values")

    if not np.all(np.isfinite(data)):
        raise ValueError("Data contains NaN or infinite values")


def setup_logging(
    verbose: bool, log_file: Optional[str] = None, level: str = "INFO"
) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: Enable verbose output
        log_file: Optional path to log file
        level: Logging level
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    if verbose:
        handlers = []

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, level.upper()), handlers=handlers, force=True
        )

        logging.info("Logging setup completed")


def log_progress(
    message: str, level: str = "INFO", error: Optional[Exception] = None
) -> None:
    """
    Log progress messages with consistent formatting.

    Args:
        message: Message to log
        level: Logging level
        error: Optional exception to include in log
    """
    log_method = getattr(logging, level.lower())

    if error:
        log_method(f"{message}: {error!s}")
    else:
        log_method(message)


def run_mining_test(ohlc_data: pd.DataFrame, config: dict):
    """
    Run complete test of the mining framework.

    Args:
        ohlc_data: DataFrame with OHLC data
    """
    # Setup logging
    setup_logging(True, "mining_test.log")

    try:
        # Use returns from the dataframe if available
        if "returns" in ohlc_data.columns:
            returns_series = ohlc_data["returns"]
        else:
            returns_series = None

        close_prices = ohlc_data["close"].values

        # Split data into train and test sets ensuring proper indexing
        split_idx = int(len(close_prices) * 0.7)
        train_data = close_prices[:split_idx]
        test_data = close_prices[split_idx:]

        if returns_series is not None:
            # Ensure returns series indices match the data splits
            train_returns = returns_series.iloc[:split_idx].copy()
            test_returns = returns_series.iloc[split_idx:].copy()
        else:
            train_returns = None
            test_returns = None

        # Initialize miner
        miner = Miner(config)

        # Train model
        log_progress("Starting model training")
        train_performance = miner.fit(train_data, returns_series=train_returns)
        log_progress(f"Training completed with performance: {train_performance:.4f}")

        # Evaluate on test data
        log_progress("\n\n\nStarting model evaluation")
        test_metrics = miner.evaluate(
            test_data, returns_series=test_returns, plot_results=True
        )

        # Print test metrics
        print("\nTest Performance Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Generate predictions for the last n_lookback bars
        min_required = max(config["n_lookback"], config["n_pivots"])
        latest_data = close_prices[-min_required:]
        latest_prediction = miner.predict(latest_data)

        print("Latest Prediction: ", latest_prediction)

        # Save the model
        miner.save(Path("trained_model.pkl"))

        return miner, test_metrics

    except Exception as e:
        log_progress("Error during testing", "ERROR", e)
        raise


# Example usage
if __name__ == "__main__":
    # Load your OHLC data
    # Example: Loading from CSV
    ohlc_data = pd.read_csv("your_data.csv")

    # Run the test
    miner, metrics = run_mining_test(ohlc_data)

    # Example of loading and using a saved model
    loaded_miner = Miner.load(Path("trained_model.pkl"))

    # Generate signals for new data
    new_data = ohlc_data["close"].values[-100:]  # Last 100 bars
    predictions = loaded_miner.predict(new_data)

    # Create custom visualization
    loaded_miner.visualizer.plot_equity_curve(
        returns=np.diff(new_data), title="Recent Performance"
    )
