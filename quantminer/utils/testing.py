import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..classes.evaluators import EvaluatorType
from ..classes.helpers import ReducerMVR
from ..pipminer import Miner

# Ignore runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class HoldingPeriodConfig:
    """Configuration for HoldingPeriodEvaluator."""

    min_samples: int = 50
    baseline_threshold: float = 0.0

    def validate(self):
        """Validate configuration parameters."""
        if self.min_samples < 1:
            raise ValueError("min_samples must be positive")
        if not 0 <= self.baseline_threshold <= 1:
            raise ValueError("baseline_threshold must be between 0 and 1")


@dataclass
class VectorPatternConfig:
    """Configuration for VectorPatternEvaluator."""

    n_future_vectors: int = 5
    n_features: int = 8
    min_samples: int = 50
    confidence_level: float = 0.95
    min_movement: float = 0.001

    def validate(self):
        """Validate configuration parameters."""
        if self.n_future_vectors < 1:
            raise ValueError("n_future_vectors must be positive")
        if not 1 <= self.n_features <= 8:
            raise ValueError("n_features must be between 1 and 8")
        if self.min_samples < 1:
            raise ValueError("min_samples must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if not 0 < self.min_movement < 1:
            raise ValueError("min_movement must be between 0 and 1")


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and process mining configuration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Dict[str, Any]: Validated configuration with defaults

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    # Required base parameters
    required_params = {
        "n_lookback": (int, lambda x: x > 0),
        "n_pivots": (int, lambda x: x > 0),
        "n_clusters": (int, lambda x: x > 1),
        "hold_period": (int, lambda x: x >= 0),
        "model_type": (str, lambda x: x in ["standard", "ts", "sequential"]),
        "reducer_type": (
            str,
            lambda x: x in ["FFT", "PIP", "Wavelet", "FFTWavelet", "MVR"],
        ),
    }

    # Optional parameters
    optional_params = {
        "wavelet": (str, lambda x: x in ["db1", "db2", "coif1", "haar", "sym5"]),
        "verbose": (bool, lambda x: isinstance(x, bool)),
        "normalize_window": (int, lambda x: x > 0),
        "min_movement": (float, lambda x: 0 < x < 1),
        "feature_selection": (
            list,
            lambda x: all(f in ReducerMVR.FEATURE_NAMES for f in x),
        ),
        "trend_window": (int, lambda x: x > 0),
        "mvr_alpha": (float, lambda x: 0 < x < 1),
    }

    # Validate required parameters
    validated_config = {}
    for param, (param_type, validator) in required_params.items():
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
        if not isinstance(config[param], param_type):
            raise ValueError(f"Invalid type for {param}: expected {param_type}")
        if not validator(config[param]):
            raise ValueError(f"Invalid value for {param}: {config[param]}")
        validated_config[param] = config[param]

    # Validate optional parameters if present
    for param, (param_type, validator) in optional_params.items():
        if param in config:
            if not isinstance(config[param], param_type):
                raise ValueError(f"Invalid type for {param}: expected {param_type}")
            if not validator(config[param]):
                raise ValueError(f"Invalid value for {param}: {config[param]}")
            validated_config[param] = config[param]

    # Process evaluator configuration
    evaluator_type = config.get("evaluator_type", EvaluatorType.HOLDING_PERIOD.value)
    evaluator_params = config.get("evaluator_params", {})

    try:
        evaluator_type = EvaluatorType(evaluator_type)
    except ValueError:
        raise ValueError(f"Invalid evaluator_type: {evaluator_type}")

    # Validate evaluator-specific parameters
    if evaluator_type == EvaluatorType.HOLDING_PERIOD:
        evaluator_config = HoldingPeriodConfig(**evaluator_params)
    else:  # VECTOR_PATTERN
        evaluator_config = VectorPatternConfig(**evaluator_params)

    evaluator_config.validate()

    # Add evaluator configuration to validated config
    validated_config["evaluator_type"] = evaluator_type
    validated_config["evaluator_params"] = evaluator_config.__dict__

    return validated_config


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
        config: Configuration dictionary

    Returns:
        Tuple[Miner, Dict]: Trained miner instance and test metrics
    """
    # Setup logging
    setup_logging(True, "mining_test.log")

    try:
        # Validate configuration
        validated_config = validate_config(config)

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
            train_returns = returns_series.iloc[:split_idx].copy()
            test_returns = returns_series.iloc[split_idx:].copy()
        else:
            train_returns = None
            test_returns = None

        # Initialize miner with validated config
        miner = Miner(validated_config)

        # Train model
        log_progress("Starting model training")
        train_performance = miner.fit(train_data, returns_series=train_returns)
        log_progress(f"Training completed with performance: {train_performance:.4f}")

        # Evaluate on test data
        log_progress("Starting model evaluation")
        test_metrics = miner.evaluate(
            test_data, returns_series=test_returns, plot_results=True
        )

        # Print test metrics with proper formatting
        print("\nTest Performance Metrics:")
        for metric, value in test_metrics.items():
            if hasattr(value, "mean_score"):
                # Handle ClusterStats objects
                print(f"{metric}:")
                print(f"  Mean Score: {value.mean_score:.4f}")
                print(f"  Score Std: {value.score_std:.4f}")
                print(f"  Sample Size: {value.sample_size}")
                if hasattr(value, "confidence_interval"):
                    print(
                        f"  Confidence Interval: ({value.confidence_interval[0]:.4f}, {value.confidence_interval[1]:.4f})"
                    )
            elif isinstance(value, (int, float)):
                # Handle numeric values
                print(f"{metric}: {value:.4f}")
            else:
                # Handle other types
                print(f"{metric}: {value}")

        # Save the model
        miner.save(Path("trained_model.pkl"))

        return miner, test_metrics

    except Exception as e:
        log_progress("Error during testing", "ERROR", e)
        raise


# Example usage
if __name__ == "__main__":
    # Example configuration with new evaluator parameters
    config = {
        "n_lookback": 25,
        "n_pivots": 4,
        "n_clusters": 8,
        "hold_period": 6,
        "model_type": "standard",
        "reducer_type": "PIP",
        "evaluator_type": "holding_period",  # or "vector_pattern"
        "evaluator_params": {"min_samples": 50, "baseline_threshold": 0.1},
    }

    # Load OHLC data
    ohlc_data = pd.read_csv("your_data.csv")

    # Run the test
    miner, metrics = run_mining_test(ohlc_data, config)

    # Example of loading and using saved model
    loaded_miner = Miner.load(Path("trained_model.pkl"))

    # Generate signals for new data
    new_data = ohlc_data["close"].values[-100:]  # Last 100 bars
    predictions = loaded_miner.predict(new_data)

    # Create visualization
    loaded_miner.visualizer.plot_equity_curve(
        returns=np.diff(new_data), title="Recent Performance"
    )
