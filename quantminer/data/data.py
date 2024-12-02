import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

load_dotenv()


def convert_timezone(
    df: pd.DataFrame, from_tz: str = "UTC", to_tz: str = "UTC"
) -> pd.DataFrame:
    """Convert DataFrame index from one timezone to another."""
    df.index = df.index.tz_localize(from_tz)
    df.index = df.index.tz_convert(to_tz)
    return df.tz_localize(None)  # Remove timezone info after conversion


def remove_outliers(
    data: np.ndarray,
    lower_percentile: float = 15.0,
    upper_percentile: Optional[float] = None,
) -> np.ndarray:
    """Remove outliers from data based on the interquartile range (IQR)."""
    data = np.asarray(data)

    if upper_percentile is None:
        upper_percentile = 100 - lower_percentile

    q1 = np.percentile(data, lower_percentile)
    q3 = np.percentile(data, upper_percentile)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Replace outliers with the nearest boundary value
    return np.clip(data, lower_bound, upper_bound)


def round_to_timeframe(df: pd.DataFrame, pandas_freq: str) -> pd.DataFrame:
    """Round timestamps to the nearest timeframe interval."""
    # Round index to the specified frequency
    df.index = df.index.round(pandas_freq)

    # Handle duplicate indices by keeping the last occurrence
    return df[~df.index.duplicated(keep="last")]


def loads_data(
    symbol: str,
    ctf: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    handle_outliers: bool = False,
    subpath: Optional[str] = None,
    buffer: int = 0,
) -> pd.DataFrame:
    """Load and preprocess market data for a given symbol and timeframe."""
    subpaths = ["daily", "crypto", "forex", "futures", "misc"]
    datapath_env = os.getenv("DATA_PATH")
    if not datapath_env:
        raise ValueError("DATA_PATH environment variable not set.")
    datapath = Path(datapath_env)
    if not datapath.is_dir():
        raise ValueError(f"Data path not found: {datapath}")

    if subpath:
        subpaths = [subpath]

    filepath = None
    for sub in subpaths:
        candidate_path = datapath / sub / f"{symbol.upper()}.parquet"
        if candidate_path.exists():
            filepath = candidate_path
            break

    if filepath is None:
        raise FileNotFoundError(f"No file found for symbol: {symbol}")

    print(f"Found path: {filepath}")
    raw_df = pd.read_parquet(filepath)
    raw_df.reset_index(inplace=True)

    required_columns = ["time", "open", "high", "low", "close", "volume"]
    if not all(col in raw_df.columns for col in required_columns):
        raise ValueError(f"Data file must contain columns: {required_columns}")

    raw_df = raw_df[required_columns]
    raw_df["time"] = pd.to_datetime(raw_df["time"], errors="coerce")
    raw_df.set_index("time", inplace=True)
    raw_df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    if raw_df["volume"].mode()[0] == 0:
        raw_df["volume"] = ta.atr(
            raw_df["high"], raw_df["low"], raw_df["close"], length=1, talib=True
        )

    if ctf != "1min":
        df = resample_data(
            raw_df,
            current_timeframe="1min",
            target_timeframe=ctf,
            buffer=buffer,
        )
    else:
        df = raw_df

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index < pd.to_datetime(end_date)]

    returns = df["close"].pct_change().shift(-1).fillna(0)
    if handle_outliers:
        returns = remove_outliers(returns, 0.0001)
    df["returns"] = returns

    return df


def load_data(
    symbol: str,
    ctf: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    handle_outliers: bool = False,
    subpath: Optional[str] = None,
    buffer: int = 0,
    broker_timezone: str = "UTC",  # Add timezone parameter
) -> pd.DataFrame:
    """
    Load and preprocess market data for a given symbol and timeframe.

    Parameters:
        symbol: Trading symbol
        ctf: Target timeframe
        start_date: Start date for data filtering
        end_date: End date for data filtering
        handle_outliers: Whether to remove outliers
        subpath: Specific data directory to look in
        buffer: Number of periods to add for look-ahead prevention
        broker_timezone: Timezone of the broker's server (e.g., 'UTC', 'US/Eastern', 'Europe/London')
    """
    subpaths = ["daily", "crypto", "forex", "futures", "misc", "MT5"]
    datapath_env = os.getenv("DATA_PATH")
    if not datapath_env:
        raise ValueError("DATA_PATH environment variable not set.")
    datapath = Path(datapath_env)
    if not datapath.is_dir():
        raise ValueError(f"Data path not found: {datapath}")

    if subpath:
        subpaths = [subpath]

    filepath = None
    for sub in subpaths:
        candidate_path = datapath / sub / f"{symbol.upper()}.parquet"
        if candidate_path.exists():
            filepath = candidate_path
            break

    if filepath is None:
        raise FileNotFoundError(f"No file found for symbol: {symbol}")

    print(f"Found path: {filepath}")
    raw_df = pd.read_parquet(filepath)
    raw_df.reset_index(inplace=True)

    required_columns = ["time", "open", "high", "low", "close", "volume"]
    if not all(col in raw_df.columns for col in required_columns):
        raise ValueError(f"Data file must contain columns: {required_columns}")

    raw_df = raw_df[required_columns]
    raw_df["time"] = pd.to_datetime(raw_df["time"], errors="coerce")
    raw_df.set_index("time", inplace=True)
    raw_df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # Convert timezone from UTC to broker's timezone
    if broker_timezone != "UTC":
        raw_df = convert_timezone(raw_df, from_tz="UTC", to_tz=broker_timezone)

    if raw_df["volume"].mode()[0] == 0:
        raw_df["volume"] = ta.atr(
            raw_df["high"], raw_df["low"], raw_df["close"], length=1, talib=True
        )

    # Round timestamps before resampling
    raw_df = round_to_timeframe(raw_df, "1min")

    if ctf != "1min":
        df = resample_data(
            raw_df,
            current_timeframe="1min",
            target_timeframe=ctf,
            buffer=buffer,
        )
    else:
        df = raw_df

    # Round final timestamps to the target timeframe
    df = round_to_timeframe(df, ctf)

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index < pd.to_datetime(end_date)]

    returns = df["close"].pct_change().shift(-1).fillna(0)
    if handle_outliers:
        returns = remove_outliers(returns, 0.0001)
    df["returns"] = returns

    return df


def resample_data(
    df: pd.DataFrame,
    current_timeframe: str,
    target_timeframe: str,
    buffer: int = 0,
) -> pd.DataFrame:
    """Resample the DataFrame to a specified target timeframe with an optional buffer."""
    required_columns = {"open", "high", "low", "close", "volume"}

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    df = df.asfreq(current_timeframe)
    current_freq = pd.to_timedelta(current_timeframe)
    target_freq = pd.to_timedelta(target_timeframe)

    if target_freq <= current_freq:
        raise ValueError(
            "Target timeframe must be a longer period than the current timeframe"
        )

    shift_amount = buffer * current_freq
    resampled = (
        df.resample(target_timeframe, origin="start", offset=shift_amount)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    return resampled


def align_dataframes(
    lower_df: pd.DataFrame,
    upper_series: Union[pd.DataFrame, pd.Series],
    lower_timeframe: str,
    upper_timeframe: str,
) -> pd.DataFrame:
    """Align a higher frequency DataFrame with a lower frequency DataFrame or Series."""
    if not isinstance(lower_df.index, pd.DatetimeIndex) or not isinstance(
        upper_series.index, pd.DatetimeIndex
    ):
        raise ValueError("DataFrames must be indexed by DatetimeIndex")

    if upper_series.empty:
        raise ValueError("upper_series cannot be empty")

    if isinstance(upper_series, pd.Series):
        upper_series = upper_series.to_frame()

    lower_freq = pd.to_timedelta(lower_timeframe)
    upper_freq = pd.to_timedelta(upper_timeframe)

    if lower_freq >= upper_freq:
        raise ValueError("lower_df must have a higher frequency than upper_series")

    resampled_upper = upper_series.resample(lower_timeframe).ffill().shift(1)
    aligned_upper = resampled_upper.reindex(lower_df.index, method="ffill")
    result_df = pd.concat([lower_df, aligned_upper], axis=1)

    return result_df


# ALIASES
upsample_data = resample_data
downsample_data = align_dataframes

if __name__ == "__main__":
    # Example usage
    symbol = "AAPL"
    ctf = "5T"  # 5-minute timeframe
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    data = load_data(symbol, ctf, start_date, end_date, handle_outliers=True)
    print(data.head())
