"""
General Utilities

Common utility functions used across the system.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import yaml

T = TypeVar("T")


# =============================================================================
# Random Seed Management
# =============================================================================


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# =============================================================================
# Configuration Loading
# =============================================================================


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(config: dict[str, Any], path: str | Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Data dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data dictionary
        path: Output path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


# =============================================================================
# DataFrame I/O
# =============================================================================


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save DataFrame to Parquet file.

    Args:
        df: DataFrame to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load DataFrame from Parquet file.

    Args:
        path: Path to Parquet file

    Returns:
        Loaded DataFrame
    """
    return pd.read_parquet(path)


def save_pickle(obj: Any, path: str | Path) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    """
    Load object from pickle file.

    Args:
        path: Path to pickle file

    Returns:
        Loaded object
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# DataFrame Utilities
# =============================================================================


def ensure_datetime_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Ensure date column is datetime type.

    Args:
        df: Input DataFrame
        date_col: Name of date column

    Returns:
        DataFrame with datetime date column
    """
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def filter_date_range(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Filter DataFrame by date range.

    Args:
        df: Input DataFrame
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        date_col: Name of date column

    Returns:
        Filtered DataFrame
    """
    df = ensure_datetime_index(df, date_col)

    if start_date is not None:
        start_date = pd.Timestamp(start_date)
        df = df[df[date_col] >= start_date]

    if end_date is not None:
        end_date = pd.Timestamp(end_date)
        df = df[df[date_col] <= end_date]

    return df


def pivot_to_wide(
    df: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Pivot long-format DataFrame to wide format.

    Args:
        df: Long-format DataFrame
        date_col: Date column name
        asset_col: Asset column name
        value_col: Value column name

    Returns:
        Wide-format DataFrame with dates as index and assets as columns
    """
    return df.pivot(index=date_col, columns=asset_col, values=value_col)


def melt_to_long(
    df: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Melt wide-format DataFrame to long format.

    Args:
        df: Wide-format DataFrame with dates as index
        date_col: Output date column name
        asset_col: Output asset column name
        value_col: Output value column name

    Returns:
        Long-format DataFrame
    """
    df = df.reset_index()
    return df.melt(
        id_vars=[df.columns[0]],
        var_name=asset_col,
        value_name=value_col,
    ).rename(columns={df.columns[0]: date_col})


# =============================================================================
# Numeric Utilities
# =============================================================================


def winsorize(
    data: np.ndarray | pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> np.ndarray | pd.Series:
    """
    Winsorize data by clipping extreme values.

    Args:
        data: Input data
        lower_pct: Lower percentile (0-1)
        upper_pct: Upper percentile (0-1)

    Returns:
        Winsorized data
    """
    lower = np.nanpercentile(data, lower_pct * 100)
    upper = np.nanpercentile(data, upper_pct * 100)
    return np.clip(data, lower, upper)


def zscore(
    data: np.ndarray | pd.Series,
    ddof: int = 1,
) -> np.ndarray | pd.Series:
    """
    Calculate z-score of data.

    Args:
        data: Input data
        ddof: Delta degrees of freedom for std calculation

    Returns:
        Z-scored data
    """
    mean = np.nanmean(data)
    std = np.nanstd(data, ddof=ddof)

    if std == 0 or np.isnan(std):
        return np.zeros_like(data)

    return (data - mean) / std


def rank_percentile(data: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """
    Calculate percentile rank of data.

    Args:
        data: Input data

    Returns:
        Percentile ranks (0-1)
    """
    if isinstance(data, pd.Series):
        return data.rank(pct=True, na_option="keep")
    else:
        # Handle numpy array
        ranked = np.empty_like(data, dtype=float)
        valid_mask = ~np.isnan(data)
        if valid_mask.sum() > 0:
            ranks = np.argsort(np.argsort(data[valid_mask]))
            ranked[valid_mask] = ranks / (len(ranks) - 1) if len(ranks) > 1 else 0.5
        ranked[~valid_mask] = np.nan
        return ranked


def safe_divide(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
    fill_value: float = 0.0,
) -> np.ndarray | float:
    """
    Safe division that handles division by zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        fill_value: Value to use when denominator is zero

    Returns:
        Result of division
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(numerator, denominator)
        if isinstance(result, np.ndarray):
            result[~np.isfinite(result)] = fill_value
        elif not np.isfinite(result):
            result = fill_value
    return result


# =============================================================================
# Hash Utilities
# =============================================================================


def hash_dict(d: dict[str, Any]) -> str:
    """
    Create a hash of a dictionary.

    Args:
        d: Dictionary to hash

    Returns:
        Hash string
    """
    json_str = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(json_str.encode()).hexdigest()[:12]


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Create a hash of a DataFrame.

    Args:
        df: DataFrame to hash

    Returns:
        Hash string
    """
    return hashlib.md5(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:12]


# =============================================================================
# Path Utilities
# =============================================================================


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure directory exists.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get project root directory.

    Returns:
        Path to project root
    """
    current = Path(__file__).resolve()
    # Go up until we find pyproject.toml
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent.parent


def get_data_dir() -> Path:
    """Get data directory."""
    return get_project_root() / "data"


def get_config_dir() -> Path:
    """Get config directory."""
    return get_project_root() / "configs"


def get_artifacts_dir() -> Path:
    """Get artifacts directory."""
    return get_data_dir() / "artifacts"


def get_reports_dir() -> Path:
    """Get reports directory."""
    return get_data_dir() / "reports"
