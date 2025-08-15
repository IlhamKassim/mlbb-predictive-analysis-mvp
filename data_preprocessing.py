"""
data_preprocessing.py
=====================

This module implements functions for loading and preprocessing the Mobile Legends
M5 World Championship dataset (or a similar dataset).  It provides basic
utilities to clean the raw CSV, engineer useful features and split the
data into features ``X`` and labels ``y`` suitable for machine‑learning models.

The functions in this module are deliberately generic.  They make minimal
assumptions about the column names in the input file so that you can adapt
them to whatever schema your dataset uses.  Where possible, the code
detects common patterns (e.g. columns named ``Kills``/``Deaths``/``Assists``
or ``Time``) and derives additional metrics such as KDA or gold per minute.

Usage:

    from data_preprocessing import load_dataset, preprocess_dataframe, split_features_labels

    df = load_dataset("/path/to/mlbb_data.csv")
    df_clean = preprocess_dataframe(df)
    X, y = split_features_labels(df_clean, label_column="win")

Feel free to extend or modify this module as you become more familiar with
the structure of your data.  For example, you might replace the simple
categorical encoding with a bespoke hero embedding or incorporate domain
knowledge about roles and item builds.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a CSV dataset from disk.

    Parameters
    ----------
    file_path: str
        Path to the CSV file containing the raw Mobile Legends match data.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the raw data.
    """
    df = pd.read_csv(file_path)
    return df


def _standardize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all string columns to lowercase and strip surrounding whitespace.

    This helper ensures that hero names, player names, team names and other
    categorical fields are normalized before encoding.  Lowercasing reduces
    redundancy (e.g. ``"Fanny"`` and ``"fanny"`` map to the same key) and
    stripping whitespace avoids accidental mismatches due to trailing spaces.

    Parameters
    ----------
    df: pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with standardized string columns.
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def _parse_time_column(series: pd.Series) -> pd.Series:
    """Parse a time column containing durations into minutes.

    The dataset sometimes stores match duration as a string in the format
    ``MM:SS`` or ``HH:MM:SS``.  This helper converts those strings into
    floating‑point minutes.  Non‑parseable values are returned unchanged.

    Parameters
    ----------
    series: pd.Series
        A pandas Series containing time strings.

    Returns
    -------
    pd.Series
        A Series of floats representing minutes, or original values if
        conversion fails.
    """
    def convert(val):
        if isinstance(val, str):
            parts = val.split(":")
            try:
                parts = [int(p) for p in parts]
            except ValueError:
                return val
            # If length is 2 -> MM:SS, if 3 -> HH:MM:SS
            if len(parts) == 2:
                minutes, seconds = parts
                return minutes + seconds / 60.0
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                return hours * 60.0 + minutes + seconds / 60.0
        return val
    return series.apply(convert)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features from the raw dataset.

    This function performs the following operations:

    * Standardizes all object (string) columns (lowercase and strip).
    * Parses a ``time`` column (if present) into minutes.
    * Computes additional performance metrics:
        - ``kda`` = (kills + assists) / max(1, deaths)
        - ``gold_per_min`` = gold earned / time_in_minutes
    * Fills missing values with zeros for numeric columns and "unknown"
      for object columns.
    * One‑hot encodes all categorical columns.

    Parameters
    ----------
    df: pd.DataFrame
        The raw dataset loaded from CSV.

    Returns
    -------
    pd.DataFrame
        A preprocessed DataFrame with engineered features and dummy variables.
    """
    # Make a copy to avoid mutating the original
    data = df.copy()

    # Standardize string columns
    data = _standardize_string_columns(data)

    # Parse a time column into minutes if present
    time_col_candidates = [c for c in data.columns if c.lower() in {"time", "duration"}]
    if time_col_candidates:
        time_col = time_col_candidates[0]
        data["time_minutes"] = _parse_time_column(data[time_col])
    else:
        data["time_minutes"] = np.nan

    # Compute KDA if kills/deaths/assists columns exist
    # Accept various common naming conventions
    kill_cols = [c for c in data.columns if c.lower() in {"k", "kills"}]
    death_cols = [c for c in data.columns if c.lower() in {"d", "deaths"}]
    assist_cols = [c for c in data.columns if c.lower() in {"a", "assists"}]
    gold_cols = [c for c in data.columns if c.lower() in {"gold", "gold_earned"}]

    if kill_cols and death_cols and assist_cols:
        kc = kill_cols[0]; dc = death_cols[0]; ac = assist_cols[0]
        deaths_replaced = data[dc].replace(0, 1)
        data["kda"] = (data[kc] + data[ac]) / deaths_replaced
    else:
        data["kda"] = np.nan

    # Compute gold per minute
    if gold_cols and not data["time_minutes"].isnull().all():
        gc = gold_cols[0]
        # Avoid division by zero
        time_minutes = data["time_minutes"].replace(0, np.nan)
        data["gold_per_min"] = data[gc] / time_minutes
    else:
        data["gold_per_min"] = np.nan

    # Fill missing numeric values with zero and missing object values with 'unknown'
    for col in data.columns:
        if data[col].dtype.kind in "biufc":  # numeric types
            data[col] = data[col].fillna(0)
        else:
            data[col] = data[col].fillna("unknown")

    # Identify label column if exists; for now we leave label in place
    # One‑hot encode categorical columns
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    return data_encoded


def split_features_labels(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a processed DataFrame into feature matrix X and label vector y.

    Parameters
    ----------
    df: pd.DataFrame
        A preprocessed DataFrame returned from ``preprocess_dataframe``.
    label_column: str
        Name of the column containing the binary win/loss label.  This
        column should exist in ``df`` and contain numeric values (0/1).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple of ``X`` (features) and ``y`` (labels).
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")
    X = df.drop(columns=[label_column])
    y = df[label_column]
    return X, y
