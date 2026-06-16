import pandas as pd
import pytest
from data_preprocessing import _standardize_string_columns, preprocess_dataframe

def test_standardize_string_columns_behavior():
    """
    Test that _standardize_string_columns converts all string columns to lowercase 
    and strips surrounding whitespace.
    """
    # GIVEN: A dataframe with inconsistent casing and whitespace
    data = {
        "hero": [" Fanny ", "Grock", " rafaela"],
        "team": ["TEAM A", "team b ", " Team C "],
        "kills": [10, 5, 8]  # Numeric column should remain untouched
    }
    df = pd.DataFrame(data)
    
    # WHEN: We standardize the columns
    standardized_df = _standardize_string_columns(df)
    
    # THEN: String columns are lowercased and stripped
    expected_heroes = ["fanny", "grock", "rafaela"]
    expected_teams = ["team a", "team b", "team c"]
    
    assert standardized_df["hero"].tolist() == expected_heroes
    assert standardized_df["team"].tolist() == expected_teams
    # AND: Numeric columns are unchanged
    assert standardized_df["kills"].tolist() == [10, 5, 8]
    assert standardized_df["kills"].dtype.kind in "i" # integer type

def test_preprocess_dataframe_kda_calculation():
    """
    Test that KDA is calculated correctly as (kills + assists) / max(1, deaths).
    """
    # GIVEN: A dataframe with K, D, A columns
    data = {
        "K": [10, 5, 0],
        "D": [2, 0, 5],
        "A": [5, 5, 5]
    }
    df = pd.DataFrame(data)
    
    # WHEN: We preprocess the dataframe
    processed_df = preprocess_dataframe(df)
    
    # THEN: KDA is calculated correctly
    # Row 0: (10 + 5) / 2 = 7.5
    # Row 1: (5 + 5) / max(1, 0) = 10.0 (Edge case: zero deaths)
    # Row 2: (0 + 5) / 5 = 1.0
    expected_kda = [7.5, 10.0, 1.0]
    assert processed_df["kda"].tolist() == expected_kda
