import pandas as pd
import numpy as np
import pytest
from app import map_user_input_to_features

def test_map_user_input_to_features_consistency():
    """
    Test that map_user_input_to_features correctly sets flags for selected heroes
    and maintains the correct feature order.
    """
    # GIVEN: A set of mock feature names from a trained model
    feature_names = np.array([
        "hero_pick1_fanny", 
        "hero_pick1_grock", 
        "hero_pick2_rafaela", 
        "hero_pick2_fanny",
        "gold",
        "kills"
    ])
    
    # Mock classifier object that has feature_names_in_
    class MockClf:
        feature_names_in_ = feature_names
    
    clf = MockClf()
    
    # WHEN: User selects fanny and rafaela
    team_heroes = ["fanny", "rafaela"]
    enemy_heroes = []
    banned_heroes = []
    
    input_row = map_user_input_to_features(clf, feature_names, team_heroes, enemy_heroes, banned_heroes)
    
    # THEN: The columns matching 'fanny' and 'rafaela' should be 1, others 0
    assert input_row.at[0, "hero_pick1_fanny"] == 1
    assert input_row.at[0, "hero_pick2_fanny"] == 1
    assert input_row.at[0, "hero_pick2_rafaela"] == 1
    assert input_row.at[0, "hero_pick1_grock"] == 0
    assert input_row.at[0, "gold"] == 0
    assert input_row.at[0, "kills"] == 0
    
    # AND: The order of columns must match feature_names exactly
    assert list(input_row.columns) == list(feature_names)

def test_map_user_input_to_features_substring_edge_case():
    """
    Test that 'ling' does not accidentally match 'sling' or 'ling_suffix' 
    if we want precise matching.
    """
    feature_names = np.array(["hero_pick1_ling", "hero_pick1_sling"])
    
    class MockClf:
        feature_names_in_ = feature_names
    clf = MockClf()
    
    # WHEN: User selects 'ling'
    team_heroes = ["ling"]
    input_row = map_user_input_to_features(clf, feature_names, team_heroes, [], [])
    
    # THEN: Only 'hero_pick1_ling' should be 1
    assert input_row.at[0, "hero_pick1_ling"] == 1
    # This might fail currently if it just does 'ling in col'
    assert input_row.at[0, "hero_pick1_sling"] == 0
