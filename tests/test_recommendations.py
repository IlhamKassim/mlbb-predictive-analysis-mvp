import pandas as pd
import pytest
from recommendation_system import compute_hero_win_rates, recommend_heroes_to_pick, recommend_heroes_to_ban

def test_compute_hero_win_rates_aggregation():
    """
    Test that win rates are correctly aggregated across multiple hero columns.
    """
    # GIVEN: A dataframe with hero picks and match outcomes
    data = {
        "hero_pick1": ["grock", "fanny", "grock"],
        "hero_pick2": ["rafaela", "grock", "rafaela"],
        "win": [1, 0, 1]
    }
    df = pd.DataFrame(data)
    hero_cols = ["hero_pick1", "hero_pick2"]
    
    # WHEN: We compute win rates
    win_rates = compute_hero_win_rates(df, hero_cols, win_column="win")
    
    # THEN: Win rates are calculated correctly
    # grock: 
    #   Pick1: 1 (win), 1 (win) -> count 2, wins 2 (Row 0 and Row 2)
    #   Pick2: 0 (loss) -> count 1, wins 0 (Row 1)
    #   Total: 3 games, 2 wins = 0.666...
    # rafaela:
    #   Pick2: 1 (win), 1 (win) -> count 2, wins 2
    #   Total: 2 games, 2 wins = 1.0
    # fanny:
    #   Pick1: 0 (loss) -> count 1, wins 0
    #   Total: 1 game, 0 wins = 0.0
    
    assert win_rates["rafaela"] == 1.0
    assert win_rates["fanny"] == 0.0
    assert win_rates["grock"] == pytest.approx(0.666, abs=1e-3)

def test_recommend_heroes_to_pick_sorting():
    """
    Test that heroes are recommended in descending order of win rate.
    """
    win_rates = {
        "grock": 0.5,
        "fanny": 0.9,
        "rafaela": 0.7,
        "ling": 0.4
    }
    
    # WHEN: We ask for top 2 picks
    recommendations = recommend_heroes_to_pick(win_rates, top_n=2)
    
    # THEN: We get fanny (0.9) and rafaela (0.7)
    assert recommendations == ["fanny", "rafaela"]

def test_recommend_heroes_to_ban_filtering():
    """
    Test that banned heroes are excluded from ban recommendations.
    """
    win_rates = {
        "fanny": 0.9,
        "rafaela": 0.7,
        "grock": 0.5
    }
    
    # WHEN: fanny is already banned
    recommendations = recommend_heroes_to_ban(win_rates, banned=["fanny"], top_n=2)
    
    # THEN: fanny is excluded, we get rafaela and grock
    assert recommendations == ["rafaela", "grock"]
    assert "fanny" not in recommendations
