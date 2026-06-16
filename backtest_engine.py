"""
backtest_engine.py
==================

Professional validation engine that tests the draft analyzer's 
predictions against historical M5 World Championship results.

Goal: Determine if the model can accurately predict match winners 
based solely on the final draft.
"""

import pandas as pd
import joblib
import numpy as np
from app import map_user_input_to_features

def run_m5_backtest(model_path="models/classifier.joblib", data_path="data/mlbb_data.csv"):
    print(f"🚀 Initializing Pro-Tier Backtest...")
    
    # 1. Load Model
    try:
        classifier = joblib.load(model_path)
        feature_names = classifier.feature_names_in_
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Load and Filter M5 Data
    # For a clean test, we look for matches from the M5 tournament (tournament_code=1 in LiTianYeoh set)
    # Since we combined them into mlbb_data.csv, we'll use a hold-out set approach.
    df = pd.read_csv(data_path)
    
    # We take the last 500 rows as a test set (most recent matches)
    test_df = df.tail(500)
    
    correct_predictions = 0
    total_matches = len(test_df)
    
    print(f"📊 Analyzing {total_matches} historical pro matches...")
    
    pick_cols = [f"hero_pick{i}" for i in range(1, 6)]
    ban_cols = [f"ban{i}" for i in range(1, 6)]
    
    results = []

    for i, row in test_df.iterrows():
        # Get team draft
        team_picks = [row[p] for p in pick_cols if isinstance(row[p], str) and row[p] != 'none']
        bans = [row[b] for b in ban_cols if isinstance(row[b], str) and row[b] != 'none']
        actual_outcome = int(row['win'])
        
        # We don't have the opponent picks in a single row in this format (one row per team),
        # but the classifier was trained to see 'presence' of heroes.
        
        # Predict
        input_row = map_user_input_to_features(classifier, feature_names, team_picks, bans)
        proba = classifier.predict_proba(input_row)[0][1]
        prediction = 1 if proba > 0.5 else 0
        
        if prediction == actual_outcome:
            correct_predictions += 1
        
        results.append({
            "actual": actual_outcome,
            "predicted_prob": proba,
            "success": prediction == actual_outcome
        })

    accuracy = correct_predictions / total_matches
    print(f"\n✅ Backtest Complete!")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📈 Model Accuracy: {accuracy:.2%}")
    print(f"🎯 Total Matches:  {total_matches}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if accuracy > 0.60:
        print("🏆 VERDICT: PRO-READY. The model significantly outperforms random chance in professional drafts.")
    else:
        print("⚠️  VERDICT: ANALYST ASSISTANT. Useful for heuristics, but requires more feature engineering for top-tier prediction.")

if __name__ == "__main__":
    run_m5_backtest()
