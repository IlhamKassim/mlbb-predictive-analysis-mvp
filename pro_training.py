"""
pro_training.py
===============

Enterprise-tier training script using XGBoost and Symmetric Matchup 
modeling to maximize draft win probability accuracy.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_matchup_data(data_path="data/mlbb_data.csv"):
    df = pd.read_csv(data_path)
    hero_picks = [f"hero_pick{i}" for i in range(1, 6)]
    
    # Get all unique heroes for features
    all_heroes = set()
    for col in hero_picks:
        all_heroes.update(df[col].dropna().unique())
    all_heroes.discard('none')
    all_heroes = sorted(list(all_heroes))
    hero_to_idx = {h: i for i, h in enumerate(all_heroes)}
    
    matchup_rows = []
    labels = []
    weights = []
    
    print(f"Creating weighted symmetric matchup vectors for {len(df)//2} matches...")
    
    total_matches = len(df) // 2
    for i in range(0, len(df) - 1, 2):
        t1 = df.iloc[i]
        t2 = df.iloc[i+1]
        
        # Team 1 Feature Vector
        v1 = np.zeros(len(all_heroes))
        for p in hero_picks:
            if t1[p] in hero_to_idx: v1[hero_to_idx[t1[p]]] = 1
            
        # Team 2 Feature Vector
        v2 = np.zeros(len(all_heroes))
        for p in hero_picks:
            if t2[p] in hero_to_idx: v2[hero_to_idx[t2[p]]] = 1
            
        # Meta Weight: Linear decay (Most recent matches have weight 1.0, oldest 0.1)
        # Matches are sorted oldest to newest in the dataset
        match_idx = i // 2
        weight = 0.1 + (0.9 * (match_idx / total_matches))
        
        matchup_rows.append(v1 - v2)
        labels.append(int(t1['win']))
        weights.append(weight)
        
        matchup_rows.append(v2 - v1)
        labels.append(int(t2['win']))
        weights.append(weight)
        
    X = np.array(matchup_rows)
    y = np.array(labels)
    w = np.array(weights)
    return X, y, w, all_heroes

def train_pro_model():
    X, y, w, heroes = prepare_matchup_data()
    
    # Stratified split to maintain balance
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )
    
    print("Training Meta-Aware XGBoost Pro Model...")
    clf = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # Train with sample weights
    clf.fit(X_train, y_train, sample_weight=w_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"🚀 Pro-Model Matchup Accuracy: {acc:.2%}")
    
    # Save model and hero list (required for inference)
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/classifier_pro.joblib")
    joblib.dump(heroes, "models/hero_list.joblib")
    print("Model saved to models/classifier_pro.joblib")
    return acc

if __name__ == "__main__":
    train_pro_model()
