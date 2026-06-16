"""
synergy_engine.py
=================

Enterprise-grade engine for calculating hero synergies and counter-pick 
probabilities using professional match data. 

Outputs:
- synergy_matrix.json: Conditional win rates for hero pairs.
- counter_matrix.json: Differential win rates for hero vs hero matchups.
"""

import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict

def calculate_advanced_metrics(data_path="data/mlbb_data.csv", output_dir="data"):
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # We need to reconstruct matches to see who played against whom
    # Our adapted data has one row per team. Pairs of rows are matches.
    # Row 0 vs Row 1, Row 2 vs Row 3, etc.
    
    hero_picks = [f"hero_pick{i}" for i in range(1, 6)]
    
    synergy_counts = defaultdict(lambda: {"wins": 0, "total": 0})
    counter_counts = defaultdict(lambda: {"wins": 0, "total": 0})
    global_hero_stats = defaultdict(lambda: {"wins": 0, "total": 0})
    
    print("Analyzing 28k professional rows for synergies and counters...")
    
    # Pre-calculate global stats for normalization
    for _, row in df.iterrows():
        team_heroes = [row[p] for p in hero_picks if isinstance(row[p], str) and row[p] != 'none']
        win = int(row['win'])
        for h in team_heroes:
            global_hero_stats[h]["total"] += 1
            global_hero_stats[h]["wins"] += win

    # Calculate Synergy (Within same team)
    for _, row in df.iterrows():
        team_heroes = [row[p] for p in hero_picks if isinstance(row[p], str) and row[p] != 'none']
        win = int(row['win'])
        
        # Pairs within the team
        for i in range(len(team_heroes)):
            for j in range(i + 1, len(team_heroes)):
                h1, h2 = sorted([team_heroes[i], team_heroes[j]])
                pair = f"{h1}|{h2}"
                synergy_counts[pair]["total"] += 1
                synergy_counts[pair]["wins"] += win

    # Calculate Counters (Between opposing teams)
    # We iterate by 2 to get Matchups (Team A vs Team B)
    for i in range(0, len(df) - 1, 2):
        row_a = df.iloc[i]
        row_b = df.iloc[i+1]
        
        heroes_a = [row_a[p] for p in hero_picks if isinstance(row_a[p], str) and row_a[p] != 'none']
        heroes_b = [row_b[p] for p in hero_picks if isinstance(row_b[p], str) and row_b[p] != 'none']
        win_a = int(row_a['win'])
        win_b = int(row_b['win'])
        
        for ha in heroes_a:
            for hb in heroes_b:
                # Team A perspective ha vs hb
                matchup_ab = f"{ha}|{hb}"
                counter_counts[matchup_ab]["total"] += 1
                counter_counts[matchup_ab]["wins"] += win_a
                
                # Team B perspective hb vs ha
                matchup_ba = f"{hb}|{ha}"
                counter_counts[matchup_ba]["total"] += 1
                counter_counts[matchup_ba]["wins"] += win_b

    # Finalize Synergy Matrix
    # We look for "Lift" (How much better is the pair than the individual win rates?)
    synergy_matrix = {}
    for pair, stats in synergy_counts.items():
        if stats["total"] > 10: # Significance threshold
            h1, h2 = pair.split("|")
            win_rate = stats["wins"] / stats["total"]
            h1_wr = global_hero_stats[h1]["wins"] / global_hero_stats[h1]["total"]
            h2_wr = global_hero_stats[h2]["wins"] / global_hero_stats[h2]["total"]
            avg_wr = (h1_wr + h2_wr) / 2
            synergy_matrix[pair] = {
                "win_rate": round(win_rate, 4),
                "lift": round(win_rate - avg_wr, 4),
                "sample": stats["total"]
            }

    # Finalize Counter Matrix
    counter_matrix = {}
    for matchup, stats in counter_counts.items():
        if stats["total"] > 10:
            ha, hb = matchup.split("|")
            win_rate = stats["wins"] / stats["total"]
            ha_wr = global_hero_stats[ha]["wins"] / global_hero_stats[ha]["total"]
            counter_matrix[matchup] = {
                "win_rate": round(win_rate, 4),
                "diff": round(win_rate - ha_wr, 4), # Positive means ha counters hb
                "sample": stats["total"]
            }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "synergy_matrix.json"), "w") as f:
        json.dump(synergy_matrix, f, indent=2)
    with open(os.path.join(output_dir, "counter_matrix.json"), "w") as f:
        json.dump(counter_matrix, f, indent=2)
    
    print(f"Analysis complete! Matrices saved to {output_dir}/")

if __name__ == "__main__":
    calculate_advanced_metrics()
