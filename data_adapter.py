"""
data_adapter.py
===============

Custom adapter for the LiTianYeoh MLBB tournament dataset.
Transforms match-level data into training-ready format.
"""

import pandas as pd
import ast
import argparse
import os

def parse_tuple_string(s):
    try:
        # Convert "('Hero1', 'Hero2')" to ['hero1', 'hero2']
        items = ast.literal_eval(s)
        return [str(i).strip().lower() for i in items]
    except:
        return []

def adapt_tournament_data(input_path: str, output_path: str):
    print(f"Loading external data from {input_path}...")
    df = pd.read_csv(input_path)
    
    adapted_rows = []
    
    print("Processing match records...")
    for _, row in df.iterrows():
        # Parse picks and bans
        t1_picks = parse_tuple_string(row['t1_picks'])
        t1_bans = parse_tuple_string(row['t1_bans'])
        t2_picks = parse_tuple_string(row['t2_picks'])
        t2_bans = parse_tuple_string(row['t2_bans'])
        
        # Team 1 perspective
        if len(t1_picks) == 5:
            entry1 = {
                'win': int(float(row['t1_result'])),
                'hero_pick1': t1_picks[0],
                'hero_pick2': t1_picks[1],
                'hero_pick3': t1_picks[2],
                'hero_pick4': t1_picks[3],
                'hero_pick5': t1_picks[4],
            }
            # Add bans (up to 5)
            for i in range(min(5, len(t1_bans))):
                entry1[f'ban{i+1}'] = t1_bans[i]
            
            # Fill missing bans if any
            for i in range(len(t1_bans), 5):
                entry1[f'ban{i+1}'] = 'none'
                
            adapted_rows.append(entry1)
            
        # Team 2 perspective
        if len(t2_picks) == 5:
            entry2 = {
                'win': int(float(row['t2_result'])),
                'hero_pick1': t2_picks[0],
                'hero_pick2': t2_picks[1],
                'hero_pick3': t2_picks[2],
                'hero_pick4': t2_picks[3],
                'hero_pick5': t2_picks[4],
            }
            # Add bans (up to 5)
            for i in range(min(5, len(t2_bans))):
                entry2[f'ban{i+1}'] = t2_bans[i]
            
            # Fill missing bans if any
            for i in range(len(t2_bans), 5):
                entry2[f'ban{i+1}'] = 'none'
                
            adapted_rows.append(entry2)

    new_df = pd.DataFrame(adapted_rows)
    print(f"Saving {len(new_df)} training rows to {output_path}...")
    new_df.to_csv(output_path, index=False)
    print("Success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt external MLBB data.")
    parser.add_argument("--input", default="external_data.csv", help="Path to external CSV")
    parser.add_argument("--output", default="data/mlbb_data_expanded.csv", help="Output path")
    
    args = parser.parse_args()
    if os.path.exists(args.input):
        adapt_tournament_data(args.input, args.output)
    else:
        print(f"Error: Input file {args.input} not found.")
