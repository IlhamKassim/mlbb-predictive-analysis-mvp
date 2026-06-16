import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import requests
import json
import plotly.express as px
from collections import defaultdict

from data_preprocessing import load_dataset
from recommendation_system import compute_hero_win_rates
from ui_constants import ROLE_ATTRIBUTES, DEFAULT_ATTRS, CSS_STYLE


# -----------------------------------------------------------------------------
# Configuration & Assets
# -----------------------------------------------------------------------------

HERO_DATA_URL = "https://gist.githubusercontent.com/vsec7/73c6dedea092fca8e0a94d448000226c/raw/mlbb-hero.json"

@st.cache_data
def load_hero_metadata():
    try:
        response = requests.get(HERO_DATA_URL)
        data = response.json()
        return {h["hero_name"].lower(): h for h in data}
    except Exception:
        return {}

@st.cache_data
def load_pro_intelligence():
    try:
        with open("data/synergy_matrix.json", "r") as f:
            synergy = json.load(f)
        with open("data/counter_matrix.json", "r") as f:
            counter = json.load(f)
        return synergy, counter
    except Exception:
        return {}, {}

def display_hero_card(hero_name: str, meta: dict, subtitle: str = ""):
    hero_info = meta.get(hero_name.lower(), {})
    portrait = hero_info.get("portrait")
    hero_class = hero_info.get("class", "Unknown")
    
    st.markdown(f"""
        <div class="hero-container">
            <div style="display: flex; align-items: center; gap: 15px;">
                <img src="{portrait}" width="50" style="border-radius: 5px; border: 1px solid #1c2c4c;">
                <div>
                    <div style="font-weight: bold; font-size: 1.1rem;">{hero_name.title()}</div>
                    <div style="font-size: 0.8rem; color: #8892b0;">{hero_class} | {subtitle}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Pro Analysis Functions
# -----------------------------------------------------------------------------

def get_radar_data(team_heroes: list[str], meta: dict):
    stats = defaultdict(list)
    for h in team_heroes:
        hero_info = meta.get(h.lower(), {})
        role = hero_info.get("class")
        attrs = ROLE_ATTRIBUTES.get(role, DEFAULT_ATTRS)
        for k, v in attrs.items():
            stats[k].append(v)
    
    if not stats:
        return pd.DataFrame(columns=["theta", "r"])
    
    avg_stats = {k: np.mean(v) for k, v in stats.items()}
    df = pd.DataFrame([
        dict(r=v, theta=k) for k, v in avg_stats.items()
    ])
    # Close the radar loop
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df

def calculate_draft_synergy(team_heroes: list[str], synergy_matrix: dict) -> float:
    if len(team_heroes) < 2: return 0.0
    lifts = []
    for i in range(len(team_heroes)):
        for j in range(i + 1, len(team_heroes)):
            h1, h2 = sorted([team_heroes[i].lower(), team_heroes[j].lower()])
            pair = f"{h1}|{h2}"
            if pair in synergy_matrix:
                lifts.append(synergy_matrix[pair]["lift"])
    return np.mean(lifts) if lifts else 0.0

def get_strategic_counters(enemy_heroes: list[str], counter_matrix: dict, top_n: int = 5) -> list[tuple[str, float]]:
    if not enemy_heroes: return []
    scores = defaultdict(float)
    for eh in enemy_heroes:
        for matchup, stats in counter_matrix.items():
            ha, hb = matchup.split("|")
            if hb == eh.lower():
                scores[ha] += stats["diff"]
    final_scores = [(h, scores[h] / len(enemy_heroes)) for h in scores]
    final_scores.sort(key=lambda x: -x[1])
    return final_scores[:top_n]

# -----------------------------------------------------------------------------
# App Logic
# -----------------------------------------------------------------------------

@st.cache_data
def load_data_and_compute_win_rates(file_path: str):
    try:
        df_raw = load_dataset(file_path)
    except FileNotFoundError:
        return [], {}
    hero_cols = [c for c in df_raw.columns if any(x in c.lower() for x in ["hero", "pick", "ban"])]
    for col in hero_cols:
        df_raw[col] = df_raw[col].astype(str).str.strip().str.lower()
    win_rates = compute_hero_win_rates(df_raw, hero_cols)
    heroes = sorted(list(set(df_raw[hero_cols].values.flatten())))
    for x in ["unknown", "nan", "none"]:
        if x in heroes: heroes.remove(x)
    return heroes, win_rates

def map_user_input_to_features(clf, feature_names, team_heroes, banned_heroes):
    row = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    selected = set(team_heroes + banned_heroes)
    for col in feature_names:
        parts = col.split("_")
        if any(h in parts for h in selected):
            row.at[0, col] = 1
    return row

def main():
    st.set_page_config(page_title="PRO War Room | MLBB", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS_STYLE, unsafe_allow_html=True)

    # Load Intelligence
    hero_meta = load_hero_metadata()
    heroes, win_rates = load_data_and_compute_win_rates("data/mlbb_data.csv")
    synergy_matrix, counter_matrix = load_pro_intelligence()
    classifier = joblib.load("models/classifier.joblib") if os.path.exists("models/classifier.joblib") else None

    # --- Header ---
    st.title("🛰️ PRO War Room")
    st.caption("Strategic Draft Intelligence Engine v2.0 | Enterprise Tier")
    st.divider()

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚡ Live Draft")
        team_heroes = st.multiselect("YOUR SQUAD", options=heroes)
        enemy_heroes = st.multiselect("ENEMY SQUAD", options=heroes)
        banned_heroes = st.multiselect("BANS", options=heroes)
        st.divider()
        analyze = st.button("RUN ANALYSIS", use_container_width=True)

    # --- Main Dashboard ---
    col_squad, col_analysis = st.columns([1, 2])

    with col_squad:
        st.subheader("🔵 SQUAD")
        if not team_heroes:
            st.info("Select heroes to begin.")
        for h in team_heroes:
            display_hero_card(h, hero_meta)
        
        if enemy_heroes:
            st.divider()
            st.subheader("🔴 ENEMY")
            for h in enemy_heroes:
                display_hero_card(h, hero_meta)

    with col_analysis:
        tab_intel, tab_stats = st.tabs(["🎯 STRATEGic INTEL", "📊 DATA INSIGHTS"])
        
        with tab_intel:
            # Metrics
            m1, m2, m3 = st.columns(3)
            
            # Prediction
            win_prob = 0.5
            if classifier and team_heroes:
                feature_names = classifier.feature_names_in_
                input_row = map_user_input_to_features(classifier, feature_names, team_heroes, banned_heroes + enemy_heroes)
                win_prob = classifier.predict_proba(input_row)[0][1]
            
            m1.metric("WIN PROB", f"{win_prob:.1%}")
            
            # Synergy
            synergy_lift = calculate_draft_synergy(team_heroes, synergy_matrix)
            m2.metric("SYNERGY LIFT", f"{synergy_lift:+.1%}")
            
            # Team Balance
            balance_score = 0.0 # Logic could be added
            m3.metric("COMP BALANCE", "PRO")

            st.divider()

            # Radar Chart
            st.subheader("🕸️ Composition Balance")
            radar_df = get_radar_data(team_heroes, hero_meta)
            if not radar_df.empty:
                fig = px.line_polar(radar_df, r='r', theta='theta', line_close=True, range_r=[0, 5], template="plotly_dark")
                fig.update_traces(fill='toself', line_color='#00d4ff', fillcolor='rgba(0, 212, 255, 0.3)')
                fig.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=False, margin=dict(l=40, r=40, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Add team picks to visualize composition balance.")

            # Priority Counters
            if enemy_heroes:
                st.divider()
                st.subheader("🔥 PRIORITY COUNTERS")
                counters = get_strategic_counters(enemy_heroes, counter_matrix)
                for h, score in counters:
                    if h not in team_heroes + banned_heroes + enemy_heroes:
                        display_hero_card(h, hero_meta, subtitle=f"Counter Strength: {score:+.1%}")

        with tab_stats:
            if win_rates:
                st.subheader("📈 Global Performance")
                wr_df = pd.DataFrame(sorted(win_rates.items(), key=lambda x: -x[1]), columns=["Hero", "Win Rate"])
                wr_df["Win Rate"] = wr_df["Win Rate"].map(lambda x: f"{x:.2%}")
                st.dataframe(wr_df, use_container_width=True, height=600)

if __name__ == "__main__":
    main()
