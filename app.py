import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

from data_preprocessing import load_dataset
from recommendation_system import compute_hero_win_rates
from ui_constants import ROLE_ATTRIBUTES, DEFAULT_ATTRS, CSS_STYLE, CLASS_TO_LANE

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

def display_hero_card(hero_name: str, meta: dict, subtitle: str = "", side="ally"):
    hero_info = meta.get(hero_name.lower(), {})
    portrait = hero_info.get("portrait", "https://via.placeholder.com/50")
    hero_class = hero_info.get("class", "Unknown")
    lane = CLASS_TO_LANE.get(hero_class, "Flex")
    
    border_color = "#00d4ff" if side == "ally" else "#ff4b4b"
    
    st.markdown(f"""
        <div class="hero-container" style="border-left: 4px solid {border_color};">
            <div style="display: flex; align-items: center; gap: 15px;">
                <img src="{portrait}" width="50" style="border-radius: 5px; border: 1px solid #1c2c4c;">
                <div>
                    <div style="font-weight: bold; font-size: 1.1rem;">{hero_name.title()}</div>
                    <div style="font-size: 0.8rem; color: #8892b0;">{lane} | {subtitle}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Pro Analysis Functions
# -----------------------------------------------------------------------------

def get_team_stats(heroes: list[str], meta: dict):
    if not heroes:
        return {k: 0 for k in ["Damage", "Durability", "Control", "Mobility", "Utility"]}
    
    stats = defaultdict(list)
    for h in heroes:
        hero_info = meta.get(h.lower(), {})
        role = hero_info.get("class")
        attrs = ROLE_ATTRIBUTES.get(role, DEFAULT_ATTRS)
        for k, v in attrs.items():
            stats[k].append(v)
    
    return {k: np.mean(v) for k, v in stats.items()}

def create_radar_chart(ally_heroes, enemy_heroes, meta):
    ally_stats = get_team_stats(ally_heroes, meta)
    enemy_stats = get_team_stats(enemy_heroes, meta)
    
    categories = list(ally_stats.keys())
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[ally_stats[c] for c in categories] + [ally_stats[categories[0]]],
        theta=categories + [categories[0]],
        fill='toself', name='YOUR SQUAD', line_color='#00d4ff', fillcolor='rgba(0, 212, 255, 0.3)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[enemy_stats[c] for c in categories] + [enemy_stats[categories[0]]],
        theta=categories + [categories[0]],
        fill='toself', name='ENEMY SQUAD', line_color='#ff4b4b', fillcolor='rgba(255, 75, 75, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], showticklabels=False, gridcolor="#1c2c4c"),
            angularaxis=dict(gridcolor="#1c2c4c", linecolor="#1c2c4c")
        ),
        showlegend=True, template="plotly_dark", margin=dict(l=80, r=80, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5)
    )
    return fig

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

def get_strategic_counters(enemy_heroes: list[str], counter_matrix: dict, team_heroes: list[str], meta: dict) -> dict:
    if not enemy_heroes: return {}
    
    # 1. Determine missing roles
    filled_lanes = set()
    for h in team_heroes:
        h_class = meta.get(h.lower(), {}).get("class", "Unknown")
        lane = CLASS_TO_LANE.get(h_class, "Flex")
        filled_lanes.add(lane)
        
    all_lanes = ["Gold Lane", "EXP Lane", "Mid Lane", "Jungle", "Roam"]
    missing_lanes = [L for L in all_lanes if L not in filled_lanes]
    if not missing_lanes: missing_lanes = all_lanes

    # 2. Score counters
    scores = defaultdict(float)
    for eh in enemy_heroes:
        for matchup, stats in counter_matrix.items():
            ha, hb = matchup.split("|")
            if hb == eh.lower():
                scores[ha] += stats["diff"]
                
    # 3. Filter by missing lane
    lane_recs = defaultdict(list)
    for h, score in scores.items():
        if score > 0 and h not in team_heroes + enemy_heroes:
            h_class = meta.get(h, {}).get("class", "Unknown")
            lane = CLASS_TO_LANE.get(h_class, "Flex")
            if lane in missing_lanes:
                lane_recs[lane].append((h, score / len(enemy_heroes)))
                
    # 4. Sort and return top 3 per lane
    final_recs = {}
    for lane, recs in lane_recs.items():
        recs.sort(key=lambda x: -x[1])
        if recs:
            final_recs[lane] = recs[:3]
        
    return final_recs

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

def map_user_input_to_features(clf, feature_names, team_heroes, enemy_heroes, banned_heroes):
    row = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    for col in feature_names:
        parts = col.split("_")
        if any(h.lower() in parts for h in team_heroes):
            row.at[0, col] = 1
        elif any(h.lower() in parts for h in enemy_heroes):
            row.at[0, col] = -1
    return row

def main():
    st.set_page_config(page_title="PRO War Room | MLBB", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS_STYLE, unsafe_allow_html=True)

    hero_meta = load_hero_metadata()
    heroes, win_rates = load_data_and_compute_win_rates("data/mlbb_data.csv")
    synergy_matrix, counter_matrix = load_pro_intelligence()
    
    pro_model = None
    if os.path.exists("models/classifier_pro.joblib"):
        pro_model = joblib.load("models/classifier_pro.joblib")
        pro_heroes = joblib.load("models/hero_list.joblib")

    # --- Header ---
    st.title("🛰️ PRO War Room")
    st.caption("Strategic Draft Intelligence Engine v3.0 | Enterprise Tier")
    st.divider()

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚡ Live Draft")
        team_heroes = st.multiselect("YOUR SQUAD", options=heroes)
        enemy_heroes = st.multiselect("ENEMY SQUAD", options=heroes)
        banned_heroes = st.multiselect("BANS", options=heroes)
        st.divider()
        st.button("RUN DEEP ANALYSIS", use_container_width=True)

    # --- Main Dashboard ---
    col_squad, col_analysis = st.columns([1, 2.2])

    with col_squad:
        st.subheader("🔵 SQUAD")
        if not team_heroes: st.info("Select allies.")
        for h in team_heroes: display_hero_card(h, hero_meta, side="ally")
        
        if enemy_heroes:
            st.divider()
            st.subheader("🔴 ENEMY")
            for h in enemy_heroes: display_hero_card(h, hero_meta, side="enemy")

    with col_analysis:
        tab_intel, tab_sim, tab_stats = st.tabs(["🎯 TACTICAL INTEL", "🔄 DRAFT SIM", "📊 DATA INSIGHTS"])
        
        with tab_intel:
            m1, m2, m3 = st.columns(3)
            win_prob = 0.5
            if pro_model and team_heroes:
                input_vec = np.zeros((1, len(pro_heroes)))
                h_to_idx = {h: i for i, h in enumerate(pro_heroes)}
                for h in team_heroes:
                    if h.lower() in h_to_idx: input_vec[0, h_to_idx[h.lower()]] += 1
                for h in enemy_heroes:
                    if h.lower() in h_to_idx: input_vec[0, h_to_idx[h.lower()]] -= 1
                win_prob = pro_model.predict_proba(input_vec)[0][1]
            
            m1.metric("WIN PROB", f"{win_prob:.1%}", delta=f"{win_prob-0.5:.1%}" if team_heroes else None)
            
            synergy_lift = calculate_draft_synergy(team_heroes, synergy_matrix)
            m2.metric("SYNERGY LIFT", f"{synergy_lift:+.1%}")
            
            enemy_threat = calculate_draft_synergy(enemy_heroes, synergy_matrix)
            m3.metric("ENEMY SYNERGY", f"{enemy_threat:+.1%}")

            st.divider()

            st.subheader("🕸️ Draft Comparison")
            fig = create_radar_chart(team_heroes, enemy_heroes, hero_meta)
            st.plotly_chart(fig, use_container_width=True, key="radar_intel")

            if enemy_heroes:
                st.divider()
                st.subheader("🔥 LANE-SPECIFIC COUNTERS")
                st.info("Recommending counters for the roles your team is currently missing.")
                
                lane_counters = get_strategic_counters(enemy_heroes, counter_matrix, team_heroes, hero_meta)
                
                if lane_counters:
                    cols = st.columns(len(lane_counters))
                    for idx, (lane, recs) in enumerate(lane_counters.items()):
                        with cols[idx]:
                            st.markdown(f"**{lane}**")
                            for h, score in recs:
                                display_hero_card(h, hero_meta, subtitle=f"Edge: {score:+.1%}", side="ally")
                else:
                    st.success("Draft Complete! No remaining lanes to fill.")

        with tab_sim:
            st.subheader("🔮 Draft Simulation")
            st.write("Simulate how adding a hero changes your squad's balance.")
            sim_hero = st.selectbox("Simulate adding hero:", options=[h for h in heroes if h not in team_heroes + enemy_heroes + banned_heroes])
            if sim_hero:
                sim_team = team_heroes + [sim_hero]
                fig_sim = create_radar_chart(sim_team, enemy_heroes, hero_meta)
                st.plotly_chart(fig_sim, use_container_width=True, key="radar_sim")

        with tab_stats:
            if win_rates:
                st.subheader("📈 Global performance")
                wr_df = pd.DataFrame(sorted(win_rates.items(), key=lambda x: -x[1]), columns=["Hero", "Win Rate"])
                wr_df["Win Rate"] = wr_df["Win Rate"].map(lambda x: f"{x:.2%}")
                st.dataframe(wr_df, use_container_width=True, height=600)

if __name__ == "__main__":
    main()
