# PRD: MLBB Enterprise Draft Analyzer (Pro-Tier)

## 1. Overview
Elevate the current MVP into a professional-grade drafting tool for esports teams. The system moves from simple win-rate heuristics to a deep-feature analysis of synergies, counters, and player-specific performance.

## 2. Core Features (Enterprise-Grade)

### A. Advanced Synergy Modeling
- **Co-occurrence Matrix:** Identify "Power Duos" and "Wombat Combos" (e.g., Tigreal + Pharsa).
- **Synergy Score:** Calculate a real-time score based on how well the currently selected heroes work together.

### B. Precision Counter-Pick Engine
- **Direct Counters:** Use historical data to identify which heroes consistently beat a specific enemy pick.
- **Role-Based Suggetions:** Ensure counters are suggested for the correct lane (e.g., Exp Lane counter vs. Roam counter).

### C. Player-Specific Intelligence (Future)
- **Profile Integration:** Map draft suggestions to specific player hero pools and comfort levels.

### D. Tactical Dashboard (UI)
- **Win Probability Chart:** Show how win probability shifts with every pick/ban.
- **Lane Balance Visualization:** 5-point radar chart for Damage, Durability, Control, Mobility, and Utility.

## 3. Technical Requirements
- **Data:** 14k+ professional match records (Already Integrated).
- **Feature Engineering:** Delimiter-aware one-hot encoding for pick/ban slots.
- **Model:** Transition from simple RF to Gradient Boosting (XGBoost/LightGBM) for higher accuracy.

## 4. Success Metrics
- **Model Accuracy:** > 60% on professional test sets.
- **Utility:** Ability to identify "Draft Win" scenarios with high confidence.
