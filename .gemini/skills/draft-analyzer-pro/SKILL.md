---
name: draft-analyzer-pro
description: Professional-grade MLBB draft analysis and strategy. Use for advanced synergy modeling, counter-pick identification, and competitive esports drafting strategies.
---

# Draft Analyzer Pro

This skill guides the development and execution of an enterprise-level MLBB draft analyzer.

## Professional Workflows

### 1. Advanced Synergy Discovery
To identify "Wombat Combos" or "Power Duos":
- Calculate the conditional win rate of Hero A when played with Hero B.
- Use a co-occurrence matrix on the `data/mlbb_data.csv` to identify statistically significant pairings.

### 2. Precise Counter-Picking
To suggest counters for a specific enemy pick:
- Filter the dataset for matches where the Target Hero was on the losing side.
- Identify which heroes were on the winning side in those specific matches.
- Sort by "Differential Win Rate" (Win rate vs. Target Hero vs. Global Win Rate).

### 3. Lane Balancing (Radar Chart)
Evaluate the draft across 5 axes:
- **Damage**: Burst and Sustained potential.
- **Durability**: Tankiness and sustain.
- **Control**: Hard and soft CC (Crowd Control).
- **Mobility**: Rotation speed and escape.
- **Utility**: Buffs, heals, and vision.

## Implementation Standards

- **Models**: Prefer Gradient Boosting (XGBoost/LightGBM) for professional-tier accuracy.
- **Validation**: Use a 10-fold cross-validation on the 14k+ tournament dataset.
- **Verification**: Every recommendation must include a "Confidence Score" and a link to the supporting historical match data.
