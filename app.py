"""
app.py
======

This file defines a Streamlit application for the Mobile Legends
predictive analytics MVP.  The app allows users to select heroes for
their team and bans, then displays a predicted win probability,
expected player performance metrics and simple draft recommendations.

To run the application locally, install the required dependencies
(`pip install -r requirements.txt`), ensure the dataset and trained
models exist in the specified locations, and then execute:

```
streamlit run mlbb_mvp/app.py
```

The app expects:

* A dataset file at ``data/mlbb_data.csv`` or similar.  This is used
  only to populate the list of available heroes and to compute hero
  win rates for draft recommendations.
* A trained classifier saved at ``models/classifier.joblib``.
* One or more regression models saved as ``models/regressor_<target>.joblib``.

If these files are missing, the app will still run but will
degrade gracefully by omitting predictions or recommendations.

Note: Because this is an MVP, the feature mapping from user input to
the model's input vector is naive: it simply looks for feature names
containing the hero's name.  For production use, you should align
exactly with the encoding performed during preprocessing (e.g. by
saving the feature list alongside the model).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from data_preprocessing import load_dataset, preprocess_dataframe
from recommendation_system import compute_hero_win_rates, recommend_heroes_to_pick, recommend_heroes_to_ban


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

@st.cache_data
def load_data_and_compute_win_rates(file_path: str, win_column: str = "win") -> tuple[list[str], dict]:
    """Load a dataset and compute hero win rates.

    Returns a sorted list of unique heroes across all hero columns and a
    dictionary mapping hero to win rate.  If the dataset cannot be
    loaded, returns empty structures.
    """
    try:
        df_raw = load_dataset(file_path)
    except FileNotFoundError:
        return [], {}
    df = df_raw.copy()
    # Standardize hero columns for computing win rates
    hero_cols = [c for c in df.columns if "hero" in c.lower() or "pick" in c.lower() or "ban" in c.lower()]
    # Lowercase hero names for consistency
    for col in hero_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    # Compute win rates if win_column exists
    if win_column not in df.columns:
        win_rates = {}
    else:
        win_rates = compute_hero_win_rates(df, hero_cols, win_column=win_column)
    # Gather list of heroes
    heroes = set()
    for col in hero_cols:
        heroes.update(df[col].dropna().unique().tolist())
    heroes.discard("unknown")
    heroes.discard("nan")
    heroes = [h for h in heroes if isinstance(h, str) and h]
    heroes = sorted(set(heroes))
    return heroes, win_rates


def map_user_input_to_features(
    clf, feature_names: np.ndarray, team_heroes: list[str], banned_heroes: list[str]
) -> pd.DataFrame:
    """Create a single‑row DataFrame matching the classifier's input format.

    This function initializes all features to zero and then sets to 1
    those dummy variables whose names contain one of the heroes selected by the
    user.  It is a simple heuristic that relies on the naming
    convention produced by ``pd.get_dummies`` in the preprocessing step.  If
    your dataset uses different prefixes (e.g. ``pick1_fanny`` vs
    ``hero_fanny``) you may need to adjust the matching logic.

    Parameters
    ----------
    clf: trained classifier
        The classification model.  Must have the attribute
        ``feature_names_in_``.
    feature_names: array-like
        List or array of feature names corresponding to the model input.
    team_heroes: list of str
        Heroes selected for the user's team.  Should already be lowercased.
    banned_heroes: list of str
        Heroes that have been banned.  Should already be lowercased.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a single row and columns matching
        ``feature_names``.
    """
    # Initialize row with zeros
    row = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    # For each feature, set to 1 if its name contains a hero in picks or bans
    for col in feature_names:
        for hero in team_heroes + banned_heroes:
            if hero in col:
                row.at[0, col] = 1
    return row


def load_trained_models(models_dir: str = "models"):
    """Attempt to load classifier and regressors from disk.

    Returns (classifier, regressors_dict).  If files are missing, returns
    ``None`` for the missing entries.  Classifier is expected at
    ``classifier.joblib``; regressors are any files beginning with
    ``regressor_``.
    """
    classifier_path = os.path.join(models_dir, "classifier.joblib")
    classifier = None
    regressors = {}
    if os.path.exists(classifier_path):
        try:
            classifier = joblib.load(classifier_path)
        except Exception:
            classifier = None
    # Load regressors
    if os.path.isdir(models_dir):
        for fname in os.listdir(models_dir):
            if fname.startswith("regressor_") and fname.endswith(".joblib"):
                target = fname[len("regressor_"):-len(".joblib")]
                try:
                    regressors[target] = joblib.load(os.path.join(models_dir, fname))
                except Exception:
                    continue
    return classifier, regressors


def main() -> None:
    st.set_page_config(page_title="MLBB Match Predictor", layout="centered")
    st.title("Mobile Legends Match Outcome Predictor")
    st.write("This MVP predicts match outcomes and suggests draft strategies based on historical Mobile Legends data.")

    # Load data for hero list and win rates
    heroes, win_rates = load_data_and_compute_win_rates(os.path.join("data", "mlbb_data.csv"), win_column="win")

    # Load models
    classifier, regressors = load_trained_models("models")

    # Sidebar for user selections
    st.sidebar.header("Draft Input")
    if heroes:
        team_heroes = st.sidebar.multiselect(
            "Select your team's heroes (max 5)", options=heroes, default=[]
        )
        banned_heroes = st.sidebar.multiselect(
            "Select banned heroes", options=heroes, default=[]
        )
    else:
        st.sidebar.warning("Dataset not found.  Please place your data at data/mlbb_data.csv to enable hero selection.")
        team_heroes = []
        banned_heroes = []

    if st.sidebar.button("Predict"):
        # Predict win probability
        if classifier is not None:
            try:
                feature_names = classifier.feature_names_in_
                input_row = map_user_input_to_features(classifier, feature_names, team_heroes, banned_heroes)
                proba = classifier.predict_proba(input_row)[0][1]
                st.subheader("Predicted Win Probability")
                st.write(f"Your team's predicted probability of winning: **{proba:.2%}**")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Classifier model not found.  Please train the model and place it under models/classifier.joblib.")

        # Player performance predictions
        if regressors and classifier is not None:
            st.subheader("Expected Player Performance")
            input_row = map_user_input_to_features(classifier, classifier.feature_names_in_, team_heroes, banned_heroes)
            perf_rows = []
            for target, reg in regressors.items():
                try:
                    val = reg.predict(input_row)[0]
                    perf_rows.append((target, val))
                except Exception:
                    continue
            if perf_rows:
                perf_df = pd.DataFrame(perf_rows, columns=["metric", "expected_value"])
                st.dataframe(perf_df)
            else:
                st.write("No regression models available or unable to generate predictions.")
        else:
            st.info("Regression models not found.  Only win probability will be displayed.")

        # Recommendations
        if win_rates:
            st.subheader("Draft Recommendations")
            # Filter out heroes already picked or banned
            available = [h for h in win_rates.keys() if h not in team_heroes + banned_heroes]
            available_rates = {h: win_rates[h] for h in available}
            top_picks = recommend_heroes_to_pick(available_rates, top_n=5)
            st.markdown("**Top Heroes to Pick:**")
            if top_picks:
                st.write(", ".join(top_picks))
            else:
                st.write("No hero recommendations available.")
            top_bans = recommend_heroes_to_ban(win_rates, banned=banned_heroes + team_heroes, top_n=5)
            st.markdown("**Top Heroes to Ban:**")
            if top_bans:
                st.write(", ".join(top_bans))
            else:
                st.write("No ban recommendations available.")
        else:
            st.info("Hero win rate data unavailable.  Please provide a dataset to enable draft recommendations.")

    # Display hero win rate table
    if win_rates:
        if st.checkbox("Show Hero Win Rate Table"):
            win_rate_df = pd.DataFrame(
                sorted(win_rates.items(), key=lambda x: -x[1]), columns=["hero", "win_rate"]
            )
            st.dataframe(win_rate_df)


if __name__ == "__main__":
    main()
