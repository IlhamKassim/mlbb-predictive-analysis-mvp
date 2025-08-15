"""
model_training.py
=================

This module encapsulates the logic for training machine‑learning models on
preprocessed Mobile Legends data.  It provides convenience functions
to train a classification model to predict match outcomes and optional
regression models to estimate player‑level statistics such as kills,
assists and gold.

The default classifier is a ``RandomForestClassifier`` because of its
robustness and ability to handle high‑dimensional, mixed‑type data.  You
can swap it out for other algorithms (e.g. logistic regression,
gradient boosting) by modifying the code below.  The regression models
use ``RandomForestRegressor`` by default.

All trained models can be saved to disk via ``save_models``.  The
module expects that the caller has already preprocessed the data and
split it into features and labels (see ``data_preprocessing.py``).
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error


def train_classification_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a classification model to predict match outcomes.

    Parameters
    ----------
    X: pd.DataFrame
        Feature matrix.
    y: pd.Series
        Target vector (1 for win, 0 for loss).
    n_estimators: int, optional
        Number of trees in the random forest.
    max_depth: int or None, optional
        Maximum depth of each tree.  ``None`` means nodes are expanded until
        all leaves are pure or until all leaves contain less than
        ``min_samples_split`` samples.
    random_state: int, optional
        Seed for reproducible results.

    Returns
    -------
    RandomForestClassifier
        The trained classifier.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X, y)
    return clf


def train_regression_models(
    X: pd.DataFrame,
    df_original: pd.DataFrame,
    target_columns: List[str] | None = None,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> Dict[str, RandomForestRegressor]:
    """Train one regression model per specified target column.

    Parameters
    ----------
    X: pd.DataFrame
        Feature matrix used for training.
    df_original: pd.DataFrame
        The original (unencoded) DataFrame; this is used to retrieve
        continuous target variables.  It must have the same row order as
        ``X``.
    target_columns: list of str or None
        The names of columns in ``df_original`` to predict.  If ``None``,
        the function attempts to automatically detect common columns
        (``kills``, ``assists``, ``gold``).
    n_estimators: int, optional
        Number of trees in each random forest.
    max_depth: int or None, optional
        Maximum depth of trees.
    random_state: int, optional
        Random seed.

    Returns
    -------
    dict
        A mapping from target column name to its trained regression model.
    """
    # Auto‑detect target columns if none provided
    if target_columns is None:
        possible = ["kills", "assists", "gold", "k", "a", "d", "kda", "gold_per_min"]
        target_columns = [c for c in df_original.columns if c.lower() in possible]
    models: Dict[str, RandomForestRegressor] = {}
    for target in target_columns:
        # Skip non‑numeric targets
        if not np.issubdtype(df_original[target].dtype, np.number):
            continue
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        reg.fit(X, df_original[target])
        models[target] = reg
    return models


def evaluate_classifier(clf: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    """Compute classification accuracy on a hold‑out split.

    This helper function performs a simple train/test split internally
    before evaluating the accuracy.  It is intended for quick sanity
    checks rather than rigorous validation.

    Parameters
    ----------
    clf: RandomForestClassifier
        The classifier to evaluate.
    X: pd.DataFrame
        Feature matrix.
    y: pd.Series
        Target vector.

    Returns
    -------
    float
        Accuracy score on the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


def evaluate_regressors(models: Dict[str, RandomForestRegressor], X: pd.DataFrame, df_original: pd.DataFrame) -> Dict[str, float]:
    """Compute mean absolute error for each regression model.

    Uses a simple train/test split for each target variable.  The caller
    can decide whether to rely on these scores for model selection.

    Parameters
    ----------
    models: dict
        A mapping from target column name to ``RandomForestRegressor``.
    X: pd.DataFrame
        Feature matrix.
    df_original: pd.DataFrame
        Original data containing target variables.

    Returns
    -------
    dict
        A mapping from target name to mean absolute error on the hold‑out test set.
    """
    scores: Dict[str, float] = {}
    for target, model in models.items():
        if target not in df_original.columns:
            continue
        y = df_original[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores[target] = mean_absolute_error(y_test, preds)
    return scores


def save_models(
    classifier: RandomForestClassifier,
    regressors: Dict[str, RandomForestRegressor],
    directory: str = "models"
) -> None:
    """Persist models to disk using joblib.

    Parameters
    ----------
    classifier: RandomForestClassifier
        The classification model to save.
    regressors: dict
        Mapping from target column name to trained regression model.
    directory: str
        Directory in which to save model files.  Will be created if it
        does not exist.
    """
    os.makedirs(directory, exist_ok=True)
    # Save classifier
    joblib.dump(classifier, os.path.join(directory, "classifier.joblib"))
    # Save each regressor under its target name
    for target, model in regressors.items():
        filename = f"regressor_{target}.joblib"
        joblib.dump(model, os.path.join(directory, filename))
