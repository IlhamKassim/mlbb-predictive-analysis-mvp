"""
recommendation_system.py
========================

This module implements simple heuristic functions for generating draft
recommendations in Mobile Legends matches.  The aim is not to replace
in‑depth strategy coaches but to provide data‑driven suggestions based
on historical win rates of heroes.

Given a preprocessed dataset with hero pick columns and a win column,
``compute_hero_win_rates`` aggregates the win rate of each hero across all
matches.  ``recommend_heroes_to_pick`` then returns a list of heroes
sorted by descending win rate.  ``recommend_heroes_to_ban`` highlights
heroes that opponents might use effectively and therefore merit banning.

These functions are intentionally simplistic.  They ignore context such
as team composition, meta shifts or counter‑picks.  For an MVP they
provide a starting point; future iterations could employ more nuanced
recommenders (e.g. collaborative filtering, reinforcement learning or
graph‑based analysis of hero matchups).
"""

from __future__ import annotations

from typing import Dict, List
import pandas as pd


def compute_hero_win_rates(
    df: pd.DataFrame,
    hero_columns: List[str],
    win_column: str = "win",
) -> Dict[str, float]:
    """Compute win rates for each hero across the specified columns.

    Parameters
    ----------
    df: pd.DataFrame
        Preprocessed DataFrame containing hero picks and a win indicator.
    hero_columns: list of str
        Column names in ``df`` representing hero selections (e.g. pick1,
        pick2, ban1).  These columns should contain hero names in lower
        case.  Non‑string values (e.g. NaN) are ignored.
    win_column: str
        Name of the binary column indicating match outcome (1 for win,
        0 for loss).

    Returns
    -------
    dict
        Mapping from hero name to their overall win rate across the
        dataset.  If a hero appears multiple times across different
        pick columns, their win rate aggregates all appearances.
    """
    win_rates: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    wins: Dict[str, int] = {}
    for col in hero_columns:
        if col not in df.columns:
            continue
        for hero, outcome in zip(df[col], df[win_column]):
            # Skip non‑string or unknown heroes
            if not isinstance(hero, str) or hero == "unknown" or hero == "":
                continue
            counts[hero] = counts.get(hero, 0) + 1
            wins[hero] = wins.get(hero, 0) + int(outcome)
    for hero in counts:
        if counts[hero] > 0:
            win_rates[hero] = wins.get(hero, 0) / counts[hero]
    return win_rates


def recommend_heroes_to_pick(win_rates: Dict[str, float], top_n: int = 5) -> List[str]:
    """Return a list of heroes with the highest win rates.

    Parameters
    ----------
    win_rates: dict
        Mapping from hero name to win rate.
    top_n: int, optional
        Number of heroes to return.  Defaults to 5.

    Returns
    -------
    list of str
        Hero names sorted by decreasing win rate.  If there are fewer
        than ``top_n`` heroes in ``win_rates``, the full list is returned.
    """
    sorted_heroes = sorted(win_rates.items(), key=lambda x: (-x[1], x[0]))
    return [hero for hero, _ in sorted_heroes[:top_n]]


def recommend_heroes_to_ban(
    win_rates: Dict[str, float],
    banned: List[str] | None = None,
    top_n: int = 5
) -> List[str]:
    """Recommend heroes to ban based on high opponent win rates.

    Parameters
    ----------
    win_rates: dict
        Mapping from hero name to win rate.  Heroes with higher win
        rates are stronger candidates for banning.
    banned: list of str or None
        Heroes already banned by the user/team.  These will be excluded
        from the recommendations.
    top_n: int, optional
        Number of heroes to recommend for banning.

    Returns
    -------
    list of str
        Recommended heroes to ban, excluding any in ``banned``.  Sorted by
        descending win rate.
    """
    banned = banned or []
    candidates = [(hero, rate) for hero, rate in win_rates.items() if hero not in banned]
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return [hero for hero, _ in candidates[:top_n]]
