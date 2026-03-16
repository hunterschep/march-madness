from __future__ import annotations

"""Tournament-level helpers for coherent team ratings from pair probabilities."""

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def fit_pair_probability_ratings(
    team_ids: Iterable[int],
    pair_rows: pd.DataFrame,
    *,
    low_col: str = "team_low_id",
    high_col: str = "team_high_id",
    probability_col: str = "probability",
    weight_col: str = "weight",
) -> pd.Series:
    """Fit one latent rating per team so rating gaps match pairwise logits."""
    teams = sorted(int(team_id) for team_id in team_ids)
    team_index = {team_id: idx for idx, team_id in enumerate(teams)}

    design = np.zeros((len(pair_rows) + 1, len(teams)))
    target = np.zeros(len(pair_rows) + 1)

    for row_index, row in enumerate(pair_rows.itertuples(index=False)):
        low_id = int(getattr(row, low_col))
        high_id = int(getattr(row, high_col))
        weight = math.sqrt(float(getattr(row, weight_col)))
        design[row_index, team_index[low_id]] = weight
        design[row_index, team_index[high_id]] = -weight
        target[row_index] = _logit(getattr(row, probability_col)) * weight

    # Force the mean rating to zero for identifiability.
    design[len(pair_rows), :] = 1.0
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return pd.Series(coefficients, index=teams, name="rating")


def pair_probability_from_ratings(
    team_low_id: int,
    team_high_id: int,
    ratings: pd.Series,
) -> float:
    gap = float(ratings.loc[int(team_low_id)] - ratings.loc[int(team_high_id)])
    return 1.0 / (1.0 + math.exp(-gap))
