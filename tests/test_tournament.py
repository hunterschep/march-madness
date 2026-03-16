from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.tournament import fit_pair_probability_ratings, pair_probability_from_ratings


def test_fit_pair_probability_ratings_reproduces_simple_ordering() -> None:
    pairs = pd.DataFrame(
        [
            {"team_low_id": 1, "team_high_id": 2, "probability": 0.75, "weight": 2.0},
            {"team_low_id": 1, "team_high_id": 3, "probability": 0.85, "weight": 2.0},
            {"team_low_id": 2, "team_high_id": 3, "probability": 0.70, "weight": 2.0},
        ]
    )
    ratings = fit_pair_probability_ratings([1, 2, 3], pairs)

    assert ratings.loc[1] > ratings.loc[2] > ratings.loc[3]
    assert pair_probability_from_ratings(1, 2, ratings) > 0.5
    assert pair_probability_from_ratings(2, 3, ratings) > 0.5
