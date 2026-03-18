from __future__ import annotations

"""Small shared utilities for submission IDs and target backtest seasons."""

import pandas as pd


TARGET_SEASONS = [2021, 2022, 2023, 2024, 2025]


def parse_submission_ids(ids: pd.Series) -> pd.DataFrame:
    """Expand Kaggle submission IDs into season / low-ID / high-ID columns."""
    pairs = ids.str.split("_", expand=True)
    return pd.DataFrame(
        {
            "ID": ids,
            "Season": pairs[0].astype(int),
            "TeamLowID": pairs[1].astype(int),
            "TeamHighID": pairs[2].astype(int),
        }
    )
