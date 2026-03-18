from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.modeling import TARGET_SEASONS, parse_submission_ids


def test_parse_submission_ids_extracts_expected_columns() -> None:
    parsed = parse_submission_ids(pd.Series(["2026_1116_1429", "2026_3116_3429"]))

    assert parsed.to_dict("records") == [
        {"ID": "2026_1116_1429", "Season": 2026, "TeamLowID": 1116, "TeamHighID": 1429},
        {"ID": "2026_3116_3429", "Season": 2026, "TeamLowID": 3116, "TeamHighID": 3429},
    ]


def test_target_seasons_match_final_backtest_window() -> None:
    assert TARGET_SEASONS == [2021, 2022, 2023, 2024, 2025]
