from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.modeh7 import (
    MODEH7_FEATURE_COLUMNS,
    Modeh7FeatureTables,
    apply_modeh7_calibration,
    build_modeh7_submission_frame,
    fit_modeh7_ensemble,
    prepare_modeh7_results,
    score_modeh7_oof,
)
import mmmania.modeh7 as modeh7


def test_prepare_modeh7_results_builds_symmetric_rows() -> None:
    raw = pd.DataFrame(
        [
            {
                "Season": 2026,
                "DayNum": 10,
                "LTeamID": 1102,
                "LScore": 70,
                "WTeamID": 1101,
                "WScore": 80,
                "NumOT": 0,
                "LFGM": 25,
                "LFGA": 55,
                "LFGM3": 7,
                "LFGA3": 20,
                "LFTM": 13,
                "LFTA": 18,
                "LOR": 8,
                "LDR": 20,
                "LAst": 12,
                "LTO": 11,
                "LStl": 4,
                "LBlk": 2,
                "LPF": 16,
                "WFGM": 30,
                "WFGA": 60,
                "WFGM3": 10,
                "WFGA3": 25,
                "WFTM": 10,
                "WFTA": 14,
                "WOR": 9,
                "WDR": 24,
                "WAst": 16,
                "WTO": 10,
                "WStl": 5,
                "WBlk": 3,
                "WPF": 15,
            }
        ]
    )

    prepared = prepare_modeh7_results(raw)

    assert len(prepared) == 2
    assert set(prepared["PointDiff"].tolist()) == {10, -10}
    assert set(prepared["win"].tolist()) == {0, 1}
    assert prepared["men_women"].tolist() == [1, 1]


def test_apply_modeh7_calibration_uses_gender_specific_calibrators() -> None:
    calibrators = {
        "men": lambda values: np.full(len(values), 0.61),
        "women": lambda values: np.full(len(values), 0.39),
    }

    probabilities = apply_modeh7_calibration(
        np.array([5.0, -3.0, 2.0, -1.0]),
        pd.Series([1, 0, 1, 0]),
        calibrators,
        calibration_mode="by_gender",
        clip_margin=25.0,
    )

    assert probabilities.tolist() == [0.61, 0.39, 0.61, 0.39]


def test_build_modeh7_submission_frame_merges_feature_tables() -> None:
    feature_tables = Modeh7FeatureTables(
        season_stats_t1=pd.DataFrame(
            [{"Season": 2026, "T1_TeamID": 1101, "T1_avg_Score": 80.0, "T1_avg_PointDiff": 10.0}]
        ),
        season_stats_t2=pd.DataFrame(
            [{"Season": 2026, "T2_TeamID": 1102, "T2_avg_Score": 70.0, "T2_avg_PointDiff": -2.0}]
        ),
        seeds_t1=pd.DataFrame([{"Season": 2026, "T1_TeamID": 1101, "T1_seed": 3}]),
        seeds_t2=pd.DataFrame([{"Season": 2026, "T2_TeamID": 1102, "T2_seed": 11}]),
        elo_t1=pd.DataFrame([{"Season": 2026, "T1_TeamID": 1101, "T1_elo": 1120.0}]),
        elo_t2=pd.DataFrame([{"Season": 2026, "T2_TeamID": 1102, "T2_elo": 1010.0}]),
        quality_t1=pd.DataFrame([{"Season": 2026, "T1_TeamID": 1101, "T1_quality": 8.0}]),
        quality_t2=pd.DataFrame([{"Season": 2026, "T2_TeamID": 1102, "T2_quality": 1.0}]),
    )

    frame = build_modeh7_submission_frame(pd.Series(["2026_1101_1102"]), feature_tables)

    assert int(frame.iloc[0]["T1_TeamID"]) == 1101
    assert int(frame.iloc[0]["T2_TeamID"]) == 1102
    assert int(frame.iloc[0]["Seed_diff"]) == 8
    assert float(frame.iloc[0]["elo_diff"]) == 110.0
    assert float(frame.iloc[0]["T1_quality"]) == 8.0
    assert float(frame.iloc[0]["T2_quality"]) == 1.0


def test_score_modeh7_oof_applies_aggressiveness_before_calibration() -> None:
    ensemble = type(
        "Ensemble",
        (),
        {
            "calibrators": {"men": lambda values: values / 10.0, "women": lambda values: values / 10.0},
            "calibration_mode": "by_gender",
            "clip_margin": 25.0,
            "prob_clip": (0.01, 0.99),
        },
    )()
    oof = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "men_women": [1, 0],
            "pred_margin": [4.0, 4.0],
            "label": [1, 1],
        }
    )

    base = score_modeh7_oof(oof, ensemble, aggressiveness=1.0)
    tuned = score_modeh7_oof(oof, ensemble, aggressiveness=0.5)

    assert tuned["overall_brier"] > base["overall_brier"]


def test_fit_modeh7_ensemble_raises_if_xgboost_runtime_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(modeh7, "_xgboost_probe_error", lambda: "sandbox shm failure")

    rows = []
    for season in (2025, 2026):
        for offset in range(4):
            margin = float((offset - 1.5) * (1 if season == 2026 else -1))
            row = {
                "Season": season,
                "PointDiff": margin,
                "win": int(margin > 0.0),
                "men_women": offset % 2,
            }
            for index, column in enumerate(MODEH7_FEATURE_COLUMNS):
                if column == "men_women":
                    row[column] = offset % 2
                else:
                    row[column] = float((index + 1) * (offset + 1) + (season - 2024))
            rows.append(row)

    training_frame = pd.DataFrame(rows)
    try:
        fit_modeh7_ensemble(
            training_frame,
            validation="loso",
            calibration_mode="combined",
        )
    except RuntimeError as exc:
        assert "sandbox shm failure" in str(exc)
    else:
        raise AssertionError("Expected fit_modeh7_ensemble to raise when XGBoost is unavailable.")
