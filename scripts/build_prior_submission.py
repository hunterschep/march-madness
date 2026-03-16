from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.config import DATA_DIR, OUTPUT_DIR, ensure_output_dirs, get_side
from mmmania.features import build_team_season_features
from mmmania.modeling import (
    PRIOR_SPEC,
    build_pair_dataset,
    build_pair_rows_for_submission,
    fit_final_model,
    tune_model,
)


def _predict_side(sample: pd.DataFrame, side_code: str) -> pd.DataFrame:
    side = get_side(side_code)
    team_features = build_team_season_features(side)
    pair_dataset = build_pair_dataset(side, team_features, PRIOR_SPEC)
    tuned_model, _ = tune_model(side, pair_dataset, PRIOR_SPEC)
    model, _ = fit_final_model(pair_dataset, tuned_model)

    side_ids = sample.loc[
        sample["ID"].str.contains("_3") if side_code == "W" else ~sample["ID"].str.contains("_3"),
        "ID",
    ]
    feature_rows = build_pair_rows_for_submission(
        side=side,
        team_features=team_features,
        season=2026,
        spec=PRIOR_SPEC,
        ids=side_ids,
    )
    probabilities = model.predict_proba(feature_rows[list(tuned_model.feature_columns)].fillna(0.0))[:, 1]
    predicted = feature_rows[["ID"]].copy()
    predicted["Pred"] = probabilities.clip(0.001, 0.999)
    return predicted


def main() -> None:
    ensure_output_dirs()

    sample = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
    men = _predict_side(sample, "M")
    women = _predict_side(sample, "W")

    submission = (
        pd.concat([men, women], ignore_index=True)
        .set_index("ID")
        .reindex(sample["ID"])
        .reset_index()
    )
    submission.to_csv(OUTPUT_DIR / "submissions" / "prior_submission_2026.csv", index=False)


if __name__ == "__main__":
    main()
