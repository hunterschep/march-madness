from __future__ import annotations

"""Tournament-only pair modeling and tuned spec reuse."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, CompetitionSide
from .features import EXTERNAL_FEATURE_PREFIX, get_model_feature_columns, get_tournament_seeds


TARGET_SEASONS = [2021, 2022, 2023, 2024, 2025]
GLOBAL_SCALE_GRID = (0.88, 0.92, 0.96, 1.0)
ROUND_SCALE_GRID = (0.9, 0.95, 1.0)
SEED_GAP_SCALE_GRID = (0.9, 0.95, 1.0)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    include_seeds: bool


@dataclass(frozen=True)
class CandidateModel:
    name: str
    feature_columns: tuple[str, ...]
    regularization_c: float
    use_seed_matchup_prior: bool = False


@dataclass(frozen=True)
class TunedModel:
    side: str
    spec: str
    feature_columns: tuple[str, ...]
    regularization_c: float
    use_seed_matchup_prior: bool
    global_scale: float
    round_bucket_scales: tuple[float, ...]
    seed_gap_bucket_scales: tuple[float, ...]
    weighted_brier: float


PRIOR_SPEC = ModelSpec(name="prior", include_seeds=False)
SEEDED_SPEC = ModelSpec(name="seeded", include_seeds=True)


def round_bucket_from_daynum(day_num: int | float) -> int:
    day_num = int(day_num)
    if day_num <= 137:
        return 0
    if day_num <= 139:
        return 1
    return 2


def round_bucket_from_round_number(round_number: int | float | None) -> int | None:
    if round_number is None or pd.isna(round_number):
        return None
    round_number = int(round_number)
    if round_number <= 1:
        return 0
    if round_number == 2:
        return 1
    return 2


def seed_gap_bucket(seed_gap: int | float | None) -> int | None:
    if seed_gap is None or pd.isna(seed_gap):
        return None
    seed_gap = abs(int(seed_gap))
    if seed_gap <= 1:
        return 0
    if seed_gap <= 4:
        return 1
    if seed_gap <= 8:
        return 2
    return 3


def calibrate_probabilities(
    probabilities: pd.Series | list[float],
    *,
    global_scale: float = 1.0,
    round_buckets: pd.Series | list[int] | None = None,
    round_bucket_scales: tuple[float, ...] = (1.0, 1.0, 1.0),
    seed_gap_buckets: pd.Series | list[int] | None = None,
    seed_gap_bucket_scales: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
) -> pd.Series:
    probabilities = pd.Series(probabilities, dtype=float).reset_index(drop=True)
    scale = pd.Series(global_scale, index=probabilities.index, dtype=float)

    if round_buckets is not None:
        round_buckets = pd.Series(round_buckets).reset_index(drop=True)
        for bucket_index, bucket_scale in enumerate(round_bucket_scales):
            scale.loc[round_buckets == bucket_index] *= bucket_scale

    if seed_gap_buckets is not None:
        seed_gap_buckets = pd.Series(seed_gap_buckets).reset_index(drop=True)
        for bucket_index, bucket_scale in enumerate(seed_gap_bucket_scales):
            scale.loc[seed_gap_buckets == bucket_index] *= bucket_scale

    adjusted = 0.5 + scale * (probabilities - 0.5)
    return adjusted.clip(0.001, 0.999)


def _tournament_games(side: CompetitionSide) -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / f"{side.code}NCAATourneyCompactResults.csv",
        usecols=["Season", "DayNum", "WTeamID", "LTeamID", "WLoc"],
    )


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


def build_pair_round_lookup(side_code: str, season: int) -> dict[str, int]:
    """Map each tournament-team pair to the round where they can meet in the fixed bracket."""
    seeds = pd.read_csv(
        DATA_DIR / f"{side_code}NCAATourneySeeds.csv",
        usecols=["Season", "Seed", "TeamID"],
    )
    seeds = seeds[seeds["Season"] == season].copy()
    possibilities = {str(row.Seed): {int(row.TeamID)} for row in seeds.itertuples(index=False)}

    slots = pd.read_csv(
        DATA_DIR / f"{side_code}NCAATourneySlots.csv",
        usecols=["Season", "Slot", "StrongSeed", "WeakSeed"],
    )
    slots = slots[slots["Season"] == season].copy()
    slots["round_num"] = slots["Slot"].map(
        lambda slot: int(slot[1]) if str(slot).startswith("R") and len(str(slot)) > 1 and str(slot)[1].isdigit() else 0
    )
    slots = slots.sort_values(["round_num", "Slot"])

    round_lookup: dict[str, int] = {}
    for row in slots.itertuples(index=False):
        strong_teams = possibilities[str(row.StrongSeed)]
        weak_teams = possibilities[str(row.WeakSeed)]
        for strong_team_id in strong_teams:
            for weak_team_id in weak_teams:
                low_id = min(strong_team_id, weak_team_id)
                high_id = max(strong_team_id, weak_team_id)
                round_lookup[f"{low_id}_{high_id}"] = int(row.round_num)
        possibilities[str(row.Slot)] = strong_teams.union(weak_teams)

    return round_lookup


def build_pair_dataset(
    side: CompetitionSide,
    team_features: pd.DataFrame,
    spec: ModelSpec,
) -> pd.DataFrame:
    """Build historical tournament matchups as pairwise feature differences."""
    features = team_features.copy()
    feature_columns = get_model_feature_columns(
        side, include_seeds=spec.include_seeds, team_features=team_features
    )
    if spec.include_seeds:
        features = features.merge(get_tournament_seeds(side), on=["Season", "TeamID"], how="left")

    tournament = _tournament_games(side).copy()
    tournament["TeamLowID"] = tournament[["WTeamID", "LTeamID"]].min(axis=1)
    tournament["TeamHighID"] = tournament[["WTeamID", "LTeamID"]].max(axis=1)
    tournament["Target"] = (tournament["WTeamID"] == tournament["TeamLowID"]).astype(int)

    low = features.rename(columns={"TeamID": "TeamLowID"})
    high = features.rename(columns={"TeamID": "TeamHighID"})

    dataset = tournament.merge(low, on=["Season", "TeamLowID"], how="left").merge(
        high, on=["Season", "TeamHighID"], how="left", suffixes=("_low", "_high")
    )

    diff_columns = []
    for column in feature_columns:
        diff_name = f"{column}_diff"
        dataset[diff_name] = dataset[f"{column}_low"] - dataset[f"{column}_high"]
        diff_columns.append(diff_name)

    base_columns = ["Season", "DayNum", "TeamLowID", "TeamHighID", "Target"]
    dataset["round_bucket"] = dataset["DayNum"].map(round_bucket_from_daynum)
    base_columns.append("round_bucket")
    if spec.include_seeds:
        dataset["seed_low"] = dataset["seed_num_low"]
        dataset["seed_high"] = dataset["seed_num_high"]
        dataset["seed_gap"] = (dataset["seed_low"] - dataset["seed_high"]).abs()
        dataset["seed_gap_bucket"] = dataset["seed_gap"].map(seed_gap_bucket)
        base_columns.extend(["seed_low", "seed_high", "seed_gap", "seed_gap_bucket"])

    dataset = dataset[[*base_columns, *diff_columns]]
    return dataset


def build_logistic_model(regularization_c: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=4000, solver="liblinear", C=regularization_c),
            ),
        ]
    )


def _seed_matchup_prior_payload(train: pd.DataFrame, alpha: float = 2.0) -> dict:
    global_mean = float(train["Target"].mean())
    counts = (
        train.groupby(["seed_low", "seed_high"])["Target"]
        .agg(["sum", "count"])
        .reset_index()
    )
    counts["seed_matchup_prior"] = (counts["sum"] + alpha * global_mean) / (counts["count"] + alpha)
    return {
        "global_mean": global_mean,
        "rows": counts[["seed_low", "seed_high", "seed_matchup_prior"]].to_dict("records"),
    }


def apply_seed_matchup_prior(frame: pd.DataFrame, payload: dict | None) -> pd.DataFrame:
    if payload is None:
        return frame

    seed_prior = pd.DataFrame(payload["rows"])
    frame = frame.merge(seed_prior, on=["seed_low", "seed_high"], how="left")
    frame["seed_matchup_prior"] = frame["seed_matchup_prior"].fillna(payload["global_mean"])
    return frame


def _with_seed_matchup_prior(
    train: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = _seed_matchup_prior_payload(train, alpha=alpha)
    return apply_seed_matchup_prior(train, payload), apply_seed_matchup_prior(test, payload)


def _round_buckets(frame: pd.DataFrame) -> pd.Series | None:
    if "round_bucket" in frame.columns:
        return frame["round_bucket"]
    if "DayNum" in frame.columns:
        return frame["DayNum"].map(round_bucket_from_daynum)
    if "possible_round" in frame.columns:
        return frame["possible_round"].map(round_bucket_from_round_number)
    return None


def _seed_gap_buckets(frame: pd.DataFrame) -> pd.Series | None:
    if "seed_gap_bucket" in frame.columns:
        return frame["seed_gap_bucket"]
    if "seed_gap" in frame.columns:
        return frame["seed_gap"].map(seed_gap_bucket)
    if {"seed_low", "seed_high"}.issubset(frame.columns):
        return (frame["seed_low"] - frame["seed_high"]).abs().map(seed_gap_bucket)
    return None


def _tune_calibration(
    side: CompetitionSide,
    spec: ModelSpec,
    train: pd.DataFrame,
    raw_probabilities: pd.Series,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    target = train["Target"]
    round_buckets = _round_buckets(train)
    gap_buckets = _seed_gap_buckets(train) if spec.include_seeds and side.code == "M" else None

    global_scale = 1.0
    round_scales = [1.0, 1.0, 1.0]
    gap_scales = [1.0, 1.0, 1.0, 1.0]

    def score(
        global_scale_value: float,
        round_scale_values: list[float] | tuple[float, ...],
        gap_scale_values: list[float] | tuple[float, ...],
    ) -> float:
        calibrated = calibrate_probabilities(
            raw_probabilities,
            global_scale=global_scale_value,
            round_buckets=round_buckets,
            round_bucket_scales=tuple(round_scale_values),
            seed_gap_buckets=gap_buckets,
            seed_gap_bucket_scales=tuple(gap_scale_values),
        )
        return float(brier_score_loss(target, calibrated))

    best_global = global_scale
    best_score = score(best_global, round_scales, gap_scales)
    for candidate_scale in GLOBAL_SCALE_GRID:
        candidate_score = score(candidate_scale, round_scales, gap_scales)
        if candidate_score < best_score:
            best_global = candidate_scale
            best_score = candidate_score
    global_scale = best_global

    if spec.include_seeds and side.code == "M" and round_buckets is not None:
        for bucket_index in range(len(round_scales)):
            best_bucket_scale = round_scales[bucket_index]
            best_bucket_score = best_score
            for candidate_scale in ROUND_SCALE_GRID:
                candidate_round_scales = list(round_scales)
                candidate_round_scales[bucket_index] = candidate_scale
                candidate_score = score(global_scale, candidate_round_scales, gap_scales)
                if candidate_score < best_bucket_score:
                    best_bucket_scale = candidate_scale
                    best_bucket_score = candidate_score
            round_scales[bucket_index] = best_bucket_scale
            best_score = best_bucket_score

    if gap_buckets is not None:
        for bucket_index in range(len(gap_scales)):
            bucket_count = int((gap_buckets == bucket_index).sum())
            if bucket_count < 12:
                continue
            best_bucket_scale = gap_scales[bucket_index]
            best_bucket_score = best_score
            for candidate_scale in SEED_GAP_SCALE_GRID:
                candidate_gap_scales = list(gap_scales)
                candidate_gap_scales[bucket_index] = candidate_scale
                candidate_score = score(global_scale, round_scales, candidate_gap_scales)
                if candidate_score < best_bucket_score:
                    best_bucket_scale = candidate_scale
                    best_bucket_score = candidate_score
            gap_scales[bucket_index] = best_bucket_scale
            best_score = best_bucket_score

    for candidate_scale in GLOBAL_SCALE_GRID:
        candidate_score = score(candidate_scale, round_scales, gap_scales)
        if candidate_score < best_score:
            global_scale = candidate_scale
            best_score = candidate_score

    return global_scale, tuple(round_scales), tuple(gap_scales)


def _oof_calibration_frame(
    train: pd.DataFrame,
    candidate: CandidateModel,
) -> tuple[pd.DataFrame, pd.Series] | None:
    seasons = sorted(int(season) for season in train["Season"].unique())
    frames = []
    predictions = []

    for season in seasons:
        inner_train = train[train["Season"] < season].copy()
        inner_valid = train[train["Season"] == season].copy()
        if inner_train.empty or inner_valid.empty:
            continue

        model = build_logistic_model(candidate.regularization_c)
        model.fit(inner_train[list(candidate.feature_columns)].fillna(0.0), inner_train["Target"])
        inner_predictions = model.predict_proba(
            inner_valid[list(candidate.feature_columns)].fillna(0.0)
        )[:, 1]
        frames.append(inner_valid)
        predictions.append(pd.Series(inner_predictions))

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True), pd.concat(predictions, ignore_index=True)


def _candidate_feature_sets(
    side: CompetitionSide,
    pair_dataset: pd.DataFrame,
    spec: ModelSpec,
) -> list[CandidateModel]:
    available = set(pair_dataset.columns)
    external_features = tuple(
        sorted(col for col in available if col.startswith(EXTERNAL_FEATURE_PREFIX) and col.endswith("_diff"))
    )

    if spec.name == PRIOR_SPEC.name:
        base_sets = [
            (
                "adj_core",
                ("adj_margin_rating_diff", "recent_margin_diff", "conference_net_eff_diff"),
                False,
            ),
            (
                "off_def_core",
                ("off_eff_diff", "def_eff_diff", "recent_margin_diff", "conference_net_eff_diff"),
                False,
            ),
            (
                "strength_core",
                ("net_eff_diff", "recent_margin_diff", "conference_net_eff_diff"),
                False,
            ),
        ]
    else:
        base_sets = [
            ("seed_only", ("seed_num_diff",), False),
            (
                "seed_core",
                ("seed_num_diff", "net_eff_diff", "recent_margin_diff", "conference_net_eff_diff"),
                False,
            ),
            (
                "seed_prior_adj_core",
                ("seed_matchup_prior", "seed_num_diff", "adj_margin_rating_diff", "recent_margin_diff"),
                True,
            ),
            (
                "seed_prior_adj_plus_conf",
                (
                    "seed_matchup_prior",
                    "seed_num_diff",
                    "adj_margin_rating_diff",
                    "recent_margin_diff",
                    "conference_net_eff_diff",
                ),
                True,
            ),
            (
                "seed_prior_core",
                ("seed_matchup_prior", "seed_num_diff", "net_eff_diff", "recent_margin_diff", "conference_net_eff_diff"),
                True,
            ),
        ]
        if side.code == "M":
            base_sets.extend(
                [
                    (
                        "seed_prior_adj_net",
                        (
                            "seed_matchup_prior",
                            "seed_num_diff",
                            "adj_margin_rating_diff",
                            "net_eff_diff",
                            "recent_margin_diff",
                        ),
                        True,
                    ),
                    (
                        "seed_prior_adj_net_conf",
                        (
                            "seed_matchup_prior",
                            "seed_num_diff",
                            "adj_margin_rating_diff",
                            "net_eff_diff",
                            "recent_margin_diff",
                            "conference_net_eff_diff",
                        ),
                        True,
                    ),
                    (
                        "seed_prior_adj_massey",
                        (
                            "seed_matchup_prior",
                            "seed_num_diff",
                            "adj_margin_rating_diff",
                            "recent_margin_diff",
                            "massey_pct_diff",
                        ),
                        True,
                    ),
                    (
                        "seed_prior_adj_net_massey",
                        (
                            "seed_matchup_prior",
                            "seed_num_diff",
                            "adj_margin_rating_diff",
                            "net_eff_diff",
                            "recent_margin_diff",
                            "massey_pct_diff",
                        ),
                        True,
                    ),
                    (
                        "seed_prior_adj_net_massey_depth",
                        (
                            "seed_matchup_prior",
                            "seed_num_diff",
                            "adj_margin_rating_diff",
                            "net_eff_diff",
                            "recent_margin_diff",
                            "massey_pct_diff",
                            "massey_system_count_diff",
                        ),
                        True,
                    ),
                ]
            )

    candidates: list[CandidateModel] = []
    for name, features, use_seed_prior in base_sets:
        if not all(feature in available or feature == "seed_matchup_prior" for feature in features):
            continue
        for regularization_c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            candidates.append(
                CandidateModel(
                    name=name,
                    feature_columns=features,
                    regularization_c=regularization_c,
                    use_seed_matchup_prior=use_seed_prior,
                )
            )
            if external_features:
                ext_features = tuple([*features, *external_features])
                if len(set(ext_features)) == len(ext_features):
                    candidates.append(
                        CandidateModel(
                            name=f"{name}_external",
                            feature_columns=ext_features,
                            regularization_c=regularization_c,
                            use_seed_matchup_prior=use_seed_prior,
                        )
                    )

    return candidates


def evaluate_candidate(
    side: CompetitionSide,
    spec: ModelSpec,
    pair_dataset: pd.DataFrame,
    candidate: CandidateModel,
    test_seasons: Iterable[int] = TARGET_SEASONS,
) -> dict:
    """Score one candidate spec with rolling held-out tournament seasons."""
    season_metrics = []
    for season in test_seasons:
        train = pair_dataset[pair_dataset["Season"] < season].copy()
        test = pair_dataset[pair_dataset["Season"] == season].copy()
        if train.empty or test.empty:
            continue

        if candidate.use_seed_matchup_prior:
            train, test = _with_seed_matchup_prior(train, test)

        model = build_logistic_model(candidate.regularization_c)
        model.fit(train[list(candidate.feature_columns)].fillna(0.0), train["Target"])
        calibration_data = _oof_calibration_frame(train, candidate)
        if calibration_data is None:
            calibration_frame = train
            calibration_probabilities = pd.Series(
                model.predict_proba(train[list(candidate.feature_columns)].fillna(0.0))[:, 1]
            )
        else:
            calibration_frame, calibration_probabilities = calibration_data
        global_scale, round_scales, gap_scales = _tune_calibration(
            side=side,
            spec=spec,
            train=calibration_frame,
            raw_probabilities=calibration_probabilities,
        )
        probabilities = pd.Series(
            model.predict_proba(test[list(candidate.feature_columns)].fillna(0.0))[:, 1]
        )
        probabilities = calibrate_probabilities(
            probabilities,
            global_scale=global_scale,
            round_buckets=_round_buckets(test),
            round_bucket_scales=round_scales,
            seed_gap_buckets=_seed_gap_buckets(test),
            seed_gap_bucket_scales=gap_scales,
        )
        brier = brier_score_loss(test["Target"], probabilities)
        season_metrics.append(
            {
                "season": int(season),
                "games": int(len(test)),
                "brier": float(brier),
            }
        )

    weighted_brier = sum(item["brier"] * item["games"] for item in season_metrics) / sum(
        item["games"] for item in season_metrics
    )
    full_train = pair_dataset.copy()
    if candidate.use_seed_matchup_prior:
        full_train = apply_seed_matchup_prior(full_train, _seed_matchup_prior_payload(full_train))
    full_calibration_data = _oof_calibration_frame(full_train, candidate)
    if full_calibration_data is None:
        full_model = build_logistic_model(candidate.regularization_c)
        full_model.fit(full_train[list(candidate.feature_columns)].fillna(0.0), full_train["Target"])
        calibration_frame = full_train
        full_train_probabilities = pd.Series(
            full_model.predict_proba(full_train[list(candidate.feature_columns)].fillna(0.0))[:, 1]
        )
    else:
        calibration_frame, full_train_probabilities = full_calibration_data
    global_scale, round_scales, gap_scales = _tune_calibration(
        side=side,
        spec=spec,
        train=calibration_frame,
        raw_probabilities=full_train_probabilities,
    )
    return {
        "candidate": candidate.name,
        "feature_columns": list(candidate.feature_columns),
        "regularization_c": candidate.regularization_c,
        "use_seed_matchup_prior": candidate.use_seed_matchup_prior,
        "global_scale": global_scale,
        "round_bucket_scales": list(round_scales),
        "seed_gap_bucket_scales": list(gap_scales),
        "weighted_brier": float(weighted_brier),
        "season_metrics": season_metrics,
    }


def tune_model(
    side: CompetitionSide,
    pair_dataset: pd.DataFrame,
    spec: ModelSpec,
    test_seasons: Iterable[int] = TARGET_SEASONS,
) -> tuple[TunedModel, list[dict]]:
    evaluations = [
        evaluate_candidate(side, spec, pair_dataset, candidate, test_seasons)
        for candidate in _candidate_feature_sets(side, pair_dataset, spec)
    ]
    best = min(evaluations, key=lambda item: item["weighted_brier"])
    tuned = TunedModel(
        side=side.label,
        spec=spec.name,
        feature_columns=tuple(best["feature_columns"]),
        regularization_c=best["regularization_c"],
        use_seed_matchup_prior=best["use_seed_matchup_prior"],
        global_scale=best["global_scale"],
        round_bucket_scales=tuple(best["round_bucket_scales"]),
        seed_gap_bucket_scales=tuple(best["seed_gap_bucket_scales"]),
        weighted_brier=best["weighted_brier"],
    )
    return tuned, evaluations


def fit_final_model(
    pair_dataset: pd.DataFrame,
    tuned_model: TunedModel,
) -> tuple[Pipeline, dict | None]:
    """Fit the chosen model spec on all available historical tournament games."""
    train = pair_dataset.copy()
    seed_prior_payload = None
    if tuned_model.use_seed_matchup_prior:
        seed_prior_payload = _seed_matchup_prior_payload(train)
        train = apply_seed_matchup_prior(train, seed_prior_payload)

    model = build_logistic_model(tuned_model.regularization_c)
    model.fit(train[list(tuned_model.feature_columns)].fillna(0.0), train["Target"])
    return model, seed_prior_payload


def serialize_tuned_model(tuned_model: TunedModel) -> dict:
    payload = asdict(tuned_model)
    payload["feature_columns"] = list(tuned_model.feature_columns)
    return payload


def load_tuned_model(backtest_path: Path, side_label: str, spec_name: str) -> TunedModel | None:
    """Reuse a previously tuned model spec instead of re-running the search."""
    if not backtest_path.exists():
        return None

    payload = json.loads(backtest_path.read_text())
    for row in payload.get("results", []):
        if row.get("side") == side_label and row.get("model") == spec_name:
            tuned = row.get("tuned_model", {})
            return TunedModel(
                side=tuned["side"],
                spec=tuned["spec"],
                feature_columns=tuple(tuned["feature_columns"]),
                regularization_c=tuned["regularization_c"],
                use_seed_matchup_prior=tuned["use_seed_matchup_prior"],
                global_scale=tuned.get("global_scale", 1.0),
                round_bucket_scales=tuple(tuned.get("round_bucket_scales", [1.0, 1.0, 1.0])),
                seed_gap_bucket_scales=tuple(tuned.get("seed_gap_bucket_scales", [1.0, 1.0, 1.0, 1.0])),
                weighted_brier=tuned["weighted_brier"],
            )
    return None


def build_pair_rows_for_submission(
    side: CompetitionSide,
    team_features: pd.DataFrame,
    season: int,
    spec: ModelSpec,
    ids: pd.Series,
) -> pd.DataFrame:
    """Build pairwise features for arbitrary submission rows."""
    frame = parse_submission_ids(ids)

    features = team_features.copy()
    if spec.include_seeds:
        features = features.merge(get_tournament_seeds(side), on=["Season", "TeamID"], how="left")

    feature_columns = get_model_feature_columns(
        side, include_seeds=spec.include_seeds, team_features=team_features
    )
    low = features.rename(columns={"TeamID": "TeamLowID"})
    high = features.rename(columns={"TeamID": "TeamHighID"})
    frame = frame.merge(low, on=["Season", "TeamLowID"], how="left").merge(
        high, on=["Season", "TeamHighID"], how="left", suffixes=("_low", "_high")
    )

    diff_columns = []
    for column in feature_columns:
        diff_name = f"{column}_diff"
        frame[diff_name] = frame[f"{column}_low"] - frame[f"{column}_high"]
        diff_columns.append(diff_name)

    if spec.include_seeds:
        frame["seed_low"] = frame["seed_num_low"]
        frame["seed_high"] = frame["seed_num_high"]
        frame["seed_gap"] = (frame["seed_low"] - frame["seed_high"]).abs()

    base_columns = ["ID", "Season", "TeamLowID", "TeamHighID"]
    if spec.include_seeds:
        base_columns.extend(["seed_low", "seed_high", "seed_gap"])
    frame = frame[[*base_columns, *diff_columns]]
    frame = frame[frame["Season"] == season].copy()
    return frame
