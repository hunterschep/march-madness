from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, CompetitionSide
from .features import EXTERNAL_FEATURE_PREFIX, get_model_feature_columns, get_tournament_seeds


TARGET_SEASONS = [2021, 2022, 2023, 2024, 2025]


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
    weighted_brier: float


PRIOR_SPEC = ModelSpec(name="prior", include_seeds=False)
SEEDED_SPEC = ModelSpec(name="seeded", include_seeds=True)


def _tournament_games(side: CompetitionSide) -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / f"{side.code}NCAATourneyCompactResults.csv",
        usecols=["Season", "DayNum", "WTeamID", "LTeamID", "WLoc"],
    )


def build_pair_dataset(
    side: CompetitionSide,
    team_features: pd.DataFrame,
    spec: ModelSpec,
) -> pd.DataFrame:
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
    if spec.include_seeds:
        dataset["seed_low"] = dataset["seed_num_low"]
        dataset["seed_high"] = dataset["seed_num_high"]
        base_columns.extend(["seed_low", "seed_high"])

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
    pair_dataset: pd.DataFrame,
    candidate: CandidateModel,
    test_seasons: Iterable[int] = TARGET_SEASONS,
) -> dict:
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
        probabilities = model.predict_proba(test[list(candidate.feature_columns)].fillna(0.0))[:, 1]
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
    return {
        "candidate": candidate.name,
        "feature_columns": list(candidate.feature_columns),
        "regularization_c": candidate.regularization_c,
        "use_seed_matchup_prior": candidate.use_seed_matchup_prior,
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
        evaluate_candidate(pair_dataset, candidate, test_seasons)
        for candidate in _candidate_feature_sets(side, pair_dataset, spec)
    ]
    best = min(evaluations, key=lambda item: item["weighted_brier"])
    tuned = TunedModel(
        side=side.label,
        spec=spec.name,
        feature_columns=tuple(best["feature_columns"]),
        regularization_c=best["regularization_c"],
        use_seed_matchup_prior=best["use_seed_matchup_prior"],
        weighted_brier=best["weighted_brier"],
    )
    return tuned, evaluations


def fit_final_model(
    pair_dataset: pd.DataFrame,
    tuned_model: TunedModel,
) -> tuple[Pipeline, dict | None]:
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


def build_pair_rows_for_submission(
    side: CompetitionSide,
    team_features: pd.DataFrame,
    season: int,
    spec: ModelSpec,
    ids: pd.Series,
) -> pd.DataFrame:
    pairs = ids.str.split("_", expand=True)
    frame = pd.DataFrame(
        {
            "ID": ids,
            "Season": pairs[0].astype(int),
            "TeamLowID": pairs[1].astype(int),
            "TeamHighID": pairs[2].astype(int),
        }
    )

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

    base_columns = ["ID", "Season", "TeamLowID", "TeamHighID"]
    if spec.include_seeds:
        base_columns.extend(["seed_low", "seed_high"])
    frame = frame[[*base_columns, *diff_columns]]
    frame = frame[frame["Season"] == season].copy()
    return frame
