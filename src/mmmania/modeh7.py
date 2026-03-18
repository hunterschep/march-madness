from __future__ import annotations

"""Notebook-style prior model based on the 2025 modeh7 approach."""

import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import brier_score_loss

from .config import DATA_DIR, OUTPUT_DIR
from .modeling import parse_submission_ids


MODEH7_FEATURE_COLUMNS = [
    "men_women",
    "T1_seed",
    "T2_seed",
    "Seed_diff",
    "T1_avg_Score",
    "T1_avg_FGA",
    "T1_avg_OR",
    "T1_avg_DR",
    "T1_avg_Blk",
    "T1_avg_PF",
    "T1_avg_opponent_FGA",
    "T1_avg_opponent_Blk",
    "T1_avg_opponent_PF",
    "T1_avg_PointDiff",
    "T2_avg_Score",
    "T2_avg_FGA",
    "T2_avg_OR",
    "T2_avg_DR",
    "T2_avg_Blk",
    "T2_avg_PF",
    "T2_avg_opponent_FGA",
    "T2_avg_opponent_Blk",
    "T2_avg_opponent_PF",
    "T2_avg_PointDiff",
    "T1_elo",
    "T2_elo",
    "elo_diff",
    "T1_quality",
    "T2_quality",
]

MODEH7_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "eta": 0.008,
    "subsample": 0.7,
    "colsample_bynode": 0.7,
    "num_parallel_tree": 2,
    "min_child_weight": 6,
    "max_depth": 3,
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_bin": 38,
    "nthread": 1,
}

MODEH7_NUM_ROUNDS = 900
MODEH7_CLIP_MARGIN = 25.0
MODEH7_PROB_CLIP = (0.01, 0.99)
MODEH7_CACHE_PATH = OUTPUT_DIR / "cache" / "modeh7_training_cache.pkl"
MODEH7_DEFAULT_VALIDATION = "loso"
MODEH7_DEFAULT_CALIBRATION_MODE = "by_gender"
MODEH7_DEFAULT_AGGRESSIVENESS = 1.01
MODEH7_BACKEND = "xgboost"
MODEH7_XGB_PROBE_TIMEOUT_SECONDS = 30

_BASE_RESULT_COLUMNS = [
    "Season",
    "DayNum",
    "LTeamID",
    "LScore",
    "WTeamID",
    "WScore",
    "NumOT",
    "LFGM",
    "LFGA",
    "LFGM3",
    "LFGA3",
    "LFTM",
    "LFTA",
    "LOR",
    "LDR",
    "LAst",
    "LTO",
    "LStl",
    "LBlk",
    "LPF",
    "WFGM",
    "WFGA",
    "WFGM3",
    "WFGA3",
    "WFTM",
    "WFTA",
    "WOR",
    "WDR",
    "WAst",
    "WTO",
    "WStl",
    "WBlk",
    "WPF",
]

_OT_ADJUST_COLUMNS = [
    "LScore",
    "WScore",
    "LFGM",
    "LFGA",
    "LFGM3",
    "LFGA3",
    "LFTM",
    "LFTA",
    "LOR",
    "LDR",
    "LAst",
    "LTO",
    "LStl",
    "LBlk",
    "LPF",
    "WFGM",
    "WFGA",
    "WFGM3",
    "WFGA3",
    "WFTM",
    "WFTA",
    "WOR",
    "WDR",
    "WAst",
    "WTO",
    "WStl",
    "WBlk",
    "WPF",
]

_BOX_SCORE_COLUMNS = [
    "T1_Score",
    "T1_FGM",
    "T1_FGA",
    "T1_FGM3",
    "T1_FGA3",
    "T1_FTM",
    "T1_FTA",
    "T1_OR",
    "T1_DR",
    "T1_Ast",
    "T1_TO",
    "T1_Stl",
    "T1_Blk",
    "T1_PF",
    "T2_Score",
    "T2_FGM",
    "T2_FGA",
    "T2_FGM3",
    "T2_FGA3",
    "T2_FTM",
    "T2_FTA",
    "T2_OR",
    "T2_DR",
    "T2_Ast",
    "T2_TO",
    "T2_Stl",
    "T2_Blk",
    "T2_PF",
    "PointDiff",
]


@dataclass
class Modeh7FeatureTables:
    season_stats_t1: pd.DataFrame
    season_stats_t2: pd.DataFrame
    seeds_t1: pd.DataFrame
    seeds_t2: pd.DataFrame
    elo_t1: pd.DataFrame
    elo_t2: pd.DataFrame
    quality_t1: pd.DataFrame
    quality_t2: pd.DataFrame


@dataclass
class Modeh7Ensemble:
    models: dict[int, Any]
    calibrators: dict[str, Any]
    feature_columns: tuple[str, ...]
    clip_margin: float
    prob_clip: tuple[float, float]
    calibration_mode: str
    backend: str


def _read_modeh7_inputs(start_season: int = 2003) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    men_regular = pd.read_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv")
    men_tourney = pd.read_csv(DATA_DIR / "MNCAATourneyDetailedResults.csv")
    men_seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")

    women_regular = pd.read_csv(DATA_DIR / "WRegularSeasonDetailedResults.csv")
    women_tourney = pd.read_csv(DATA_DIR / "WNCAATourneyDetailedResults.csv")
    women_seeds = pd.read_csv(DATA_DIR / "WNCAATourneySeeds.csv")

    regular = pd.concat([men_regular, women_regular], ignore_index=True)
    tourney = pd.concat([men_tourney, women_tourney], ignore_index=True)
    seeds = pd.concat([men_seeds, women_seeds], ignore_index=True)

    regular = regular.loc[regular["Season"] >= start_season].copy()
    tourney = tourney.loc[tourney["Season"] >= start_season].copy()
    seeds = seeds.loc[seeds["Season"] >= start_season].copy()
    return regular, tourney, seeds


def prepare_modeh7_results(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the notebook's symmetric doubled game representation."""
    frame = df[_BASE_RESULT_COLUMNS].copy()
    overtime_factor = (40.0 + 5.0 * frame["NumOT"]) / 40.0
    for column in _OT_ADJUST_COLUMNS:
        frame[column] = frame[column] / overtime_factor

    swapped = frame.copy()
    frame.columns = [col.replace("W", "T1_").replace("L", "T2_") for col in frame.columns]
    swapped.columns = [col.replace("L", "T1_").replace("W", "T2_") for col in swapped.columns]

    output = pd.concat([frame, swapped], ignore_index=True)
    output["PointDiff"] = output["T1_Score"] - output["T2_Score"]
    output["win"] = (output["PointDiff"] > 0).astype(int)
    output["men_women"] = (output["T1_TeamID"] < 3000).astype(int)
    return output


def _build_seed_tables(seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = seeds.copy()
    seeds["seed"] = seeds["Seed"].map(lambda value: int(str(value)[1:3]))
    seeds_t1 = seeds[["Season", "TeamID", "seed"]].rename(
        columns={"TeamID": "T1_TeamID", "seed": "T1_seed"}
    )
    seeds_t2 = seeds[["Season", "TeamID", "seed"]].rename(
        columns={"TeamID": "T2_TeamID", "seed": "T2_seed"}
    )
    return seeds_t1, seeds_t2


def _build_season_stat_tables(regular_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_stats = regular_data.groupby(["Season", "T1_TeamID"])[_BOX_SCORE_COLUMNS].mean().reset_index()

    stats_t1 = season_stats.copy()
    stats_t1.columns = [
        "T1_avg_" + col.replace("T1_", "").replace("T2_", "opponent_")
        for col in stats_t1.columns
    ]
    stats_t1 = stats_t1.rename(columns={"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID"})

    stats_t2 = season_stats.copy()
    stats_t2.columns = [
        "T2_avg_" + col.replace("T1_", "").replace("T2_", "opponent_")
        for col in stats_t2.columns
    ]
    stats_t2 = stats_t2.rename(columns={"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID"})
    return stats_t1, stats_t2


def _expected_elo(elo_a: float, elo_b: float, elo_width: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / elo_width))


def _build_elo_tables(
    regular_data: pd.DataFrame,
    seasons: list[int],
    *,
    base_elo: float = 1000.0,
    elo_width: float = 400.0,
    k_factor: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    elo_rows = []
    winners = regular_data.loc[regular_data["win"] == 1].reset_index(drop=True)

    for season in seasons:
        season_games = winners.loc[winners["Season"] == season].reset_index(drop=True)
        teams = sorted(set(season_games["T1_TeamID"]).union(set(season_games["T2_TeamID"])))
        elo = {int(team_id): float(base_elo) for team_id in teams}

        for row in season_games.itertuples(index=False):
            winner = int(row.T1_TeamID)
            loser = int(row.T2_TeamID)
            expected_win = _expected_elo(elo[winner], elo[loser], elo_width)
            delta = k_factor * (1.0 - expected_win)
            elo[winner] += delta
            elo[loser] -= delta

        for team_id, rating in elo.items():
            elo_rows.append({"Season": int(season), "TeamID": int(team_id), "elo": float(rating)})

    elos = pd.DataFrame(elo_rows)
    elos_t1 = elos.rename(columns={"TeamID": "T1_TeamID", "elo": "T1_elo"})
    elos_t2 = elos.rename(columns={"TeamID": "T2_TeamID", "elo": "T2_elo"})
    return elos_t1, elos_t2


def _build_quality_tables(
    regular_data: pd.DataFrame,
    seeds_t1: pd.DataFrame,
    seeds_t2: pd.DataFrame,
    seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regular = regular_data.copy()
    regular["ST1"] = regular.apply(
        lambda row: f"{int(row['Season'])}/{int(row['T1_TeamID'])}",
        axis=1,
    )
    regular["ST2"] = regular.apply(
        lambda row: f"{int(row['Season'])}/{int(row['T2_TeamID'])}",
        axis=1,
    )

    seed_st1 = seeds_t1.copy()
    seed_st2 = seeds_t2.copy()
    seed_st1["ST1"] = seed_st1.apply(
        lambda row: f"{int(row['Season'])}/{int(row['T1_TeamID'])}",
        axis=1,
    )
    seed_st2["ST2"] = seed_st2.apply(
        lambda row: f"{int(row['Season'])}/{int(row['T2_TeamID'])}",
        axis=1,
    )

    tournament_team_keys = set(seed_st1["ST1"]).union(set(seed_st2["ST2"]))
    tournament_team_keys = tournament_team_keys.union(
        set(
            regular.loc[
                (regular["PointDiff"] > 0) & (regular["ST2"].isin(tournament_team_keys)),
                "ST1",
            ]
        )
    )

    design_data = regular.loc[
        regular["ST1"].isin(tournament_team_keys) | regular["ST2"].isin(tournament_team_keys)
    ].copy()
    design_data["T1_TeamID"] = design_data["T1_TeamID"].astype(str)
    design_data["T2_TeamID"] = design_data["T2_TeamID"].astype(str)
    design_data.loc[~design_data["ST1"].isin(tournament_team_keys), "T1_TeamID"] = "0000"
    design_data.loc[~design_data["ST2"].isin(tournament_team_keys), "T2_TeamID"] = "0000"

    quality_rows = []
    for season in seasons:
        for men_women in (0, 1):
            if men_women == 0 and season < 2010:
                continue
            subset = design_data.loc[
                (design_data["Season"] == season) & (design_data["men_women"] == men_women)
            ].copy()
            if subset.empty:
                continue

            glm = sm.GLM.from_formula(
                formula="PointDiff~-1+T1_TeamID+T2_TeamID",
                data=subset,
                family=sm.families.Gaussian(),
            ).fit()

            quality = pd.DataFrame(glm.params).reset_index()
            quality.columns = ["TeamID", "quality"]
            quality["Season"] = int(season)
            quality = quality.loc[quality["TeamID"].str.contains("T1_")].reset_index(drop=True)
            quality["TeamID"] = quality["TeamID"].str.extract(r"(\d+)")[0].astype(int)
            quality_rows.append(quality[["Season", "TeamID", "quality"]])

    if not quality_rows:
        raise ValueError("Modeh7 quality fit produced no rows.")

    quality_all = pd.concat(quality_rows, ignore_index=True)
    quality_t1 = quality_all.rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"})
    quality_t2 = quality_all.rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"})
    return quality_t1, quality_t2


def build_modeh7_training_frame(
    *,
    start_season: int = 2003,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> tuple[pd.DataFrame, Modeh7FeatureTables]:
    """Build the notebook-style tournament training frame and reusable feature tables."""
    cache_path = MODEH7_CACHE_PATH
    if use_cache and cache_path.exists() and not refresh_cache:
        payload = pickle.loads(cache_path.read_bytes())
        return payload["training_frame"], payload["feature_tables"]

    regular_results, tourney_results, seeds = _read_modeh7_inputs(start_season)
    regular_data = prepare_modeh7_results(regular_results)
    tourney_data = prepare_modeh7_results(tourney_results)
    seasons = sorted(int(season) for season in seeds["Season"].unique())

    seeds_t1, seeds_t2 = _build_seed_tables(seeds)
    stats_t1, stats_t2 = _build_season_stat_tables(regular_data)
    elo_t1, elo_t2 = _build_elo_tables(regular_data, seasons)
    quality_t1, quality_t2 = _build_quality_tables(regular_data, seeds_t1, seeds_t2, seasons)

    feature_tables = Modeh7FeatureTables(
        season_stats_t1=stats_t1,
        season_stats_t2=stats_t2,
        seeds_t1=seeds_t1,
        seeds_t2=seeds_t2,
        elo_t1=elo_t1,
        elo_t2=elo_t2,
        quality_t1=quality_t1,
        quality_t2=quality_t2,
    )

    training_frame = _merge_modeh7_features(
        tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]],
        feature_tables,
    )

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(
            pickle.dumps({"training_frame": training_frame, "feature_tables": feature_tables})
        )

    return training_frame, feature_tables


def _merge_modeh7_features(frame: pd.DataFrame, feature_tables: Modeh7FeatureTables) -> pd.DataFrame:
    merged = frame.copy()
    merged = merged.merge(feature_tables.season_stats_t1, on=["Season", "T1_TeamID"], how="left")
    merged = merged.merge(feature_tables.season_stats_t2, on=["Season", "T2_TeamID"], how="left")
    merged = merged.merge(feature_tables.seeds_t1, on=["Season", "T1_TeamID"], how="left")
    merged = merged.merge(feature_tables.seeds_t2, on=["Season", "T2_TeamID"], how="left")
    merged = merged.merge(feature_tables.quality_t1, on=["Season", "T1_TeamID"], how="left")
    merged = merged.merge(feature_tables.quality_t2, on=["Season", "T2_TeamID"], how="left")
    merged = merged.merge(feature_tables.elo_t1, on=["Season", "T1_TeamID"], how="left")
    merged = merged.merge(feature_tables.elo_t2, on=["Season", "T2_TeamID"], how="left")
    merged["Seed_diff"] = merged["T2_seed"] - merged["T1_seed"]
    merged["elo_diff"] = merged["T1_elo"] - merged["T2_elo"]
    return merged


def build_modeh7_submission_frame(
    ids: pd.Series,
    feature_tables: Modeh7FeatureTables,
) -> pd.DataFrame:
    parsed = parse_submission_ids(ids)
    frame = pd.DataFrame(
        {
            "ID": parsed["ID"],
            "Season": parsed["Season"],
            "T1_TeamID": parsed["TeamLowID"],
            "T2_TeamID": parsed["TeamHighID"],
        }
    )
    frame["men_women"] = (frame["T1_TeamID"] < 3000).astype(int)
    return _merge_modeh7_features(frame, feature_tables)


def _prepare_xgboost_runtime_env(env: dict[str, str] | None = None) -> dict[str, str]:
    target = os.environ if env is None else env
    libomp_candidates = [
        "/opt/homebrew/opt/libomp/lib",
        "/usr/local/opt/libomp/lib",
    ]
    for env_key in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        existing = target.get(env_key, "")
        parts = [part for part in existing.split(":") if part]
        for candidate in libomp_candidates:
            if os.path.isdir(candidate) and candidate not in parts:
                parts.insert(0, candidate)
        if parts:
            target[env_key] = ":".join(parts)
    target.setdefault("OMP_NUM_THREADS", "1")
    target.setdefault("OPENBLAS_NUM_THREADS", "1")
    target.setdefault("MKL_NUM_THREADS", "1")
    target.setdefault("NUMEXPR_NUM_THREADS", "1")
    target.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    target.setdefault("BLIS_NUM_THREADS", "1")
    target.setdefault("KMP_BLOCKTIME", "0")
    return target


@lru_cache(maxsize=1)
def _xgboost_probe_error() -> str | None:
    probe_code = """
import numpy as np
import xgboost as xgb

X = np.array(
    [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [3.0, 2.0, 1.0],
    ],
    dtype=float,
)
y = np.array([0.0, 1.0, -1.0, 2.0], dtype=float)
dtrain = xgb.DMatrix(X, label=y)
xgb.train(
    params={
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": 2,
        "eta": 0.1,
        "nthread": 1,
    },
    dtrain=dtrain,
    num_boost_round=2,
)
"""
    env = _prepare_xgboost_runtime_env(dict(os.environ))
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            env=env,
            timeout=MODEH7_XGB_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except Exception as exc:
        return f"xgboost runtime probe failed: {type(exc).__name__}: {exc}"

    combined_output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part and part.strip()
    )
    if result.returncode == 0:
        return None

    if "Can't open SHM2 failed" in combined_output or "OMP: Error #179" in combined_output:
        return "xgboost/libomp failed to initialize shared memory in this environment"
    if combined_output:
        return combined_output.splitlines()[-1]
    return f"xgboost runtime probe exited with code {result.returncode}"


def _ensure_xgboost_runtime() -> None:
    probe_error = _xgboost_probe_error()
    if probe_error is not None:
        raise RuntimeError(f"XGBoost backend unavailable: {probe_error}")


def _train_xgb_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    *,
    params: dict | None = None,
    num_rounds: int = MODEH7_NUM_ROUNDS,
):
    _ensure_xgboost_runtime()
    _prepare_xgboost_runtime_env()
    import xgboost as xgb

    booster_params = dict(MODEH7_XGB_PARAMS if params is None else params)
    dtrain = xgb.DMatrix(train_x.values, label=train_y.values)
    return xgb.train(params=booster_params, dtrain=dtrain, num_boost_round=num_rounds)


def _predict_margin(model: Any, features: pd.DataFrame) -> np.ndarray:
    _ensure_xgboost_runtime()
    _prepare_xgboost_runtime_env()
    import xgboost as xgb

    dmatrix = xgb.DMatrix(features.values)
    return model.predict(dmatrix)


def _fit_single_calibrator(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    clip_margin: float,
    degree: int,
) -> Any:
    order = np.argsort(predictions)
    x = np.clip(predictions[order], -clip_margin, clip_margin)
    y = labels[order].astype(float)
    unique_count = np.unique(x).size
    spline_degree = min(degree, max(1, unique_count - 1))
    return UnivariateSpline(x, y, k=spline_degree)


def fit_modeh7_calibrators(
    oof_frame: pd.DataFrame,
    *,
    calibration_mode: str = "by_gender",
    clip_margin: float = MODEH7_CLIP_MARGIN,
    degree: int = 5,
) -> dict[str, Any]:
    if calibration_mode not in {"combined", "by_gender"}:
        raise ValueError(f"Unknown calibration mode: {calibration_mode}")

    calibrators: dict[str, Any] = {}
    if calibration_mode == "combined":
        calibrators["all"] = _fit_single_calibrator(
            oof_frame["pred_margin"].to_numpy(),
            oof_frame["label"].to_numpy(),
            clip_margin=clip_margin,
            degree=degree,
        )
        return calibrators

    for men_women, label in [(1, "men"), (0, "women")]:
        subset = oof_frame.loc[oof_frame["men_women"] == men_women].copy()
        calibrators[label] = _fit_single_calibrator(
            subset["pred_margin"].to_numpy(),
            subset["label"].to_numpy(),
            clip_margin=clip_margin,
            degree=degree,
        )
    return calibrators


def apply_modeh7_calibration(
    predictions: np.ndarray,
    men_women: pd.Series,
    calibrators: dict[str, Any],
    *,
    calibration_mode: str,
    clip_margin: float,
    prob_clip: tuple[float, float] = MODEH7_PROB_CLIP,
) -> np.ndarray:
    clipped = np.clip(predictions, -clip_margin, clip_margin)
    if calibration_mode == "combined":
        probabilities = calibrators["all"](clipped)
        return np.clip(probabilities, prob_clip[0], prob_clip[1])

    output = np.zeros(len(clipped), dtype=float)
    men_mask = men_women.to_numpy().astype(int) == 1
    women_mask = ~men_mask
    output[men_mask] = calibrators["men"](clipped[men_mask])
    output[women_mask] = calibrators["women"](clipped[women_mask])
    return np.clip(output, prob_clip[0], prob_clip[1])


def fit_modeh7_ensemble(
    training_frame: pd.DataFrame,
    *,
    validation: str = "loso",
    eval_seasons: list[int] | None = None,
    calibration_mode: str = "by_gender",
    params: dict | None = None,
    num_rounds: int = MODEH7_NUM_ROUNDS,
) -> tuple[Modeh7Ensemble, pd.DataFrame]:
    """Fit the notebook-style ensemble and return OOF predictions."""
    if validation not in {"loso", "rolling"}:
        raise ValueError(f"Unknown validation mode: {validation}")

    seasons = sorted(int(season) for season in training_frame["Season"].unique())
    if eval_seasons is not None:
        eval_set = set(int(season) for season in eval_seasons)
        seasons = [season for season in seasons if season in eval_set]

    models: dict[int, Any] = {}
    oof_rows = []
    feature_columns = list(MODEH7_FEATURE_COLUMNS)
    _ensure_xgboost_runtime()

    for season in seasons:
        if validation == "loso":
            train = training_frame.loc[training_frame["Season"] != season].copy()
        else:
            train = training_frame.loc[training_frame["Season"] < season].copy()
        valid = training_frame.loc[training_frame["Season"] == season].copy()
        if train.empty or valid.empty:
            continue

        model = _train_xgb_model(
            train[feature_columns],
            train["PointDiff"],
            params=params,
            num_rounds=num_rounds,
        )
        models[int(season)] = model
        valid_pred = _predict_margin(model, valid[feature_columns])
        fold_frame = valid[["Season", "men_women", "PointDiff", "win"]].copy()
        fold_frame["pred_margin"] = valid_pred
        fold_frame["label"] = valid["win"].astype(int).to_numpy()
        oof_rows.append(fold_frame)

    if not oof_rows:
        raise ValueError("Modeh7 ensemble fit produced no out-of-fold rows.")

    oof_frame = pd.concat(oof_rows, ignore_index=True)
    calibrators = fit_modeh7_calibrators(
        oof_frame,
        calibration_mode=calibration_mode,
    )
    ensemble = Modeh7Ensemble(
        models=models,
        calibrators=calibrators,
        feature_columns=tuple(feature_columns),
        clip_margin=MODEH7_CLIP_MARGIN,
        prob_clip=MODEH7_PROB_CLIP,
        calibration_mode=calibration_mode,
        backend=MODEH7_BACKEND,
    )
    return ensemble, oof_frame


def predict_modeh7_probabilities(
    feature_frame: pd.DataFrame,
    ensemble: Modeh7Ensemble,
    *,
    aggressiveness: float = MODEH7_DEFAULT_AGGRESSIVENESS,
) -> np.ndarray:
    margins = []
    features = feature_frame[list(ensemble.feature_columns)]
    for model in ensemble.models.values():
        margin_pred = _predict_margin(model, features) * aggressiveness
        margins.append(margin_pred)
    mean_margin = np.mean(np.vstack(margins), axis=0)
    return apply_modeh7_calibration(
        mean_margin,
        feature_frame["men_women"],
        ensemble.calibrators,
        calibration_mode=ensemble.calibration_mode,
        clip_margin=ensemble.clip_margin,
        prob_clip=ensemble.prob_clip,
    )


def score_modeh7_oof(
    oof_frame: pd.DataFrame,
    ensemble: Modeh7Ensemble,
    *,
    aggressiveness: float = MODEH7_DEFAULT_AGGRESSIVENESS,
) -> dict[str, Any]:
    probabilities = apply_modeh7_calibration(
        oof_frame["pred_margin"].to_numpy() * aggressiveness,
        oof_frame["men_women"],
        ensemble.calibrators,
        calibration_mode=ensemble.calibration_mode,
        clip_margin=ensemble.clip_margin,
        prob_clip=ensemble.prob_clip,
    )
    scored = oof_frame.copy()
    scored["probability"] = probabilities

    summary = {
        "overall_brier": float(brier_score_loss(scored["label"], scored["probability"])),
        "men_brier": float(
            brier_score_loss(
                scored.loc[scored["men_women"] == 1, "label"],
                scored.loc[scored["men_women"] == 1, "probability"],
            )
        ),
        "women_brier": float(
            brier_score_loss(
                scored.loc[scored["men_women"] == 0, "label"],
                scored.loc[scored["men_women"] == 0, "probability"],
            )
        ),
        "season_metrics": [],
    }
    for season, subset in scored.groupby("Season", sort=True):
        summary["season_metrics"].append(
            {
                "season": int(season),
                "games": int(len(subset)),
                "overall_brier": float(brier_score_loss(subset["label"], subset["probability"])),
            }
        )
    return summary


def generate_modeh7_submission(
    ids: pd.Series,
    *,
    start_season: int = 2003,
    validation: str = MODEH7_DEFAULT_VALIDATION,
    eval_seasons: list[int] | None = None,
    calibration_mode: str = MODEH7_DEFAULT_CALIBRATION_MODE,
    params: dict | None = None,
    num_rounds: int = MODEH7_NUM_ROUNDS,
    use_cache: bool = True,
    refresh_cache: bool = False,
    aggressiveness: float = MODEH7_DEFAULT_AGGRESSIVENESS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train the notebook-style ensemble and generate probabilities for submission IDs."""
    submission_ids = pd.Series(ids, copy=False).reset_index(drop=True)
    training_frame, feature_tables = build_modeh7_training_frame(
        start_season=start_season,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
    ensemble, oof_frame = fit_modeh7_ensemble(
        training_frame,
        validation=validation,
        eval_seasons=eval_seasons,
        calibration_mode=calibration_mode,
        params=params,
        num_rounds=num_rounds,
    )
    prediction_frame = build_modeh7_submission_frame(submission_ids, feature_tables)
    probabilities = predict_modeh7_probabilities(
        prediction_frame,
        ensemble,
        aggressiveness=aggressiveness,
    )

    summary = score_modeh7_oof(oof_frame, ensemble, aggressiveness=aggressiveness)
    summary.update(
        {
            "validation": validation,
            "calibration_mode": calibration_mode,
            "backend": ensemble.backend,
            "num_models": int(len(ensemble.models)),
            "prediction_rows": int(len(submission_ids)),
            "aggressiveness": float(aggressiveness),
        }
    )
    if eval_seasons is not None:
        summary["eval_seasons"] = [int(season) for season in eval_seasons]

    submission = pd.DataFrame({"ID": submission_ids, "Pred": probabilities})
    return submission, summary
