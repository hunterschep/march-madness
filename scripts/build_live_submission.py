from __future__ import annotations

"""Assemble the current competition-ready submission with model and market layers."""


import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.config import DATA_DIR, INPUT_DIR, OUTPUT_DIR, ensure_output_dirs, get_side
from mmmania.features import build_team_season_features, get_tournament_seeds
from mmmania.modeling import (
    PRIOR_SPEC,
    SEEDED_SPEC,
    apply_seed_matchup_prior,
    build_pair_round_lookup,
    build_pair_dataset,
    build_pair_rows_for_submission,
    calibrate_probabilities,
    fit_final_model,
    load_tuned_model,
    parse_submission_ids,
    round_bucket_from_round_number,
    seed_gap_bucket,
    tune_model,
)
from mmmania.silver import (
    load_men_silver_probabilities,
    silver_direct_share_for_round,
    silver_observation_weight_for_round,
)
from mmmania.tournament import fit_pair_probability_ratings, pair_probability_from_ratings


SEASON = 2026
BACKTEST_PATH = OUTPUT_DIR / "backtests" / "rolling_backtests.json"
MARKET_PATH = OUTPUT_DIR / "markets" / "market_consensus.csv"
BPI_PATH = INPUT_DIR / "odds" / "men_bpi_probabilities.csv"
OUTPUT_PATH = OUTPUT_DIR / "submissions" / "live_submission_2026.csv"
SUMMARY_PATH = OUTPUT_DIR / "reports" / "live_submission_summary_2026.json"


def _market_weight(row: pd.Series) -> float:
    """Weight direct moneyline consensus heavily on posted games."""
    return 0.9 if row.get("moneyline_count", 0) > 0 else 0.0


def _weighted_mse(errors: pd.Series, weights: pd.Series | None = None) -> float:
    if errors.empty:
        return 0.0
    if weights is None:
        return float((errors**2).mean())
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return float((errors**2).mean())
    return float(((errors**2) * weights).sum() / total_weight)


def _load_bpi_probabilities() -> pd.DataFrame:
    if not BPI_PATH.exists():
        return pd.DataFrame(columns=["ID", "team_low_id", "team_high_id", "p_low_bpi", "weight"])

    bpi = pd.read_csv(BPI_PATH)
    required = {"team_low_id", "team_high_id", "p_low_bpi"}
    missing = required - set(bpi.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing BPI columns: {missing_list}")

    bpi = bpi.copy()
    if "weight" not in bpi.columns:
        bpi["weight"] = 1.0
    bpi["ID"] = bpi.apply(
        lambda row: f"{SEASON}_{int(row['team_low_id'])}_{int(row['team_high_id'])}",
        axis=1,
    )
    return bpi[["ID", "team_low_id", "team_high_id", "p_low_bpi", "weight"]]


def _load_silver_probabilities() -> pd.DataFrame:
    silver = load_men_silver_probabilities(SEASON)
    if silver.empty:
        return pd.DataFrame(
            columns=[
                "ID",
                "team_low_id",
                "team_high_id",
                "p_low_silver",
                "silver_round",
                "silver_source_ts",
            ]
        )

    silver = silver.copy()
    silver["ID"] = silver.apply(
        lambda row: f"{SEASON}_{int(row['team_low_id'])}_{int(row['team_high_id'])}",
        axis=1,
    )
    return silver


def _apply_tuned_calibration(
    frame: pd.DataFrame,
    probabilities: pd.Series,
    tuned_model,
    pair_round_lookup: dict[str, int] | None = None,
) -> pd.Series:
    round_buckets = None
    if pair_round_lookup is not None:
        round_buckets = frame.apply(
            lambda row: round_bucket_from_round_number(
                pair_round_lookup.get(f"{int(row['TeamLowID'])}_{int(row['TeamHighID'])}")
            ),
            axis=1,
        )

    seed_gap_buckets = None
    if "seed_gap" in frame.columns:
        seed_gap_buckets = frame["seed_gap"].map(seed_gap_bucket)

    return calibrate_probabilities(
        probabilities,
        global_scale=tuned_model.global_scale,
        round_buckets=round_buckets,
        round_bucket_scales=tuned_model.round_bucket_scales,
        seed_gap_buckets=seed_gap_buckets,
        seed_gap_bucket_scales=tuned_model.seed_gap_bucket_scales,
    )


def _men_tournament_layer_from_weights(
    seeded_lookup: pd.DataFrame,
    market: pd.DataFrame,
    bpi: pd.DataFrame,
    silver: pd.DataFrame,
    tournament_team_ids: set[int],
    *,
    market_observation_weight: float,
    bpi_observation_weight: float,
    silver_first_four_observation_weight: float,
    silver_round64_observation_weight: float,
    latent_blend: float,
) -> pd.DataFrame:
    observations = seeded_lookup.rename(
        columns={
            "TeamLowID": "team_low_id",
            "TeamHighID": "team_high_id",
            "Pred": "probability",
        }
    )[["team_low_id", "team_high_id", "probability"]].copy()
    observations["weight"] = 1.0

    if not market.empty:
        market_rows = market.rename(columns={"p_low_consensus": "probability"})[
            ["team_low_id", "team_high_id", "probability"]
        ].copy()
        market_rows["weight"] = market_observation_weight
        observations = pd.concat([observations, market_rows], ignore_index=True)

    if not bpi.empty:
        bpi_rows = bpi.rename(columns={"p_low_bpi": "probability"})[
            ["team_low_id", "team_high_id", "probability", "weight"]
        ].copy()
        bpi_rows["weight"] = bpi_rows["weight"] * bpi_observation_weight
        observations = pd.concat([observations, bpi_rows], ignore_index=True)

    if not silver.empty:
        silver_rows = silver.rename(columns={"p_low_silver": "probability"})[
            ["team_low_id", "team_high_id", "probability", "silver_round"]
        ].copy()
        silver_rows["weight"] = silver_rows["silver_round"].map(
            lambda round_name: silver_observation_weight_for_round(
                round_name,
                first_four_weight=silver_first_four_observation_weight,
                round64_weight=silver_round64_observation_weight,
            )
        )
        silver_rows = silver_rows.drop(columns=["silver_round"])
        observations = pd.concat([observations, silver_rows], ignore_index=True)

    ratings = fit_pair_probability_ratings(tournament_team_ids, observations)
    men_layer = seeded_lookup.copy()
    men_layer["latent_pred"] = men_layer.apply(
        lambda row: pair_probability_from_ratings(row["TeamLowID"], row["TeamHighID"], ratings),
        axis=1,
    )
    men_layer["Pred"] = latent_blend * men_layer["latent_pred"] + (1.0 - latent_blend) * men_layer["Pred"]
    return men_layer


def _tune_men_tournament_layer_weights(
    seeded_lookup: pd.DataFrame,
    market: pd.DataFrame,
    bpi: pd.DataFrame,
    silver: pd.DataFrame,
    tournament_team_ids: set[int],
) -> tuple[dict, pd.DataFrame]:
    if market.empty and silver.empty:
        weights = {
            "market_observation_weight": 8.0,
            "bpi_observation_weight": 2.0,
            "silver_first_four_observation_weight": 6.0,
            "silver_round64_observation_weight": 14.0,
            "latent_blend": 0.75,
        }
        return weights, _men_tournament_layer_from_weights(
            seeded_lookup,
            market,
            bpi,
            silver,
            tournament_team_ids,
            **weights,
        )

    best_weights = None
    best_layer = None
    best_score = None
    market_lookup = market.set_index("ID")["p_low_consensus"] if not market.empty else pd.Series(dtype=float)
    silver_lookup = (
        silver.set_index("ID")[["p_low_silver", "silver_round"]] if not silver.empty else pd.DataFrame()
    )

    bpi_weight_grid = (1.0,) if bpi.empty else (1.0, 2.0, 3.0)
    silver_first_four_grid = (4.0, 6.0, 8.0) if not silver.empty else (0.0,)
    silver_round64_grid = (10.0, 14.0, 18.0) if not silver.empty else (0.0,)
    for market_observation_weight in (6.0, 8.0, 10.0):
        for bpi_observation_weight in bpi_weight_grid:
            for silver_first_four_observation_weight in silver_first_four_grid:
                for silver_round64_observation_weight in silver_round64_grid:
                    for latent_blend in (0.75, 0.85, 0.9):
                        weights = {
                            "market_observation_weight": market_observation_weight,
                            "bpi_observation_weight": bpi_observation_weight,
                            "silver_first_four_observation_weight": silver_first_four_observation_weight,
                            "silver_round64_observation_weight": silver_round64_observation_weight,
                            "latent_blend": latent_blend,
                        }
                        layer = _men_tournament_layer_from_weights(
                            seeded_lookup,
                            market,
                            bpi,
                            silver,
                            tournament_team_ids,
                            **weights,
                        )
                        layer_lookup = layer.set_index("ID")["Pred"]
                        score_components = []
                        if not market_lookup.empty:
                            comparison = (
                                layer_lookup.rename("layer_pred")
                                .to_frame()
                                .join(market_lookup.rename("market_pred"), how="inner")
                            )
                            if not comparison.empty:
                                score_components.append(
                                    0.60 * _weighted_mse(
                                        comparison["layer_pred"] - comparison["market_pred"]
                                    )
                                )
                        if not silver_lookup.empty:
                            comparison = (
                                layer_lookup.rename("layer_pred")
                                .to_frame()
                                .join(silver_lookup, how="inner")
                            )
                            if not comparison.empty:
                                silver_score_weights = comparison["silver_round"].map(
                                    lambda round_name: 2.0 if str(round_name) == "Round of 64" else 1.0
                                )
                                score_components.append(
                                    0.40
                                    * _weighted_mse(
                                        comparison["layer_pred"] - comparison["p_low_silver"],
                                        silver_score_weights,
                                    )
                                )
                        if not score_components:
                            score = 0.0
                        else:
                            score = sum(score_components)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_weights = weights
                            best_layer = layer

    assert best_weights is not None and best_layer is not None
    best_weights["proxy_external_score"] = best_score
    return best_weights, best_layer


def _men_tournament_probability_layer(
    seeded_lookup: pd.DataFrame,
    market: pd.DataFrame,
    tournament_team_ids: set[int],
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Use current first-round info to fit coherent men-only tournament ratings."""
    bpi = _load_bpi_probabilities()
    silver = _load_silver_probabilities()
    weights, men_layer = _tune_men_tournament_layer_weights(
        seeded_lookup,
        market,
        bpi,
        silver,
        tournament_team_ids,
    )
    return men_layer, weights, silver


def _predict_side(sample_pairs: pd.DataFrame, side_code: str) -> tuple[pd.DataFrame, dict]:
    side = get_side(side_code)
    side_mask = sample_pairs["TeamLowID"] >= 3000 if side_code == "W" else sample_pairs["TeamLowID"] < 3000
    side_pairs = sample_pairs.loc[side_mask, ["ID", "Season", "TeamLowID", "TeamHighID"]].copy()
    team_features = build_team_season_features(side)
    pair_round_lookup = build_pair_round_lookup(side_code, SEASON) if side_code in {"M", "W"} else {}

    prior_dataset = build_pair_dataset(side, team_features, PRIOR_SPEC)
    prior_tuned = load_tuned_model(BACKTEST_PATH, side.label, PRIOR_SPEC.name)
    if prior_tuned is None:
        prior_tuned, _ = tune_model(side, prior_dataset, PRIOR_SPEC)
    prior_model, _ = fit_final_model(prior_dataset, prior_tuned)

    prior_rows = build_pair_rows_for_submission(
        side=side,
        team_features=team_features,
        season=SEASON,
        spec=PRIOR_SPEC,
        ids=side_pairs["ID"],
    )
    side_pred = prior_rows[["ID"]].copy()
    prior_probabilities = pd.Series(
        prior_model.predict_proba(
            prior_rows[list(prior_tuned.feature_columns)].fillna(0.0)
        )[:, 1]
    )
    side_pred["Pred"] = _apply_tuned_calibration(
        prior_rows,
        prior_probabilities,
        prior_tuned,
        pair_round_lookup=None,
    )

    tournament_team_ids = set(get_tournament_seeds(side).loc[lambda frame: frame["Season"] == SEASON, "TeamID"])
    tournament_mask = side_pairs["TeamLowID"].isin(tournament_team_ids) & side_pairs["TeamHighID"].isin(tournament_team_ids)
    tournament_ids = side_pairs.loc[tournament_mask, "ID"]

    summary = {
        "side": side.label,
        "prior_rows": int(len(side_pred)),
        "seeded_overrides": 0,
        "market_overrides": 0,
    }

    seeded_lookup = None
    if not tournament_ids.empty:
        seeded_dataset = build_pair_dataset(side, team_features, SEEDED_SPEC)
        seeded_tuned = load_tuned_model(BACKTEST_PATH, side.label, SEEDED_SPEC.name)
        if seeded_tuned is None:
            seeded_tuned, _ = tune_model(side, seeded_dataset, SEEDED_SPEC)
        seeded_model, seed_prior_payload = fit_final_model(seeded_dataset, seeded_tuned)

        seeded_rows = build_pair_rows_for_submission(
            side=side,
            team_features=team_features,
            season=SEASON,
            spec=SEEDED_SPEC,
            ids=tournament_ids,
        )
        if seeded_tuned.use_seed_matchup_prior:
            seeded_rows = apply_seed_matchup_prior(seeded_rows, seed_prior_payload)

        seeded_lookup = seeded_rows[["ID"]].copy()
        seeded_lookup["TeamLowID"] = seeded_rows["TeamLowID"].to_numpy()
        seeded_lookup["TeamHighID"] = seeded_rows["TeamHighID"].to_numpy()
        seeded_lookup["seed_gap"] = seeded_rows["seed_gap"].to_numpy()
        seeded_probabilities = pd.Series(
            seeded_model.predict_proba(
                seeded_rows[list(seeded_tuned.feature_columns)].fillna(0.0)
            )[:, 1]
        )
        seeded_lookup["Pred"] = _apply_tuned_calibration(
            seeded_rows,
            seeded_probabilities,
            seeded_tuned,
            pair_round_lookup=pair_round_lookup,
        )

        market = pd.DataFrame()
        if MARKET_PATH.exists():
            market = pd.read_csv(MARKET_PATH)
            market = market[market["side"] == side_code].copy()
            if not market.empty:
                market["ID"] = market.apply(
                    lambda row: f"{SEASON}_{int(row['team_low_id'])}_{int(row['team_high_id'])}",
                    axis=1,
                )
                market["market_weight"] = market.apply(_market_weight, axis=1)
                market = market[market["market_weight"] > 0].copy()

        silver = pd.DataFrame()
        if side_code == "M":
            seeded_lookup, men_layer_weights, silver = _men_tournament_probability_layer(
                seeded_lookup,
                market,
                tournament_team_ids,
            )
            men_layer_weights["silver_market_direct_share_first_four"] = 0.15
            men_layer_weights["silver_market_direct_share_round64"] = 0.35
            men_layer_weights["silver_only_direct_share_first_four"] = 0.35
            men_layer_weights["silver_only_direct_share_round64"] = 0.60
            summary["men_layer_weights"] = men_layer_weights
            summary["silver_rows"] = int(len(silver))

            if not silver.empty:
                silver = silver.copy()
                silver["has_market"] = silver["ID"].isin(set(market["ID"])) if not market.empty else False

                silver_only = silver[~silver["has_market"]].copy()
                if not silver_only.empty:
                    silver_only = silver_only.merge(
                        seeded_lookup[["ID", "Pred"]].rename(columns={"Pred": "model_pred"}),
                        on="ID",
                        how="left",
                    )
                    silver_only["silver_direct_share"] = silver_only["silver_round"].map(
                        lambda round_name: silver_direct_share_for_round(
                            round_name,
                            first_four_share=0.35,
                            round64_share=0.60,
                        )
                    )
                    silver_only["Pred"] = (
                        (1.0 - silver_only["silver_direct_share"]) * silver_only["model_pred"]
                        + silver_only["silver_direct_share"] * silver_only["p_low_silver"]
                    )
                    seeded_lookup = seeded_lookup.merge(
                        silver_only[["ID", "Pred"]].rename(columns={"Pred": "silver_pred"}),
                        on="ID",
                        how="left",
                    )
                    seeded_lookup["Pred"] = seeded_lookup["silver_pred"].fillna(seeded_lookup["Pred"])
                    seeded_lookup = seeded_lookup.drop(columns=["silver_pred"])
                    summary["silver_only_overrides"] = int(len(silver_only))

        side_pred = side_pred.set_index("ID")
        side_pred.loc[seeded_lookup["ID"], "Pred"] = seeded_lookup.set_index("ID")["Pred"]
        side_pred = side_pred.reset_index()
        summary["seeded_overrides"] = int(len(seeded_lookup))

        if not market.empty:
            if not silver.empty:
                market = market.merge(
                    silver[["ID", "p_low_silver", "silver_round"]],
                    on="ID",
                    how="left",
                )
            market = market.merge(
                seeded_lookup[["ID", "Pred"]].rename(columns={"Pred": "model_pred"}),
                on="ID",
                how="left",
            )
            market = market[market["model_pred"].notna()].copy()
            external_prob = market["p_low_consensus"]
            if "p_low_silver" in market.columns:
                market["silver_direct_share"] = market["silver_round"].map(
                    lambda round_name: silver_direct_share_for_round(
                        round_name,
                        first_four_share=0.15,
                        round64_share=0.35,
                    )
                )
                market["silver_direct_share"] = market["silver_direct_share"].where(
                    market["p_low_silver"].notna(),
                    0.0,
                )
                external_prob = (
                    (1.0 - market["silver_direct_share"]) * market["p_low_consensus"]
                    + market["silver_direct_share"]
                    * market["p_low_silver"].fillna(market["p_low_consensus"])
                )
            market["Pred"] = (
                market["market_weight"] * external_prob
                + (1.0 - market["market_weight"]) * market["model_pred"]
            )
            if not market.empty:
                side_pred = side_pred.set_index("ID")
                side_pred.loc[market["ID"], "Pred"] = market.set_index("ID")["Pred"]
                side_pred = side_pred.reset_index()
                summary["market_overrides"] = int(len(market))

    side_pred["Pred"] = side_pred["Pred"].clip(0.001, 0.999)
    return side_pred, summary


def main() -> None:
    ensure_output_dirs()

    sample = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
    sample_pairs = parse_submission_ids(sample["ID"])

    predictions = []
    summaries = []
    for side_code in ("M", "W"):
        side_pred, summary = _predict_side(sample_pairs, side_code)
        predictions.append(side_pred)
        summaries.append(summary)

    submission = (
        pd.concat(predictions, ignore_index=True)
        .set_index("ID")
        .reindex(sample["ID"])
        .reset_index()
    )
    submission.to_csv(OUTPUT_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps({"season": SEASON, "summaries": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
