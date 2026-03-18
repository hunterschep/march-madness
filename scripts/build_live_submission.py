from __future__ import annotations

"""Assemble the competition submission from the modeh7 baseline and live men overlays."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.config import DATA_DIR, INPUT_DIR, OUTPUT_DIR, ensure_output_dirs, get_side
from mmmania.features import get_tournament_seeds
from mmmania.modeh7 import (
    MODEH7_DEFAULT_AGGRESSIVENESS,
    MODEH7_DEFAULT_CALIBRATION_MODE,
    MODEH7_DEFAULT_VALIDATION,
    generate_modeh7_submission,
)
from mmmania.modeling import parse_submission_ids
from mmmania.silver import (
    load_men_silver_probabilities,
    silver_direct_share_for_round,
    silver_observation_weight_for_round,
)
from mmmania.tournament import fit_pair_probability_ratings, pair_probability_from_ratings


SEASON = 2026
MARKET_PATH = OUTPUT_DIR / "markets" / "market_consensus.csv"
BPI_PATH = INPUT_DIR / "odds" / "men_bpi_probabilities.csv"
PRIOR_PATH = OUTPUT_DIR / "submissions" / "prior_submission_2026.csv"
PRIOR_SUMMARY_PATH = OUTPUT_DIR / "reports" / "prior_submission_summary_2026.json"
OUTPUT_PATH = OUTPUT_DIR / "submissions" / "live_submission_2026.csv"
SUMMARY_PATH = OUTPUT_DIR / "reports" / "live_submission_summary_2026.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior-path",
        type=Path,
        default=PRIOR_PATH,
        help="Path to the model-only prior submission CSV.",
    )
    parser.add_argument(
        "--refresh-prior",
        action="store_true",
        help="Rebuild the modeh7 prior instead of reusing the saved CSV.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild the cached modeh7 feature tables while generating an inline prior.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the modeh7 feature-table cache while generating an inline prior.",
    )
    return parser.parse_args()


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


def _load_or_build_prior(
    sample_ids: pd.Series,
    *,
    prior_path: Path,
    refresh_prior: bool,
    use_cache: bool,
    refresh_cache: bool,
) -> tuple[pd.DataFrame, dict]:
    if prior_path.exists() and not refresh_prior:
        prior = pd.read_csv(prior_path)[["ID", "Pred"]].copy()
        summary: dict = {
            "source": "file",
            "path": str(prior_path),
        }
        if PRIOR_SUMMARY_PATH.exists():
            try:
                summary["saved_summary"] = json.loads(PRIOR_SUMMARY_PATH.read_text())
            except json.JSONDecodeError:
                summary["saved_summary"] = {"error": f"Could not parse {PRIOR_SUMMARY_PATH.name}"}
        saved = summary.get("saved_summary", {})
        if saved.get("backend") != "xgboost":
            raise ValueError(
                "The saved prior was not produced with backend=xgboost. "
                "Re-run python scripts/build_prior_submission.py or use --refresh-prior."
            )
    else:
        prior, model_summary = generate_modeh7_submission(
            sample_ids,
            validation=MODEH7_DEFAULT_VALIDATION,
            calibration_mode=MODEH7_DEFAULT_CALIBRATION_MODE,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            aggressiveness=MODEH7_DEFAULT_AGGRESSIVENESS,
        )
        prior_to_write = (
            prior.set_index("ID")
            .reindex(sample_ids)
            .reset_index()
        )
        prior_to_write["Pred"] = prior_to_write["Pred"].round(6)
        prior_path.parent.mkdir(parents=True, exist_ok=True)
        prior_to_write.to_csv(prior_path, index=False)
        PRIOR_SUMMARY_PATH.write_text(
            json.dumps(
                {
                    "season": SEASON,
                    "output_path": str(prior_path),
                    **model_summary,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        summary = {
            "source": "inline_rebuild",
            "path": str(prior_path),
            "saved_summary": model_summary,
        }

    prior = (
        prior.set_index("ID")
        .reindex(sample_ids)
        .reset_index()
    )
    if prior["Pred"].isna().any():
        missing_count = int(prior["Pred"].isna().sum())
        raise ValueError(f"Prior submission is missing {missing_count} IDs needed for the live build.")
    return prior, summary


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


def _men_tournament_layer_from_weights(
    model_lookup: pd.DataFrame,
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
    observations = model_lookup.rename(
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
    men_layer = model_lookup.copy()
    men_layer["latent_pred"] = men_layer.apply(
        lambda row: pair_probability_from_ratings(row["TeamLowID"], row["TeamHighID"], ratings),
        axis=1,
    )
    men_layer["Pred"] = latent_blend * men_layer["latent_pred"] + (1.0 - latent_blend) * men_layer["Pred"]
    return men_layer


def _tune_men_tournament_layer_weights(
    model_lookup: pd.DataFrame,
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
            model_lookup,
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
                            model_lookup,
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
                        score = 0.0 if not score_components else sum(score_components)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_weights = weights
                            best_layer = layer

    assert best_weights is not None and best_layer is not None
    best_weights["proxy_external_score"] = best_score
    return best_weights, best_layer


def _men_tournament_probability_layer(
    model_lookup: pd.DataFrame,
    market: pd.DataFrame,
    tournament_team_ids: set[int],
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Use current first-round info to fit coherent men-only tournament ratings."""
    bpi = _load_bpi_probabilities()
    silver = _load_silver_probabilities()
    weights, men_layer = _tune_men_tournament_layer_weights(
        model_lookup,
        market,
        bpi,
        silver,
        tournament_team_ids,
    )
    return men_layer, weights, silver


def _predict_side(
    sample_pairs: pd.DataFrame,
    prior_predictions: pd.DataFrame,
    side_code: str,
) -> tuple[pd.DataFrame, dict]:
    side = get_side(side_code)
    side_mask = sample_pairs["TeamLowID"] >= 3000 if side_code == "W" else sample_pairs["TeamLowID"] < 3000
    side_pairs = sample_pairs.loc[side_mask, ["ID", "Season", "TeamLowID", "TeamHighID"]].copy()

    side_pred = side_pairs.merge(prior_predictions, on="ID", how="left")
    if side_pred["Pred"].isna().any():
        missing_count = int(side_pred["Pred"].isna().sum())
        raise ValueError(f"Modeh7 prior is missing {missing_count} {side.label} rows.")

    tournament_team_ids = set(
        get_tournament_seeds(side).loc[lambda frame: frame["Season"] == SEASON, "TeamID"]
    )
    tournament_mask = side_pairs["TeamLowID"].isin(tournament_team_ids) & side_pairs["TeamHighID"].isin(
        tournament_team_ids
    )

    summary = {
        "side": side.label,
        "base_model": "modeh7",
        "base_rows": int(len(side_pred)),
        "latent_layer_rows": 0,
        "market_overrides": 0,
    }

    if side_code == "M":
        model_lookup = side_pred.loc[tournament_mask, ["ID", "TeamLowID", "TeamHighID", "Pred"]].copy()

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

        model_lookup, men_layer_weights, silver = _men_tournament_probability_layer(
            model_lookup,
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
                    model_lookup[["ID", "Pred"]].rename(columns={"Pred": "model_pred"}),
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
                model_lookup = model_lookup.merge(
                    silver_only[["ID", "Pred"]].rename(columns={"Pred": "silver_pred"}),
                    on="ID",
                    how="left",
                )
                model_lookup["Pred"] = model_lookup["silver_pred"].fillna(model_lookup["Pred"])
                model_lookup = model_lookup.drop(columns=["silver_pred"])
                summary["silver_only_overrides"] = int(len(silver_only))

        side_pred = side_pred.set_index("ID")
        side_pred.loc[model_lookup["ID"], "Pred"] = model_lookup.set_index("ID")["Pred"]
        side_pred = side_pred.reset_index()
        summary["latent_layer_rows"] = int(len(model_lookup))

        if not market.empty:
            if not silver.empty:
                market = market.merge(
                    silver[["ID", "p_low_silver", "silver_round"]],
                    on="ID",
                    how="left",
                )
            market = market.merge(
                model_lookup[["ID", "Pred"]].rename(columns={"Pred": "model_pred"}),
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
    return side_pred[["ID", "Pred"]], summary


def main() -> None:
    ensure_output_dirs()
    args = _parse_args()

    sample = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
    sample_pairs = parse_submission_ids(sample["ID"])
    prior_predictions, prior_summary = _load_or_build_prior(
        sample["ID"],
        prior_path=args.prior_path,
        refresh_prior=args.refresh_prior,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )

    predictions = []
    summaries = []
    for side_code in ("M", "W"):
        side_pred, summary = _predict_side(sample_pairs, prior_predictions, side_code)
        predictions.append(side_pred)
        summaries.append(summary)

    submission = (
        pd.concat(predictions, ignore_index=True)
        .set_index("ID")
        .reindex(sample["ID"])
        .reset_index()
    )
    submission["Pred"] = submission["Pred"].round(6)
    submission.to_csv(OUTPUT_PATH, index=False)

    payload = {
        "season": SEASON,
        "base_model": "modeh7",
        "prior": prior_summary,
        "summaries": summaries,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
