from __future__ import annotations

"""Export a readable board comparing men’s posted-game market and model probabilities."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.config import DATA_DIR, INPUT_DIR, OUTPUT_DIR, ensure_output_dirs, get_side
from mmmania.features import build_team_season_features
from mmmania.modeling import (
    SEEDED_SPEC,
    apply_seed_matchup_prior,
    build_pair_round_lookup,
    build_pair_dataset,
    build_pair_rows_for_submission,
    calibrate_probabilities,
    fit_final_model,
    load_tuned_model,
    round_bucket_from_round_number,
    seed_gap_bucket,
    tune_model,
)
from mmmania.silver import load_men_silver_probabilities


SEASON = 2026
CONSENSUS_PATH = OUTPUT_DIR / "markets" / "market_consensus.csv"
MONEYLINE_PATH = INPUT_DIR / "odds" / "tournament_moneylines.csv"
BACKTEST_PATH = OUTPUT_DIR / "backtests" / "rolling_backtests.json"
LIVE_SUBMISSION_PATH = OUTPUT_DIR / "submissions" / "live_submission_2026.csv"
OUTPUT_PATH = OUTPUT_DIR / "markets" / "men_tournament_board_2026.csv"


def _load_market_pairs() -> pd.DataFrame:
    """Attach moneyline metadata to the market consensus rows for readable diagnostics."""
    if not CONSENSUS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CONSENSUS_PATH}. Run scripts/build_market_consensus.py first."
        )

    consensus = pd.read_csv(CONSENSUS_PATH)
    consensus = consensus[(consensus["side"] == "M")].copy()
    if consensus.empty:
        raise ValueError("No men's market consensus rows found.")

    extras = consensus[["team_low_id", "team_high_id", "p_low_consensus", "quote_count", "moneyline_count"]]
    if MONEYLINE_PATH.exists():
        moneylines = pd.read_csv(MONEYLINE_PATH)
        moneylines = moneylines[moneylines["side"] == "M"].copy()
        if "round" not in moneylines.columns:
            moneylines["round"] = ""
        if "favorite_team_id" not in moneylines.columns:
            moneylines["favorite_team_id"] = pd.NA
        if "favorite_spread" not in moneylines.columns:
            moneylines["favorite_spread"] = pd.NA
        moneylines = (
            moneylines.sort_values(["quote_ts", "book"])
            .drop_duplicates(["team_low_id", "team_high_id"], keep="last")
            [["team_low_id", "team_high_id", "favorite_team_id", "favorite_spread", "round"]]
        )
        extras = extras.merge(moneylines, on=["team_low_id", "team_high_id"], how="left")
    return extras


def main() -> None:
    ensure_output_dirs()

    market_pairs = _load_market_pairs()
    ids = market_pairs.apply(
        lambda row: f"{SEASON}_{int(row['team_low_id'])}_{int(row['team_high_id'])}",
        axis=1,
    )

    side = get_side("M")
    team_features = build_team_season_features(side)
    pair_dataset = build_pair_dataset(side, team_features, SEEDED_SPEC)
    tuned_model = load_tuned_model(BACKTEST_PATH, side.label, SEEDED_SPEC.name)
    if tuned_model is None:
        tuned_model, _ = tune_model(side, pair_dataset, SEEDED_SPEC)
    model, seed_prior_payload = fit_final_model(pair_dataset, tuned_model)

    feature_rows = build_pair_rows_for_submission(
        side=side,
        team_features=team_features,
        season=SEASON,
        spec=SEEDED_SPEC,
        ids=ids,
    )
    if tuned_model.use_seed_matchup_prior:
        feature_rows = apply_seed_matchup_prior(feature_rows, seed_prior_payload)

    pair_round_lookup = build_pair_round_lookup("M", SEASON)
    raw_probabilities = model.predict_proba(
        feature_rows[list(tuned_model.feature_columns)].fillna(0.0)
    )[:, 1]
    round_buckets = feature_rows.apply(
        lambda row: round_bucket_from_round_number(
            pair_round_lookup.get(f"{int(row['TeamLowID'])}_{int(row['TeamHighID'])}")
        ),
        axis=1,
    )
    seed_gap_buckets = feature_rows["seed_gap"].map(seed_gap_bucket)
    feature_rows["model_p_low"] = calibrate_probabilities(
        raw_probabilities,
        global_scale=tuned_model.global_scale,
        round_buckets=round_buckets,
        round_bucket_scales=tuned_model.round_bucket_scales,
        seed_gap_buckets=seed_gap_buckets,
        seed_gap_bucket_scales=tuned_model.seed_gap_bucket_scales,
    )

    teams = pd.read_csv(DATA_DIR / "MTeams.csv", usecols=["TeamID", "TeamName"])
    seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv", usecols=["Season", "Seed", "TeamID"])
    seeds = seeds[seeds["Season"] == SEASON].copy()

    low_teams = teams.rename(columns={"TeamID": "team_low_id", "TeamName": "team_low_name"})
    high_teams = teams.rename(columns={"TeamID": "team_high_id", "TeamName": "team_high_name"})
    low_seeds = seeds.rename(columns={"TeamID": "team_low_id", "Seed": "seed_low"})
    high_seeds = seeds.rename(columns={"TeamID": "team_high_id", "Seed": "seed_high"})
    favorite_names = teams.rename(columns={"TeamID": "favorite_team_id", "TeamName": "favorite_team_name"})

    board = (
        market_pairs.merge(
            feature_rows[["ID", "TeamLowID", "TeamHighID", "model_p_low"]],
            left_on=["team_low_id", "team_high_id"],
            right_on=["TeamLowID", "TeamHighID"],
            how="left",
        )
        .merge(low_teams, on="team_low_id", how="left")
        .merge(high_teams, on="team_high_id", how="left")
        .merge(low_seeds[["team_low_id", "seed_low"]], on="team_low_id", how="left")
        .merge(high_seeds[["team_high_id", "seed_high"]], on="team_high_id", how="left")
        .merge(favorite_names, on="favorite_team_id", how="left")
    )

    silver = load_men_silver_probabilities(SEASON)
    if not silver.empty:
        silver["ID"] = silver.apply(
            lambda row: f"{SEASON}_{int(row['team_low_id'])}_{int(row['team_high_id'])}",
            axis=1,
        )
        board = board.merge(
            silver[["ID", "p_low_silver", "silver_round"]],
            on="ID",
            how="left",
        )
    else:
        board["p_low_silver"] = pd.NA
        board["silver_round"] = ""

    if LIVE_SUBMISSION_PATH.exists():
        live = pd.read_csv(LIVE_SUBMISSION_PATH)[["ID", "Pred"]].rename(columns={"Pred": "submission_p_low"})
        board = board.merge(live, on="ID", how="left")
    else:
        board["submission_p_low"] = board["model_p_low"]

    board["market_minus_model"] = board["p_low_consensus"] - board["model_p_low"]
    board["market_minus_submission"] = board["p_low_consensus"] - board["submission_p_low"]
    board["silver_minus_submission"] = board["p_low_silver"] - board["submission_p_low"]
    board = board[
        [
            "round",
            "ID",
            "seed_low",
            "team_low_name",
            "seed_high",
            "team_high_name",
            "favorite_team_name",
            "favorite_spread",
            "model_p_low",
            "submission_p_low",
            "p_low_consensus",
            "p_low_silver",
            "market_minus_model",
            "market_minus_submission",
            "silver_minus_submission",
            "quote_count",
            "moneyline_count",
        ]
    ].sort_values(["round", "seed_low", "seed_high", "team_low_name", "team_high_name"])

    board.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
