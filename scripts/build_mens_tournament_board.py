from __future__ import annotations

"""Export a readable board comparing the modeh7 baseline, live submission, and market."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.config import DATA_DIR, INPUT_DIR, OUTPUT_DIR, ensure_output_dirs
from mmmania.modeh7 import (
    MODEH7_DEFAULT_AGGRESSIVENESS,
    MODEH7_DEFAULT_CALIBRATION_MODE,
    MODEH7_DEFAULT_VALIDATION,
    generate_modeh7_submission,
)
from mmmania.silver import load_men_silver_probabilities


SEASON = 2026
CONSENSUS_PATH = OUTPUT_DIR / "markets" / "market_consensus.csv"
MONEYLINE_PATH = INPUT_DIR / "odds" / "tournament_moneylines.csv"
PRIOR_PATH = OUTPUT_DIR / "submissions" / "prior_submission_2026.csv"
LIVE_SUBMISSION_PATH = OUTPUT_DIR / "submissions" / "live_submission_2026.csv"
OUTPUT_PATH = OUTPUT_DIR / "markets" / "men_tournament_board_2026.csv"


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
        help="Rebuild modeh7 predictions inline for the board instead of reusing the saved prior CSV.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild the cached modeh7 feature tables while generating inline predictions.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the modeh7 feature-table cache while generating inline predictions.",
    )
    return parser.parse_args()


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


def _load_or_build_model_predictions(
    ids: pd.Series,
    *,
    prior_path: Path,
    refresh_prior: bool,
    use_cache: bool,
    refresh_cache: bool,
) -> pd.DataFrame:
    ids = pd.Series(ids, copy=False, name="ID")
    if prior_path.exists() and not refresh_prior:
        model = pd.read_csv(prior_path)[["ID", "Pred"]].copy()
    else:
        model, _ = generate_modeh7_submission(
            ids,
            validation=MODEH7_DEFAULT_VALIDATION,
            calibration_mode=MODEH7_DEFAULT_CALIBRATION_MODE,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            aggressiveness=MODEH7_DEFAULT_AGGRESSIVENESS,
        )

    model = (
        model.rename(columns={"Pred": "model_p_low"})
        .set_index("ID")
        .reindex(ids)
        .reset_index()
    )
    if model["model_p_low"].isna().any():
        missing_count = int(model["model_p_low"].isna().sum())
        raise ValueError(f"Modeh7 model predictions are missing {missing_count} board rows.")
    return model


def main() -> None:
    ensure_output_dirs()
    args = _parse_args()

    market_pairs = _load_market_pairs()
    ids = market_pairs.apply(
        lambda row: f"{SEASON}_{int(row['team_low_id'])}_{int(row['team_high_id'])}",
        axis=1,
    ).rename("ID")
    model_rows = _load_or_build_model_predictions(
        ids,
        prior_path=args.prior_path,
        refresh_prior=args.refresh_prior,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
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
        market_pairs.assign(ID=ids)
        .merge(model_rows, on="ID", how="left")
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
