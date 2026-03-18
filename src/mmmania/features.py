from __future__ import annotations

"""Small team and seed loaders used by the final submission pipeline."""

import pandas as pd

from .config import DATA_DIR, CompetitionSide


def get_active_teams(side: CompetitionSide, season: int) -> pd.DataFrame:
    """Return the active teams for the requested side and season."""
    if side.code == "M":
        teams = pd.read_csv(
            DATA_DIR / side.team_file,
            usecols=["TeamID", "TeamName", "FirstD1Season", "LastD1Season"],
        )
        active = teams.loc[
            (teams["FirstD1Season"] <= season) & (teams["LastD1Season"] >= season),
            ["TeamID", "TeamName"],
        ].copy()
        return active.sort_values("TeamID").reset_index(drop=True)

    conferences = pd.read_csv(
        DATA_DIR / side.conference_file,
        usecols=["Season", "TeamID"],
    )
    active_ids = conferences.loc[conferences["Season"] == season, "TeamID"].drop_duplicates()
    teams = pd.read_csv(DATA_DIR / side.team_file, usecols=["TeamID", "TeamName"])
    active = teams.loc[teams["TeamID"].isin(active_ids), ["TeamID", "TeamName"]].copy()
    return active.sort_values("TeamID").reset_index(drop=True)


def get_tournament_seeds(side: CompetitionSide) -> pd.DataFrame:
    """Return tournament seeds with numeric seed values."""
    seeds = pd.read_csv(
        DATA_DIR / f"{side.code}NCAATourneySeeds.csv",
        usecols=["Season", "Seed", "TeamID"],
    )
    seeds["seed_num"] = seeds["Seed"].map(lambda seed: int(str(seed)[1:3]))
    return seeds[["Season", "TeamID", "seed_num"]]
