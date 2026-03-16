from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import lstsq

from .config import DATA_DIR, INPUT_DIR, CompetitionSide


BASE_FEATURE_COLUMNS = [
    "games",
    "win_pct",
    "avg_margin",
    "avg_score",
    "avg_allowed",
    "recent_win_pct",
    "recent_margin",
    "home_win_pct",
    "away_win_pct",
    "neutral_win_pct",
    "off_eff",
    "def_eff",
    "net_eff",
    "adj_margin_rating",
    "ast_rate",
    "tov_rate",
    "ft_rate",
    "three_share",
    "conference_net_eff",
    "conference_win_pct",
]

MEN_ONLY_FEATURE_COLUMNS = [
    "massey_pct",
    "massey_system_count",
]

EXTERNAL_FEATURE_PREFIX = "ext_"


def _read_csv(filename: str, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename, usecols=list(usecols) if usecols else None)


def seed_to_number(seed: str) -> int:
    return int(str(seed)[1:3])


def _compact_team_games(side: CompetitionSide) -> pd.DataFrame:
    regular = _read_csv(
        f"{side.code}RegularSeasonCompactResults.csv",
        usecols=["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc"],
    )

    winners = regular[
        ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"]
    ].copy()
    winners.columns = [
        "Season",
        "DayNum",
        "TeamID",
        "OppID",
        "Score",
        "OppScore",
        "Loc",
    ]
    winners["Win"] = 1

    losers = regular[
        ["Season", "DayNum", "LTeamID", "WTeamID", "LScore", "WScore", "WLoc"]
    ].copy()
    losers.columns = [
        "Season",
        "DayNum",
        "TeamID",
        "OppID",
        "Score",
        "OppScore",
        "Loc",
    ]
    losers["Loc"] = losers["Loc"].map({"H": "A", "A": "H", "N": "N"})
    losers["Win"] = 0

    games = pd.concat([winners, losers], ignore_index=True)
    games["Margin"] = games["Score"] - games["OppScore"]
    return games.sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)


def _adjusted_margin_features(side: CompetitionSide) -> pd.DataFrame:
    regular = _read_csv(
        f"{side.code}RegularSeasonCompactResults.csv",
        usecols=["Season", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"],
    )

    rows = []
    for season, season_games in regular.groupby("Season", sort=True):
        teams = sorted(set(season_games["WTeamID"]).union(set(season_games["LTeamID"])))
        team_index = {team_id: idx for idx, team_id in enumerate(teams)}
        team_count = len(teams)

        design = np.zeros((len(season_games) + 1, team_count + 1))
        target = np.zeros(len(season_games) + 1)

        for row_index, game in enumerate(season_games.itertuples(index=False)):
            design[row_index, team_index[game.WTeamID]] = 1.0
            design[row_index, team_index[game.LTeamID]] = -1.0
            design[row_index, team_count] = 1.0 if game.WLoc == "H" else (-1.0 if game.WLoc == "A" else 0.0)
            target[row_index] = game.WScore - game.LScore

        # Constrain the ratings to sum to zero for identifiability.
        design[len(season_games), :team_count] = 1.0
        coefficients, _, _, _ = lstsq(design, target)

        for team_id, idx in team_index.items():
            rows.append(
                {
                    "Season": int(season),
                    "TeamID": int(team_id),
                    "adj_margin_rating": float(coefficients[idx]),
                }
            )

    return pd.DataFrame(rows)


def _detailed_team_games(side: CompetitionSide) -> pd.DataFrame:
    detailed = _read_csv(
        f"{side.code}RegularSeasonDetailedResults.csv",
        usecols=[
            "Season",
            "DayNum",
            "WTeamID",
            "WScore",
            "LTeamID",
            "LScore",
            "WFGM3",
            "WFGA",
            "WFTA",
            "WOR",
            "WAst",
            "WTO",
            "LFGM3",
            "LFGA",
            "LFTA",
            "LOR",
            "LAst",
            "LTO",
        ],
    )

    winners = pd.DataFrame(
        {
            "Season": detailed["Season"],
            "DayNum": detailed["DayNum"],
            "TeamID": detailed["WTeamID"],
            "Score": detailed["WScore"],
            "OppScore": detailed["LScore"],
            "FGA": detailed["WFGA"],
            "FGM3": detailed["WFGM3"],
            "FTA": detailed["WFTA"],
            "OR": detailed["WOR"],
            "Ast": detailed["WAst"],
            "TO": detailed["WTO"],
            "OppFGA": detailed["LFGA"],
            "OppFGM3": detailed["LFGM3"],
            "OppFTA": detailed["LFTA"],
            "OppOR": detailed["LOR"],
            "OppTO": detailed["LTO"],
        }
    )
    losers = pd.DataFrame(
        {
            "Season": detailed["Season"],
            "DayNum": detailed["DayNum"],
            "TeamID": detailed["LTeamID"],
            "Score": detailed["LScore"],
            "OppScore": detailed["WScore"],
            "FGA": detailed["LFGA"],
            "FGM3": detailed["LFGM3"],
            "FTA": detailed["LFTA"],
            "OR": detailed["LOR"],
            "Ast": detailed["LAst"],
            "TO": detailed["LTO"],
            "OppFGA": detailed["WFGA"],
            "OppFGM3": detailed["WFGM3"],
            "OppFTA": detailed["WFTA"],
            "OppOR": detailed["WOR"],
            "OppTO": detailed["WTO"],
        }
    )

    team_games = pd.concat([winners, losers], ignore_index=True)

    team_possessions = (
        team_games["FGA"] - team_games["OR"] + team_games["TO"] + 0.475 * team_games["FTA"]
    )
    opp_possessions = (
        team_games["OppFGA"]
        - team_games["OppOR"]
        + team_games["OppTO"]
        + 0.475 * team_games["OppFTA"]
    )
    possessions = ((team_possessions + opp_possessions) / 2.0).replace(0, np.nan)

    team_games["off_eff"] = team_games["Score"] / possessions * 100.0
    team_games["def_eff"] = team_games["OppScore"] / possessions * 100.0
    team_games["net_eff"] = team_games["off_eff"] - team_games["def_eff"]
    team_games["ast_rate"] = team_games["Ast"] / possessions * 100.0
    team_games["tov_rate"] = team_games["TO"] / possessions * 100.0
    team_games["ft_rate"] = team_games["FTA"] / team_games["FGA"].replace(0, np.nan)
    team_games["three_share"] = team_games["FGM3"] / team_games["FGA"].replace(0, np.nan)

    return team_games


def _conference_features(side: CompetitionSide, team_features: pd.DataFrame) -> pd.DataFrame:
    conferences = _read_csv(side.conference_file, usecols=["Season", "TeamID", "ConfAbbrev"])
    merged = conferences.merge(
        team_features[["Season", "TeamID", "net_eff", "win_pct"]],
        on=["Season", "TeamID"],
        how="left",
    )
    conference_strength = (
        merged.groupby(["Season", "ConfAbbrev"])
        .agg(
            conference_net_eff=("net_eff", "mean"),
            conference_win_pct=("win_pct", "mean"),
        )
        .reset_index()
    )
    return conferences.merge(conference_strength, on=["Season", "ConfAbbrev"], how="left")


def _massey_features() -> pd.DataFrame:
    massey = _read_csv(
        "MMasseyOrdinals.csv",
        usecols=["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"],
    )
    latest_day = massey.groupby("Season")["RankingDayNum"].transform("max")
    massey = massey[massey["RankingDayNum"] == latest_day].copy()
    massey["max_rank"] = massey.groupby(["Season", "SystemName"])["OrdinalRank"].transform("max")
    massey["massey_pct"] = (massey["max_rank"] + 1 - massey["OrdinalRank"]) / massey["max_rank"]
    return (
        massey.groupby(["Season", "TeamID"])
        .agg(
            massey_pct=("massey_pct", "mean"),
            massey_system_count=("massey_pct", "size"),
        )
        .reset_index()
    )


def _external_team_features(side: CompetitionSide) -> pd.DataFrame | None:
    path = INPUT_DIR / "external" / "team_features.csv"
    if not path.exists():
        return None

    external = pd.read_csv(path)
    required = {"side", "season", "team_id"}
    missing = required - set(external.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing external feature columns: {missing_list}")

    external = external[external["side"] == side.code].copy()
    if external.empty:
        return None

    rename_map = {"season": "Season", "team_id": "TeamID"}
    value_columns = []
    for column in external.columns:
        if column in {"side", "season", "team_id"}:
            continue
        if pd.api.types.is_numeric_dtype(external[column]):
            prefixed = column if column.startswith(EXTERNAL_FEATURE_PREFIX) else f"{EXTERNAL_FEATURE_PREFIX}{column}"
            rename_map[column] = prefixed
            value_columns.append(column)

    if not value_columns:
        return None

    external = external.rename(columns=rename_map)
    keep_columns = ["Season", "TeamID", *[rename_map[col] for col in value_columns]]
    return external[keep_columns].drop_duplicates(["Season", "TeamID"])


def build_team_season_features(side: CompetitionSide) -> pd.DataFrame:
    compact_games = _compact_team_games(side)
    detailed_games = _detailed_team_games(side)
    adjusted_margin = _adjusted_margin_features(side)

    base = (
        compact_games.groupby(["Season", "TeamID"])
        .agg(
            games=("Win", "size"),
            win_pct=("Win", "mean"),
            avg_margin=("Margin", "mean"),
            avg_score=("Score", "mean"),
            avg_allowed=("OppScore", "mean"),
        )
        .reset_index()
    )

    recent = (
        compact_games.groupby(["Season", "TeamID"])
        .tail(10)
        .groupby(["Season", "TeamID"])
        .agg(
            recent_win_pct=("Win", "mean"),
            recent_margin=("Margin", "mean"),
        )
        .reset_index()
    )

    split_rows = []
    for loc_value, loc_label in [("H", "home"), ("A", "away"), ("N", "neutral")]:
        subset = compact_games[compact_games["Loc"] == loc_value]
        loc_features = (
            subset.groupby(["Season", "TeamID"])
            .agg(**{f"{loc_label}_win_pct": ("Win", "mean")})
            .reset_index()
        )
        split_rows.append(loc_features)

    efficiency = (
        detailed_games.groupby(["Season", "TeamID"])
        .agg(
            off_eff=("off_eff", "mean"),
            def_eff=("def_eff", "mean"),
            net_eff=("net_eff", "mean"),
            ast_rate=("ast_rate", "mean"),
            tov_rate=("tov_rate", "mean"),
            ft_rate=("ft_rate", "mean"),
            three_share=("three_share", "mean"),
        )
        .reset_index()
    )

    features = base.merge(recent, on=["Season", "TeamID"], how="left").merge(
        efficiency, on=["Season", "TeamID"], how="left"
    )
    features = features.merge(adjusted_margin, on=["Season", "TeamID"], how="left")

    for split in split_rows:
        features = features.merge(split, on=["Season", "TeamID"], how="left")

    features[["home_win_pct", "away_win_pct", "neutral_win_pct"]] = features[
        ["home_win_pct", "away_win_pct", "neutral_win_pct"]
    ].fillna(0.5)

    conference_features = _conference_features(side, features)
    features = features.merge(
        conference_features[
            ["Season", "TeamID", "conference_net_eff", "conference_win_pct"]
        ],
        on=["Season", "TeamID"],
        how="left",
    )

    if side.uses_massey:
        features = features.merge(_massey_features(), on=["Season", "TeamID"], how="left")
    else:
        features["massey_pct"] = np.nan
        features["massey_system_count"] = np.nan

    external = _external_team_features(side)
    if external is not None:
        features = features.merge(external, on=["Season", "TeamID"], how="left")

    return features.sort_values(["Season", "TeamID"]).reset_index(drop=True)


def get_model_feature_columns(
    side: CompetitionSide,
    include_seeds: bool,
    team_features: pd.DataFrame | None = None,
) -> list[str]:
    columns = list(BASE_FEATURE_COLUMNS)
    if side.uses_massey:
        columns.extend(MEN_ONLY_FEATURE_COLUMNS)
    if team_features is not None:
        columns.extend(
            sorted(col for col in team_features.columns if col.startswith(EXTERNAL_FEATURE_PREFIX))
        )
    if include_seeds:
        columns.append("seed_num")
    return columns


def get_active_teams(side: CompetitionSide, season: int) -> pd.DataFrame:
    if side.code == "M":
        teams = _read_csv(
            side.team_file,
            usecols=["TeamID", "TeamName", "FirstD1Season", "LastD1Season"],
        )
        active = teams[
            (teams["FirstD1Season"] <= season) & (teams["LastD1Season"] >= season)
        ].copy()
        return active[["TeamID", "TeamName"]].sort_values("TeamID").reset_index(drop=True)

    conferences = _read_csv(side.conference_file, usecols=["Season", "TeamID"])
    active_team_ids = conferences.loc[conferences["Season"] == season, "TeamID"].drop_duplicates()
    teams = _read_csv(side.team_file, usecols=["TeamID", "TeamName"])
    active = teams[teams["TeamID"].isin(active_team_ids)].copy()
    return active[["TeamID", "TeamName"]].sort_values("TeamID").reset_index(drop=True)


def get_tournament_seeds(side: CompetitionSide) -> pd.DataFrame:
    seeds = _read_csv(f"{side.code}NCAATourneySeeds.csv", usecols=["Season", "Seed", "TeamID"])
    seeds["seed_num"] = seeds["Seed"].map(seed_to_number)
    return seeds[["Season", "TeamID", "seed_num"]]
