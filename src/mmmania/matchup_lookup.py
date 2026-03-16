from __future__ import annotations

"""Build a compact matchup lookup bundle from the current submission file."""

import re
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR, get_side
from .features import get_active_teams
from .modeling import parse_submission_ids


def normalize_team_text(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _side_alias_bundle(side_code: str, season: int) -> dict:
    side = get_side(side_code)
    active = get_active_teams(side, season)
    team_ids = set(int(team_id) for team_id in active["TeamID"])
    teams = active.sort_values("TeamName").reset_index(drop=True)

    aliases = {}
    for row in teams.itertuples(index=False):
        aliases[normalize_team_text(row.TeamName)] = int(row.TeamID)

    spellings_path = DATA_DIR / f"{side.code}TeamSpellings.csv"
    if spellings_path.exists():
        spellings = pd.read_csv(spellings_path)
        spellings = spellings[spellings["TeamID"].isin(team_ids)].copy()
        for row in spellings.itertuples(index=False):
            normalized = normalize_team_text(row.TeamNameSpelling)
            if normalized and normalized not in aliases:
                aliases[normalized] = int(row.TeamID)

    return {
        "label": side.label.title(),
        "teams": [
            {"teamId": int(row.TeamID), "teamName": row.TeamName}
            for row in teams.itertuples(index=False)
        ],
        "aliases": aliases,
    }


def _submission_probability_lookup(submission_path: Path, side_code: str, season: int) -> dict[str, float]:
    submission = pd.read_csv(submission_path)
    parsed = parse_submission_ids(submission["ID"])
    parsed["Pred"] = submission["Pred"]

    if side_code == "M":
        mask = parsed["TeamLowID"] < 3000
    else:
        mask = parsed["TeamLowID"] >= 3000

    parsed = parsed[(parsed["Season"] == season) & mask].copy()
    return {
        f"{int(row.TeamLowID)}_{int(row.TeamHighID)}": float(row.Pred)
        for row in parsed.itertuples(index=False)
    }


def build_matchup_lookup_bundle(
    season: int = 2026,
    submission_path: Path | None = None,
) -> dict:
    submission_path = submission_path or (OUTPUT_DIR / "submissions" / "live_submission_2026.csv")
    if not submission_path.exists():
        raise FileNotFoundError(
            f"Could not find {submission_path}. Run scripts/build_live_submission.py first."
        )

    men = _side_alias_bundle("M", season)
    women = _side_alias_bundle("W", season)

    return {
        "season": season,
        "sides": {
            "M": {
                **men,
                "probabilities": _submission_probability_lookup(submission_path, "M", season),
            },
            "W": {
                **women,
                "probabilities": _submission_probability_lookup(submission_path, "W", season),
            },
        },
    }
