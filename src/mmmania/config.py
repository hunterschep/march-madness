from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
INPUT_DIR = ROOT / "inputs"
OUTPUT_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class CompetitionSide:
    code: str
    label: str
    uses_massey: bool
    team_file: str
    conference_file: str


SIDES = {
    "M": CompetitionSide(
        code="M",
        label="men",
        uses_massey=True,
        team_file="MTeams.csv",
        conference_file="MTeamConferences.csv",
    ),
    "W": CompetitionSide(
        code="W",
        label="women",
        uses_massey=False,
        team_file="WTeams.csv",
        conference_file="WTeamConferences.csv",
    ),
}


def get_side(code: str) -> CompetitionSide:
    if code not in SIDES:
        raise ValueError(f"Unknown competition side: {code}")
    return SIDES[code]


def ensure_output_dirs() -> None:
    for path in [
        OUTPUT_DIR,
        OUTPUT_DIR / "backtests",
        OUTPUT_DIR / "submissions",
        OUTPUT_DIR / "features",
        OUTPUT_DIR / "markets",
        OUTPUT_DIR / "reports",
    ]:
        path.mkdir(parents=True, exist_ok=True)
