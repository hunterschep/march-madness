from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.config import get_side
from mmmania.features import get_active_teams


def test_active_2026_team_counts_match_submission_scope() -> None:
    assert len(get_active_teams(get_side("M"), 2026)) == 365
    assert len(get_active_teams(get_side("W"), 2026)) == 363

