from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.odds import american_to_implied_prob, consensus_probability, devig_two_way_probabilities


def test_american_to_implied_probability() -> None:
    assert round(american_to_implied_prob(+150), 4) == 0.4
    assert round(american_to_implied_prob(-150), 4) == 0.6


def test_devig_two_way_probabilities_sum_to_one() -> None:
    team_a, team_b = devig_two_way_probabilities(-120, +110)
    assert round(team_a + team_b, 8) == 1.0
    assert 0.0 < team_a < 1.0
    assert 0.0 < team_b < 1.0


def test_consensus_probability() -> None:
    probability = consensus_probability(
        [
            {"probability": 0.60, "weight": 2.0},
            {"probability": 0.50, "weight": 1.0},
        ]
    )
    assert round(probability, 4) == 0.5667

