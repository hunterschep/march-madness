from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.silver import (
    load_men_silver_probabilities,
    silver_direct_share_for_round,
    silver_observation_weight_for_round,
)


def test_load_men_silver_probabilities_maps_expected_rows() -> None:
    silver = load_men_silver_probabilities(2026)

    assert len(silver) == 40

    duke = silver[(silver["team_low_id"] == 1181) & (silver["team_high_id"] == 1373)].iloc[0]
    assert round(float(duke["p_low_silver"]), 4) == 0.9897
    assert duke["silver_round"] == "Round of 64"

    playin = silver[(silver["team_low_id"] == 1250) & (silver["team_high_id"] == 1341)].iloc[0]
    assert round(float(playin["p_low_silver"]), 4) == 0.4606
    assert playin["silver_round"] == "First Four"

    conditional = silver[(silver["team_low_id"] == 1224) & (silver["team_high_id"] == 1276)].iloc[0]
    assert round(float(conditional["p_low_silver"]), 4) == 0.0361


def test_silver_round_specific_weights_favor_round_of_64() -> None:
    assert silver_observation_weight_for_round(
        "First Four",
        first_four_weight=6.0,
        round64_weight=14.0,
    ) == 6.0
    assert silver_observation_weight_for_round(
        "Round of 64",
        first_four_weight=6.0,
        round64_weight=14.0,
    ) == 14.0
    assert silver_direct_share_for_round(
        "First Four",
        first_four_share=0.15,
        round64_share=0.35,
    ) == 0.15
    assert silver_direct_share_for_round(
        "Round of 64",
        first_four_share=0.15,
        round64_share=0.35,
    ) == 0.35
