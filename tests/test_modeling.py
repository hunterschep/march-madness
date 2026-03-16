from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.modeling import build_pair_round_lookup, calibrate_probabilities


def test_calibrate_probabilities_shrinks_toward_half() -> None:
    raw = pd.Series([0.1, 0.5, 0.9])
    calibrated = calibrate_probabilities(raw, global_scale=0.9)

    assert round(float(calibrated.iloc[0]), 4) == 0.14
    assert round(float(calibrated.iloc[1]), 4) == 0.5
    assert round(float(calibrated.iloc[2]), 4) == 0.86


def test_calibrate_probabilities_supports_round_and_gap_buckets() -> None:
    raw = pd.Series([0.8, 0.8, 0.8])
    calibrated = calibrate_probabilities(
        raw,
        global_scale=1.0,
        round_buckets=pd.Series([0, 1, 2]),
        round_bucket_scales=(0.9, 1.0, 1.0),
        seed_gap_buckets=pd.Series([0, 2, 3]),
        seed_gap_bucket_scales=(1.0, 1.0, 0.95, 0.9),
    )

    assert float(calibrated.iloc[0]) < float(calibrated.iloc[1])
    assert float(calibrated.iloc[2]) < float(calibrated.iloc[1])


def test_mens_pair_round_lookup_matches_known_paths() -> None:
    lookup = build_pair_round_lookup("M", 2026)

    assert lookup["1181_1373"] == 1
    assert lookup["1163_1181"] == 4
    assert lookup["1112_1181"] == 6
