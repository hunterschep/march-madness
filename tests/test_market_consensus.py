from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_market_consensus import _preferred_quotes


def test_preferred_quotes_keep_latest_quote_per_book() -> None:
    quotes = pd.DataFrame(
        [
            {
                "side": "M",
                "team_low_id": 1101,
                "team_high_id": 1102,
                "probability": 0.61,
                "book": "sharp_book",
                "quote_ts": "2026-03-19T10:00:00Z",
                "weight": 0.7,
            },
            {
                "side": "M",
                "team_low_id": 1101,
                "team_high_id": 1102,
                "probability": 0.58,
                "book": "sharp_book",
                "quote_ts": "2026-03-19T10:05:00Z",
                "weight": 1.0,
            },
        ]
    )

    preferred = _preferred_quotes(quotes)
    assert len(preferred) == 1
    assert preferred.iloc[0]["probability"] == 0.58


def test_preferred_quotes_keep_one_latest_quote_per_book() -> None:
    quotes = pd.DataFrame(
        [
            {
                "side": "M",
                "team_low_id": 1101,
                "team_high_id": 1102,
                "probability": 0.57,
                "book": "book_a",
                "quote_ts": "2026-03-19T09:55:00Z",
                "weight": 1.0,
            },
            {
                "side": "M",
                "team_low_id": 1101,
                "team_high_id": 1102,
                "probability": 0.59,
                "book": "book_a",
                "quote_ts": "2026-03-19T10:05:00Z",
                "weight": 1.0,
            },
            {
                "side": "M",
                "team_low_id": 1101,
                "team_high_id": 1102,
                "probability": 0.6,
                "book": "book_b",
                "quote_ts": "2026-03-19T10:00:00Z",
                "weight": 1.0,
            },
        ]
    )

    preferred = _preferred_quotes(quotes)
    assert len(preferred) == 2
    assert sorted(preferred["probability"].tolist()) == [0.59, 0.6]
