from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmmania.matchup_lookup import build_matchup_lookup_bundle


def test_build_matchup_lookup_bundle_resolves_aliases_and_submission_probs(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame(
        [
            {"ID": "2026_1116_1429", "Pred": 0.4132},
            {"ID": "2026_3116_3429", "Pred": 0.6184},
        ]
    ).to_csv(submission_path, index=False)

    bundle = build_matchup_lookup_bundle(season=2026, submission_path=submission_path)

    men = bundle["sides"]["M"]
    women = bundle["sides"]["W"]

    assert men["aliases"]["arkansas"] == 1116
    assert men["aliases"]["utah state"] == 1429
    assert women["aliases"]["arkansas"] == 3116
    assert women["aliases"]["utah state"] == 3429

    assert men["probabilities"]["1116_1429"] == 0.4132
    assert women["probabilities"]["3116_3429"] == 0.6184
