from __future__ import annotations

"""Run rolling tournament-only backtests and persist the tuned specs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.config import OUTPUT_DIR, ensure_output_dirs, get_side
from mmmania.features import build_team_season_features
from mmmania.modeling import (
    PRIOR_SPEC,
    SEEDED_SPEC,
    build_pair_dataset,
    serialize_tuned_model,
    tune_model,
)


def main() -> None:
    ensure_output_dirs()

    payload = {"results": []}
    report_lines = ["# Rolling Tournament Backtests", ""]

    for side_code in ["M", "W"]:
        side = get_side(side_code)
        team_features = build_team_season_features(side)
        report_lines.append(f"## {side.label.title()}")
        report_lines.append("")

        for spec in [PRIOR_SPEC, SEEDED_SPEC]:
            pair_dataset = build_pair_dataset(side, team_features, spec)
            tuned_model, evaluations = tune_model(side, pair_dataset, spec)
            top_candidates = sorted(evaluations, key=lambda item: item["weighted_brier"])[:5]
            payload["results"].append(
                {
                    "side": side.label,
                    "model": spec.name,
                    "tuned_model": serialize_tuned_model(tuned_model),
                    "top_candidates": top_candidates,
                }
            )

            report_lines.append(
                f"- `{spec.name}` weighted Brier: `{tuned_model.weighted_brier:.4f}`"
                + f" via `{tuned_model.feature_columns}`"
                + f" with `C={tuned_model.regularization_c}`"
                + (" and seed prior" if tuned_model.use_seed_matchup_prior else "")
                + f"; global shrink `{tuned_model.global_scale:.2f}`"
                + (
                    f", round scales `{tuned_model.round_bucket_scales}`"
                    if any(scale != 1.0 for scale in tuned_model.round_bucket_scales)
                    else ""
                )
                + (
                    f", gap scales `{tuned_model.seed_gap_bucket_scales}`"
                    if any(scale != 1.0 for scale in tuned_model.seed_gap_bucket_scales)
                    else ""
                )
            )
            for season_metric in top_candidates[0]["season_metrics"]:
                report_lines.append(
                    "  "
                    + f"{season_metric['season']}: `{season_metric['brier']:.4f}`"
                    + f" across {season_metric['games']} games"
                )
            report_lines.append("")

    backtest_path = OUTPUT_DIR / "backtests" / "rolling_backtests.json"
    backtest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUTPUT_DIR / "reports" / "rolling_backtests.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
