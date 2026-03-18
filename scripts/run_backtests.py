from __future__ import annotations

"""Run modeh7 backtests and write both fair and notebook-style audit summaries."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.config import OUTPUT_DIR, ensure_output_dirs
from mmmania.modeh7 import (
    MODEH7_DEFAULT_AGGRESSIVENESS,
    build_modeh7_training_frame,
    fit_modeh7_ensemble,
    score_modeh7_oof,
)
from mmmania.modeling import TARGET_SEASONS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild the cached modeh7 feature tables instead of reusing them.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the modeh7 feature-table cache.",
    )
    return parser.parse_args()


def _render_section(title: str, summary: dict) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append(f"- Overall Brier: `{summary['overall_brier']:.4f}`")
    lines.append(f"- Men Brier: `{summary['men_brier']:.4f}`")
    lines.append(f"- Women Brier: `{summary['women_brier']:.4f}`")
    lines.append(f"- Models: `{summary['num_models']}`")
    if "backend" in summary:
        lines.append(f"- Backend: `{summary['backend']}`")
    lines.append(f"- Calibration: `{summary['calibration_mode']}`")
    lines.append(f"- Aggressiveness: `{summary['aggressiveness']}`")
    if "eval_seasons" in summary:
        lines.append(f"- Eval seasons: `{summary['eval_seasons']}`")
    lines.append("")
    for season_metric in summary["season_metrics"]:
        lines.append(
            "- "
            + f"{season_metric['season']}: `{season_metric['overall_brier']:.4f}`"
            + f" across {season_metric['games']} games"
        )
    lines.append("")
    return lines


def main() -> None:
    ensure_output_dirs()
    args = _parse_args()

    training_frame, _ = build_modeh7_training_frame(
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )

    rolling_ensemble, rolling_oof = fit_modeh7_ensemble(
        training_frame,
        validation="rolling",
        eval_seasons=TARGET_SEASONS,
        calibration_mode="by_gender",
    )
    rolling_summary = score_modeh7_oof(
        rolling_oof,
        rolling_ensemble,
        aggressiveness=MODEH7_DEFAULT_AGGRESSIVENESS,
    )
    rolling_summary.update(
        {
            "validation": "rolling",
            "backend": rolling_ensemble.backend,
            "calibration_mode": "by_gender",
            "num_models": int(len(rolling_ensemble.models)),
            "aggressiveness": MODEH7_DEFAULT_AGGRESSIVENESS,
            "eval_seasons": TARGET_SEASONS,
        }
    )
    loso_ensemble, loso_oof = fit_modeh7_ensemble(
        training_frame,
        validation="loso",
        calibration_mode="by_gender",
    )
    loso_full_summary = score_modeh7_oof(
        loso_oof,
        loso_ensemble,
        aggressiveness=MODEH7_DEFAULT_AGGRESSIVENESS,
    )
    loso_full_summary.update(
        {
            "validation": "loso",
            "backend": loso_ensemble.backend,
            "calibration_mode": "by_gender",
            "num_models": int(len(loso_ensemble.models)),
            "aggressiveness": MODEH7_DEFAULT_AGGRESSIVENESS,
        }
    )
    loso_target_oof = loso_oof[loso_oof["Season"].isin(TARGET_SEASONS)].copy()
    loso_target_summary = score_modeh7_oof(
        loso_target_oof,
        loso_ensemble,
        aggressiveness=MODEH7_DEFAULT_AGGRESSIVENESS,
    )
    loso_target_summary.update(
        {
            "validation": "loso",
            "backend": loso_ensemble.backend,
            "calibration_mode": "by_gender",
            "num_models": int(len(loso_ensemble.models)),
            "aggressiveness": MODEH7_DEFAULT_AGGRESSIVENESS,
            "eval_seasons": TARGET_SEASONS,
        }
    )
    payload = {
        "modeh7": {
            "rolling_target_2021_2025": rolling_summary,
            "loso_target_2021_2025": loso_target_summary,
            "loso_full_history": loso_full_summary,
        }
    }

    report_lines = [
        "# Modeh7 Backtests",
        "",
        "The fair score is the rolling `2021-2025` section below.",
        "The LOSO sections are included only to compare against the notebook-style headline numbers.",
        "",
    ]
    report_lines.extend(_render_section("Fair Rolling 2021-2025", rolling_summary))
    report_lines.extend(_render_section("Notebook-Style LOSO 2021-2025 Slice", loso_target_summary))
    report_lines.extend(_render_section("Notebook-Style LOSO Full History", loso_full_summary))

    backtest_path = OUTPUT_DIR / "backtests" / "rolling_backtests.json"
    backtest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUTPUT_DIR / "reports" / "rolling_backtests.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
