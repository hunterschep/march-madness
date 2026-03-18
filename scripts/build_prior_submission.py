from __future__ import annotations

"""Build the notebook-style all-pairs submission baseline."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.config import DATA_DIR, OUTPUT_DIR, ensure_output_dirs
from mmmania.modeh7 import (
    MODEH7_DEFAULT_AGGRESSIVENESS,
    MODEH7_DEFAULT_CALIBRATION_MODE,
    MODEH7_DEFAULT_VALIDATION,
    generate_modeh7_submission,
)


SEASON = 2026
OUTPUT_PATH = OUTPUT_DIR / "submissions" / "prior_submission_2026.csv"
SUMMARY_PATH = OUTPUT_DIR / "reports" / "prior_submission_summary_2026.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        choices=["loso", "rolling"],
        default=MODEH7_DEFAULT_VALIDATION,
        help="Training / OOF scheme used for the ensemble and calibrator.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["combined", "by_gender"],
        default=MODEH7_DEFAULT_CALIBRATION_MODE,
        help="Probability calibration strategy.",
    )
    parser.add_argument(
        "--aggressiveness",
        type=float,
        default=MODEH7_DEFAULT_AGGRESSIVENESS,
        help="Multiplier applied to predicted margins before calibration.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild the cached feature tables instead of reusing them.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the modeh7 feature-table cache.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_output_dirs()
    args = _parse_args()

    sample = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
    submission, summary = generate_modeh7_submission(
        sample["ID"],
        validation=args.validation,
        calibration_mode=args.calibration_mode,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
        aggressiveness=args.aggressiveness,
    )

    submission = (
        submission.set_index("ID")
        .reindex(sample["ID"])
        .reset_index()
    )
    submission["Pred"] = submission["Pred"].round(6)
    submission.to_csv(OUTPUT_PATH, index=False)

    payload = {
        "season": SEASON,
        "output_path": str(OUTPUT_PATH),
        **summary,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
