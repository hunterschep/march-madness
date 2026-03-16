from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.config import OUTPUT_DIR, ensure_output_dirs, get_side
from mmmania.features import build_team_season_features, get_active_teams


def main() -> None:
    ensure_output_dirs()

    for side_code in ["M", "W"]:
        side = get_side(side_code)
        features = build_team_season_features(side)
        active = get_active_teams(side, 2026)
        snapshot = active.merge(
            features[features["Season"] == 2026],
            on="TeamID",
            how="left",
        ).sort_values("TeamID")
        output_path = OUTPUT_DIR / "features" / f"{side.label}_team_features_2026.csv"
        snapshot.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

