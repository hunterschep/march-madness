from __future__ import annotations

"""Load Nate Silver tournament forecasts and map them onto Kaggle team pairs."""

from pathlib import Path

import pandas as pd

from .config import DATA_DIR


MEN_REGION_CODES = {
    "East": "W",
    "South": "X",
    "Midwest": "Y",
    "West": "Z",
}


def silver_observation_weight_for_round(
    round_name: str,
    *,
    first_four_weight: float,
    round64_weight: float,
) -> float:
    """Return the latent-layer observation weight for a Silver forecast row."""
    return round64_weight if str(round_name) == "Round of 64" else first_four_weight


def silver_direct_share_for_round(
    round_name: str,
    *,
    first_four_share: float,
    round64_share: float,
) -> float:
    """Return the direct blend share assigned to Silver on priced rows."""
    return round64_share if str(round_name) == "Round of 64" else first_four_share


def _silver_seed_label(region_name: str, sb_name: str) -> str:
    region_code = MEN_REGION_CODES[region_name]
    sb_name = str(sb_name).strip().lower()
    digits = "".join(char for char in sb_name if char.isdigit())
    suffix = "".join(char for char in sb_name if char.isalpha())
    return f"{region_code}{int(digits):02d}{suffix}"


def _load_silver_team_table(season: int, directory: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["source_file"] = path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    silver = pd.concat(frames, ignore_index=True)
    required = {
        "team_region",
        "sb_name",
        "playin_flag",
        "rd1_win",
        "rd2_win",
        "timestamp",
    }
    missing = required - set(silver.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing Silver columns: {missing_list}")

    silver = silver.copy()
    silver["Seed"] = silver.apply(
        lambda row: _silver_seed_label(row["team_region"], row["sb_name"]),
        axis=1,
    )

    seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    seeds = seeds[seeds["Season"] == season][["Seed", "TeamID"]].copy()
    silver = silver.merge(seeds, on="Seed", how="left")
    if silver["TeamID"].isna().any():
        missing_seeds = sorted(silver.loc[silver["TeamID"].isna(), "Seed"].unique())
        raise ValueError(f"Could not map Silver seeds to Kaggle TeamIDs: {missing_seeds}")

    silver["TeamID"] = silver["TeamID"].astype(int)
    silver["playin_flag"] = silver["playin_flag"].astype(int)
    silver["rd1_win"] = silver["rd1_win"].astype(float) / 100.0
    silver["rd2_win"] = silver["rd2_win"].astype(float) / 100.0
    silver["cond_round1_win"] = silver.apply(
        lambda row: 0.0 if row["rd1_win"] <= 0 else row["rd2_win"] / row["rd1_win"],
        axis=1,
    )
    return silver


def load_men_silver_probabilities(
    season: int = 2026,
    directory: Path | None = None,
) -> pd.DataFrame:
    """Return men’s first-four and round-of-64 pair probabilities from Silver’s projections."""
    directory = directory or (DATA_DIR / "silver")
    if not directory.exists():
        return pd.DataFrame(
            columns=[
                "team_low_id",
                "team_high_id",
                "p_low_silver",
                "silver_round",
                "silver_source_ts",
            ]
        )

    teams = _load_silver_team_table(season, directory)
    if teams.empty:
        return pd.DataFrame(
            columns=[
                "team_low_id",
                "team_high_id",
                "p_low_silver",
                "silver_round",
                "silver_source_ts",
            ]
        )

    team_lookup = {str(row.Seed): row for row in teams.itertuples(index=False)}

    slots = pd.read_csv(DATA_DIR / "MNCAATourneySlots.csv")
    slots = slots[slots["Season"] == season].copy()

    results = []

    # First Four direct probabilities.
    playin_slots = slots[~slots["Slot"].str.startswith("R")].copy()
    for row in playin_slots.itertuples(index=False):
        strong = team_lookup.get(str(row.StrongSeed))
        weak = team_lookup.get(str(row.WeakSeed))
        if strong is None or weak is None:
            continue

        strong_prob = float(strong.rd1_win)
        weak_prob = float(weak.rd1_win)
        total = strong_prob + weak_prob
        if total <= 0:
            continue
        strong_prob /= total

        low_id = min(int(strong.TeamID), int(weak.TeamID))
        high_id = max(int(strong.TeamID), int(weak.TeamID))
        p_low = strong_prob if int(strong.TeamID) == low_id else 1.0 - strong_prob
        results.append(
            {
                "team_low_id": low_id,
                "team_high_id": high_id,
                "p_low_silver": p_low,
                "silver_round": "First Four",
                "silver_source_ts": str(strong.timestamp),
            }
        )

    # Round of 64 direct probabilities, including the conditional pair rows against play-in teams.
    round1_slots = slots[slots["Slot"].str.startswith("R1")].copy()
    slot_children = playin_slots.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict("index")

    def candidates(seed_or_slot: str) -> list:
        if seed_or_slot in team_lookup:
            return [team_lookup[seed_or_slot]]
        child = slot_children.get(seed_or_slot)
        if child is None:
            return []
        rows = []
        for child_seed in (str(child["StrongSeed"]), str(child["WeakSeed"])):
            team = team_lookup.get(child_seed)
            if team is not None:
                rows.append(team)
        return rows

    for row in round1_slots.itertuples(index=False):
        strong_candidates = candidates(str(row.StrongSeed))
        weak_candidates = candidates(str(row.WeakSeed))
        for strong in strong_candidates:
            for weak in weak_candidates:
                if int(strong.playin_flag) == 1 and int(weak.playin_flag) == 0:
                    strong_prob = float(strong.cond_round1_win)
                elif int(weak.playin_flag) == 1 and int(strong.playin_flag) == 0:
                    strong_prob = 1.0 - float(weak.cond_round1_win)
                else:
                    strong_prob = float(strong.cond_round1_win)
                    weak_prob = float(weak.cond_round1_win)
                    total = strong_prob + weak_prob
                    if total <= 0:
                        continue
                    strong_prob /= total

                low_id = min(int(strong.TeamID), int(weak.TeamID))
                high_id = max(int(strong.TeamID), int(weak.TeamID))
                p_low = strong_prob if int(strong.TeamID) == low_id else 1.0 - strong_prob
                results.append(
                    {
                        "team_low_id": low_id,
                        "team_high_id": high_id,
                        "p_low_silver": p_low,
                        "silver_round": "Round of 64",
                        "silver_source_ts": str(strong.timestamp),
                    }
                )

    silver_pairs = pd.DataFrame(results).drop_duplicates(["team_low_id", "team_high_id"])
    if silver_pairs.empty:
        return pd.DataFrame(
            columns=[
                "team_low_id",
                "team_high_id",
                "p_low_silver",
                "silver_round",
                "silver_source_ts",
            ]
        )

    silver_pairs["p_low_silver"] = silver_pairs["p_low_silver"].clip(0.001, 0.999)
    return silver_pairs.sort_values(["silver_round", "team_low_id", "team_high_id"]).reset_index(drop=True)
