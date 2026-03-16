from __future__ import annotations

"""Build market consensus probabilities from raw moneyline quotes."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.config import OUTPUT_DIR, ensure_output_dirs
from mmmania.odds import (
    consensus_probability,
    devig_two_way_probabilities,
)


DEFAULT_MONEYLINE_PATH = ROOT / "inputs" / "odds" / "tournament_moneylines.csv"
MONEYLINE_TEMPLATE_PATH = ROOT / "templates" / "tournament_moneylines_template.csv"
OUTPUT_PATH = OUTPUT_DIR / "markets" / "market_consensus.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-vig market consensus from moneyline quotes.")
    parser.add_argument(
        "--moneylines",
        type=Path,
        default=DEFAULT_MONEYLINE_PATH,
        help="CSV of raw moneyline quotes. Use the template file as the schema reference.",
    )
    return parser.parse_args()


def _moneyline_quotes(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    quotes = pd.read_csv(path)
    required = {"side", "team_low_id", "team_high_id", "low_odds", "high_odds", "book", "quote_ts"}
    missing = required - set(quotes.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing moneyline columns: {missing_list}")

    quotes = quotes.copy()
    quotes["probability"] = quotes.apply(
        lambda row: devig_two_way_probabilities(row["low_odds"], row["high_odds"])[0],
        axis=1,
    )
    if "weight" not in quotes.columns:
        quotes["weight"] = 1.0
    return quotes[["side", "team_low_id", "team_high_id", "probability", "book", "quote_ts", "weight"]]


def _preferred_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Keep one latest moneyline quote per book for each matchup."""
    if quotes.empty:
        return quotes

    quotes = quotes.copy()
    quotes["quote_ts"] = pd.to_datetime(quotes["quote_ts"], utc=True, errors="coerce")
    if quotes["quote_ts"].isna().any():
        raise ValueError("All market quotes must have valid ISO timestamps.")

    quotes = quotes.sort_values(
        ["side", "team_low_id", "team_high_id", "book", "quote_ts"]
    ).drop_duplicates(
        ["side", "team_low_id", "team_high_id", "book"],
        keep="last",
    )
    quotes["quote_ts"] = quotes["quote_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return quotes.sort_values(["side", "team_low_id", "team_high_id", "book"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    quotes = _moneyline_quotes(args.moneylines)

    if quotes.empty:
        raise FileNotFoundError(
            "Could not find any market inputs. Provide moneylines using "
            f"{MONEYLINE_TEMPLATE_PATH} as the schema reference."
        )

    quotes = _preferred_quotes(quotes)

    grouped_rows = []
    for (side, low_id, high_id), group in quotes.groupby(["side", "team_low_id", "team_high_id"]):
        grouped_rows.append(
            {
                "side": side,
                "team_low_id": int(low_id),
                "team_high_id": int(high_id),
                "p_low_consensus": consensus_probability(
                    group[["probability", "weight"]]
                    .to_dict("records")
                ),
                "quote_count": int(len(group)),
                "moneyline_count": int(len(group)),
                "latest_quote_ts": group["quote_ts"].max(),
            }
        )

    consensus = pd.DataFrame(grouped_rows).sort_values(["side", "team_low_id", "team_high_id"])
    consensus.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
