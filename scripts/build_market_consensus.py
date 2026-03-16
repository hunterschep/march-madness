from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.config import OUTPUT_DIR, ensure_output_dirs
from mmmania.odds import consensus_probability, devig_two_way_probabilities


DEFAULT_INPUT_PATH = ROOT / "inputs" / "odds" / "tournament_moneylines.csv"
TEMPLATE_PATH = ROOT / "templates" / "tournament_moneylines_template.csv"
OUTPUT_PATH = OUTPUT_DIR / "markets" / "market_consensus.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-vig market consensus from moneyline quotes.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV of raw moneyline quotes. Use the template file as the schema reference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Could not find {args.input}. Copy or export real odds to that path using "
            f"{TEMPLATE_PATH} as the schema reference."
        )

    quotes = pd.read_csv(args.input)
    required = {"side", "team_low_id", "team_high_id", "low_odds", "high_odds", "book", "quote_ts"}
    missing = required - set(quotes.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")

    quotes = quotes.copy()
    quote_probabilities = quotes.apply(
        lambda row: devig_two_way_probabilities(row["low_odds"], row["high_odds"])[0],
        axis=1,
    )
    quotes["p_low_devig"] = quote_probabilities
    if "weight" not in quotes.columns:
        quotes["weight"] = 1.0

    grouped_rows = []
    for (side, low_id, high_id), group in quotes.groupby(["side", "team_low_id", "team_high_id"]):
        grouped_rows.append(
            {
                "side": side,
                "team_low_id": int(low_id),
                "team_high_id": int(high_id),
                "p_low_consensus": consensus_probability(
                    group[["p_low_devig", "weight"]]
                    .rename(columns={"p_low_devig": "probability"})
                    .to_dict("records")
                ),
                "quote_count": int(len(group)),
                "latest_quote_ts": group["quote_ts"].max(),
            }
        )

    consensus = pd.DataFrame(grouped_rows).sort_values(["side", "team_low_id", "team_high_id"])
    consensus.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
