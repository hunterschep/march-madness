# March Machine Learning Mania 2026

Compact men-first repo for March Madness probability modeling.

## What This Repo Does

- Builds end-of-regular-season team features from the Kaggle data.
- Tunes simple pairwise tournament models against rolling historical Brier.
- Generates a pre-bracket 2026 prior submission.
- Prepares for post-bracket odds ingestion and tournament overrides.

The current strongest men’s seeded baseline is:

- `seed_matchup_prior`
- `seed_num_diff`
- `adj_margin_rating_diff`
- `recent_margin_diff`

## Current Backtests

Rolling tournament-only Brier on `2021-2025`:

- Men prior: `0.1970`
- Men seeded: `0.1935`
- Women prior: `0.1427`
- Women seeded: `0.1403`

Detailed results live in [outputs/reports/rolling_backtests.md](/Users/hunterschep/march-madness/outputs/reports/rolling_backtests.md).

## Repo Layout

- `data/`: raw Kaggle CSVs
- `src/mmmania/`: loaders, features, modeling, odds helpers
- `scripts/`: runnable entry points
- `templates/`: schemas for future external inputs
- `outputs/`: generated reports, features, submissions
- `plan.md`: high-level competition strategy

## Commands

Run tests:

```bash
pytest -q
```

Rebuild backtests:

```bash
python scripts/run_backtests.py
```

Export 2026 team snapshots:

```bash
python scripts/export_feature_snapshots.py
```

Build the current pre-bracket prior submission:

```bash
python scripts/build_prior_submission.py
```

Build market consensus once real odds exist:

```bash
python scripts/build_market_consensus.py
```

## External Inputs

Optional historical/current rating features:

- input: `inputs/external/team_features.csv`
- schema: `templates/external_team_features_template.csv`

Any numeric columns in that file are merged automatically as model features.

Tournament moneylines:

- input: `inputs/odds/tournament_moneylines.csv`
- schema: `templates/tournament_moneylines_template.csv`

The odds script converts American moneylines to no-vig consensus probabilities for the lower `TeamID`.

## Main Outputs

- `outputs/backtests/rolling_backtests.json`
- `outputs/reports/rolling_backtests.md`
- `outputs/features/men_team_features_2026.csv`
- `outputs/features/women_team_features_2026.csv`
- `outputs/submissions/prior_submission_2026.csv`

`prior_submission_2026.csv` is only a fallback baseline. It does not use real 2026 seeds or live odds.

## When The Bracket Drops

1. Refresh the Kaggle data in `data/`.
2. Re-run:

```bash
python scripts/export_feature_snapshots.py
python scripts/run_backtests.py
python scripts/build_prior_submission.py
```

3. Add real market quotes to `inputs/odds/tournament_moneylines.csv`.
4. Run `python scripts/build_market_consensus.py`.
5. Resume bracket-aware tournament overrides and final submission prep.
