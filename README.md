# March Machine Learning Mania 2026

March Madness probability modeling. I incorporate both the Kaggle data (For all matchups), Vegas odds, and Nate Silver's projections (For R64 matchups). 

## What This Repo Does

- Builds end-of-regular-season team features from the Kaggle data. Features are just a fancy way of saying all the different metrics Kaggle provides (Off. efficiency, Point differencial, etc)
- Tunes simple pairwise tournament models against rolling historical Brier. This means we build all head to head tournament matchups, then tune the parameters to MINIMIZE the brier score on past tournaments. 
- Builds a live tournament submission that combines the prior, the seeded tournament model, Silver predictions, and posted moneylines.

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
- Women seeded: `0.1404`

Detailed results live in [outputs/reports/rolling_backtests.md](/outputs/reports/rolling_backtests.md).

## Repo Layout

- `data/`: raw Kaggle CSVs
- `context/`: competition rules, dataset notes, and strategy docs
- `src/mmmania/`: loaders, features, modeling, odds helpers
- `scripts/`: runnable entry points
- `templates/`: schemas for future external inputs
- `outputs/`: generated reports, features, submissions
- `context/plan.md`: high-level competition strategy

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

Build the current live hybrid submission:

```bash
python scripts/build_live_submission.py
```

Build market consensus once real odds exist:

```bash
python scripts/build_market_consensus.py
```

Build the live men’s board for posted tournament games:

```bash
python scripts/build_mens_tournament_board.py
```

Build the simple team-name matchup lookup tool:

```bash
python scripts/build_matchup_lookup_tool.py
```

Full rebuild from raw Kaggle files to both submission artifacts:

```bash
pytest -q
python scripts/export_feature_snapshots.py
python scripts/run_backtests.py
python scripts/build_prior_submission.py
python scripts/build_market_consensus.py
python scripts/build_live_submission.py
python scripts/build_mens_tournament_board.py
python scripts/build_matchup_lookup_tool.py
```

Prior-only rebuild:

```bash
pytest -q
python scripts/export_feature_snapshots.py
python scripts/run_backtests.py
python scripts/build_prior_submission.py
```

## External Inputs

Optional historical/current rating features:

- input: `inputs/external/team_features.csv`
- schema: `templates/external_team_features_template.csv`

Any numeric columns in that file are merged automatically as model features.

Tournament moneylines:

- input: `inputs/odds/tournament_moneylines.csv`
- schema: `templates/tournament_moneylines_template.csv`

Optional men-only BPI straight-up probabilities for posted games:

- input: `inputs/odds/men_bpi_probabilities.csv`

Optional men-only Nate Silver forecast files for the first four and round of 64:

- input directory: `data/silver/`

The market script converts American moneylines to no-vig consensus probabilities for the lower `TeamID`.

In the current setup, posted moneylines are the only live market input used by the final submission pipeline.

Silver and BPI are additional men-only external forecast sources used inside the live tournament layer and the direct posted-game overrides.

## Main Outputs

- `outputs/backtests/rolling_backtests.json`
- `outputs/reports/rolling_backtests.md`
- `outputs/features/men_team_features_2026.csv`
- `outputs/features/women_team_features_2026.csv`
- `outputs/markets/market_consensus.csv`
- `outputs/markets/men_tournament_board_2026.csv`
- `outputs/submissions/prior_submission_2026.csv`
- `outputs/submissions/live_submission_2026.csv`
- `outputs/tools/matchup_lookup_2026.html`
- `outputs/reports/live_submission_summary_2026.json`

`prior_submission_2026.csv` is only a fallback baseline. It does not use real 2026 seeds or live odds.

`live_submission_2026.csv` is the current competition-ready artifact:

- prior model for all teams,
- seeded model for tournament-team pairs,
- a men-only tournament rating layer that uses seeded probabilities, posted moneylines, optional Nate Silver probabilities, and optional BPI win probabilities,
- market-primary overrides for posted games, with an intentionally heavier Silver contribution on men’s Round of 64 rows so strong Silver positions also influence downstream later-round matchups.

`men_tournament_board_2026.csv` compares the seeded model, current submission probability, and market on posted men’s games.

`matchup_lookup_2026.html` is a small self-contained browser tool that lets you type two team names and see the exact matchup probability from `live_submission_2026.csv`.
