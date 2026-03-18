# March Machine Learning Mania 2026

This repo builds one final submission path for Kaggle March Machine Learning Mania 2026.

The statistical core is a pooled men+women notebook-style model inspired by the 2025 public modeh7 solution:

- symmetric doubled training rows for every game
- tournament seed features
- season-average box-score and opponent box-score features
- season-reset Elo
- per-season Gaussian GLM team quality
- XGBoost regression on point differential
- spline calibration from predicted margin to win probability

That model produces the model-only baseline in `outputs/submissions/prior_submission_2026.csv`.

The competition submission in `outputs/submissions/live_submission_2026.csv` starts from that XGBoost prior, then applies the men-only tournament layer built from posted moneylines, optional BPI inputs, and optional Nate Silver probabilities.

## Trimmed Repo Layout

Only the active files for the final pipeline remain:

- `src/mmmania/modeh7.py`: pooled men+women XGBoost prior model
- `src/mmmania/tournament.py`: men-only latent tournament rating layer
- `src/mmmania/odds.py`: moneyline conversion and consensus helpers
- `src/mmmania/silver.py`: Nate Silver import and pair-probability mapping
- `src/mmmania/features.py`: active-team and tournament-seed loaders
- `src/mmmania/modeling.py`: submission ID parsing and target backtest seasons
- `src/mmmania/matchup_lookup.py`: bundle builder for the HTML probability viewer
- `scripts/`: final build entry points only
- `templates/tournament_moneylines_template.csv`: the only retained schema template
- `tests/`: coverage for the active modules only

Old snapshot exporters, external-feature scaffolding, and scratch docs were removed.

## Optimal Path

This repo is now XGBoost-only.

Current tuned defaults for the prior:

- validation: `loso`
- calibration: `by_gender`
- aggressiveness: `1.01`
- XGBoost: `eta 0.008`, `max_depth 3`, `min_child_weight 6`, `subsample 0.7`, `colsample_bynode 0.7`, `num_parallel_tree 2`, `num_rounds 900`

Latest saved audit in `outputs/reports/rolling_backtests.md`:

- fair rolling `2021-2025`: `0.1675` overall, `0.1937` men, `0.1410` women
- notebook-style LOSO `2021-2025`: `0.1676` overall, `0.1955` men, `0.1394` women
- notebook-style LOSO full history: `0.1680` overall, `0.1857` men, `0.1412` women

## Mac XGBoost Setup

The build should run under a Python where `xgboost` is installed and linked against `libomp`.

If needed:

```bash
brew install libomp
```

Before running the full build, export the Homebrew OpenMP path:

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_FALLBACK_LIBRARY_PATH
```

Quick runtime probe:

```bash
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('src').resolve())); import mmmania.modeh7 as m; print(m._xgboost_probe_error())"
```

Expected result:

```python
None
```

If the probe returns an error, do not run the final build yet. Fix the Python/XGBoost/libomp environment first.

## What The Scripts Do

- `scripts/run_backtests.py`: rebuilds the fair rolling and notebook-style LOSO audits
- `scripts/build_prior_submission.py`: trains the XGBoost prior and writes the model-only Kaggle submission
- `scripts/build_market_consensus.py`: converts posted moneylines into no-vig market probabilities
- `scripts/build_live_submission.py`: blends the XGBoost prior with the men tournament layer and writes the final submission
- `scripts/build_mens_tournament_board.py`: exports a readable board for posted men’s games
- `scripts/build_matchup_lookup_tool.py`: builds the self-contained HTML matchup viewer from the live submission

## Full Build

Run from the repo root:

```bash
pytest -q
python scripts/run_backtests.py
python scripts/build_prior_submission.py
python scripts/build_market_consensus.py
python scripts/build_live_submission.py
python scripts/build_mens_tournament_board.py
python scripts/build_matchup_lookup_tool.py
```

If you want the live build to force a fresh prior instead of reusing `outputs/submissions/prior_submission_2026.csv`:

```bash
python scripts/build_live_submission.py --refresh-prior
```

## Submit

Submit:

- `outputs/submissions/live_submission_2026.csv`

Do not submit the prior unless you explicitly want the model-only baseline.

Before uploading, verify `outputs/reports/prior_submission_summary_2026.json` contains:

- `"backend": "xgboost"`

and that `outputs/reports/live_submission_summary_2026.json` was rebuilt in the same run.

## External Inputs

Tournament moneylines:

- input: `inputs/odds/tournament_moneylines.csv`
- schema: `templates/tournament_moneylines_template.csv`

Optional men-only BPI probabilities:

- input: `inputs/odds/men_bpi_probabilities.csv`

Optional men-only Nate Silver files:

- input directory: `data/silver/`

## Main Outputs

- `outputs/backtests/rolling_backtests.json`
- `outputs/reports/rolling_backtests.md`
- `outputs/reports/prior_submission_summary_2026.json`
- `outputs/reports/live_submission_summary_2026.json`
- `outputs/markets/market_consensus.csv`
- `outputs/markets/men_tournament_board_2026.csv`
- `outputs/submissions/prior_submission_2026.csv`
- `outputs/submissions/live_submission_2026.csv`
- `outputs/tools/matchup_lookup_2026.html`
