# Winning Plan for March Machine Learning Mania 2026

## Objective

Minimize Brier score on the actual 2026 NCAA tournament games, not on the full `132,133`-row submission.

That means:

- Treat this as a late-stage tournament forecasting problem.
- Use the full-grid submission only as a container.
- Spend almost all modeling and QA effort on the teams and matchups that can actually be scored once the 2026 bracket is known.

## Core Thesis

The highest-EV path is:

1. Build a strong all-pairs prior model now.
2. Wait for the real 2026 bracket, seeds, sites, and First Four results.
3. Pull Vegas odds from multiple books for the actual posted tournament games.
4. Convert those odds to no-vig implied probabilities.
5. Use market probabilities as the dominant signal for games with posted lines.
6. Use a bracket-aware power model for all other tournament-team pairs that could be played later.
7. Submit as late as possible on `March 19, 2026` before the Kaggle deadline.

If external odds are allowed, the plan should be explicitly market-first, not model-first.

## Why This Is the Best Shot to Win

- Only about `134` games will actually be scored across men and women.
- The deadline comes after Selection Sunday and after the First Four.
- Once the bracket is released, the problem changes from generic pairwise prediction to a much smaller set of realistic tournament paths.
- For actual games that already have sportsbook lines, the market is likely the single best feature.
- The remaining edge comes from what the market does not directly hand us:
  - future-round hypothetical matchups,
  - women-specific hosting/site structure,
  - clean no-vig consensus construction,
  - calibration,
  - and disciplined late execution.

## Non-Negotiable Principles

- Separate men and women end to end.
- Optimize for actual scored games, not the full submission uniformly.
- Use seeds as a prior, not as the whole model.
- Trust market odds heavily where they exist.
- Build pair probabilities for tournament teams after the bracket is known, not just generic regular-season strength estimates.
- Calibrate probabilities for Brier score, not for picking winners.
- Manual final submission selection on Kaggle. No reliance on auto-select.

## Model Stack

## 1. Full-Grid Prior Model

Build a probability for every possible 2026 matchup before the bracket is known.

Purpose:

- Fill the entire submission cleanly.
- Provide fallback probabilities for non-tournament teams and non-posted matchups.
- Provide a prior for later-round tournament pairings that do not have direct market lines.

Recommended structure:

- Separate men and women models.
- Main target: probability lower `TeamID` beats higher `TeamID`.
- Core features:
  - regular-season adjusted efficiency / margin features,
  - recency-weighted team strength,
  - conference strength,
  - seed-informed priors once seeds arrive,
  - men-only Massey consensus,
  - women-specific hosting/site features where inferable after the bracket.

Recommended modeling family:

- Bradley-Terry / logistic pair model on team strength differences.
- Add calibration layer on top.
- Keep it interpretable and stable. Do not chase a giant black-box stack unless it clearly wins out-of-fold.

## 2. Market Layer for Actual Posted Games

For games with sportsbook odds available, use market consensus as the primary probability source.

Data to ingest:

- moneylines from multiple books,
- point spreads,
- totals,
- timestamp of quote,
- book identity,
- open and latest line if available.

Process:

1. Map sportsbook team names to Kaggle team IDs.
2. Convert moneylines to implied probabilities.
3. Remove vig using a two-outcome normalization.
4. Build a consensus probability across books.
5. Prefer the most liquid / sharp books when available.
6. Use the freshest pre-deadline consensus, not stale overnight lines.

Default rule:

- If a real tournament game has a reliable market line, market should carry most of the weight.

## 3. Bracket-Aware Tournament Pair Model

This is where the extra EV lives after direct market lines.

Once the bracket is released:

- identify the actual 68-team field,
- identify exact seeds and regions,
- identify First Four slots,
- determine which pairs can meet and in what round/site context.

Important implication:

- A pair is not just "Team A vs Team B".
- After Selection Sunday, many pairs imply a specific round and site type if they ever occur.

Examples:

- Two teams from different regions can only meet in the Final Four or title game.
- Two teams in the same region can only meet on that region path.
- In the women’s tournament, early-round site/hosting structure can materially change the true probability of a pair.

Recommended approach:

- Build latent team ratings after bracket release using:
  - pre-tournament prior model outputs,
  - real seeds,
  - real posted market lines,
  - optionally futures prices if available.
- Use those latent ratings to produce pairwise probabilities for all tournament-team pairs, including later-round hypothetical matchups without direct lines.

## 4. Blend Rule

For tournament games or tournament-team pairs:

`p_final = w_market * p_market + w_model * p_bracket_model + w_prior * p_prior`

with:

- `w_market` highest when a direct market line exists,
- `w_bracket_model` highest for later-round tournament pairs with no direct line,
- `w_prior` mainly a stabilizer.

Target weighting philosophy:

- Men:
  - very high market weight when direct odds exist.
  - smaller but useful role for seed, Massey, efficiency, and consensus rating features.
- Women:
  - still market-first when lines exist,
  - but invest more in bracket/site/hosting adjustments because the structure differs and markets can be thinner.

## Backtesting Plan

Backtest exactly the scenario we expect to face in 2026.

## A. Pre-Bracket Model Backtest

Train on earlier seasons, predict later seasons, evaluate on NCAA tournament games only.

Measure:

- Brier score overall,
- Brier by gender,
- Brier by round,
- Brier by seed-difference bucket,
- Brier on favorites vs toss-ups.

## B. Post-Bracket / Late-Submission Simulation

For past tournaments:

- use the real seeds and slots,
- use only information that would have been available by the deadline,
- simulate First Four resolution,
- simulate which direct lines would have existed at submission time,
- compare:
  - prior model only,
  - market only,
  - market + bracket model,
  - market + bracket model + calibration.

This is the benchmark that matters.

## C. Calibration Backtest

Test:

- raw market no-vig probabilities,
- raw model probabilities,
- blended probabilities,
- isotonic calibration,
- Platt / beta calibration.

Select the version with the best rolling Brier, not the prettiest calibration curve.

## Highest-EV Enhancements

## 1. Women-Specific Site Logic

Treat this as a first-class feature, not a footnote.

- Early women’s tournament games are not structurally the same as men’s.
- Hosting and site context can matter materially.
- Build pair probabilities using actual bracket path and likely site context.

## 2. Futures-Derived Team Strength

If accessible, use futures markets to improve latent ratings for later-round hypothetical matchups.

Examples:

- title odds,
- Final Four odds,
- region odds,
- Sweet 16 / Elite Eight advancement props.

These can help infer team strength for future matchups where no direct head-to-head line exists yet.

## 3. Market Quality Controls

Do not use a naive average of all books.

Preferred process:

- drop stale books,
- identify outliers,
- favor sharper books,
- use weighted median or robust average,
- store timestamps and line movement.

If markets disagree materially, prefer the freshest sharp consensus.

## 4. Manual Exception Layer

Have a very small list of manual overrides for:

- confirmed major injuries,
- suspensions,
- coach/news shocks,
- obvious team-mapping errors,
- broken or stale odds pulls.

This should be small and deliberate. Do not turn the final step into vibes.

## Execution Timeline

## Now

- Build and validate the full-grid prior model.
- Build tournament-only backtest harness.
- Build sportsbook ingestion and team-name mapping.
- Build no-vig converter and consensus builder.
- Build bracket-aware pair generator.
- Build calibration pipeline.
- Build final submission override logic:
  - start from full-grid prior,
  - overwrite tournament-team pairs after the bracket is known,
  - overwrite direct posted games with market-dominant probabilities.

## Selection Sunday: March 15, 2026

- Ingest official 2026 seeds and bracket.
- Resolve team IDs for the full field.
- Generate all tournament-team pairs.
- Attach round/path/site context to each pair.

## First Four: March 17-18, 2026

- Replace play-in uncertainty with actual winners.
- Refresh direct market lines for the now-known Round 1 games.
- Recompute bracket-aware probabilities for downstream pairs involving those teams.

## Submission Day: March 19, 2026

Run this sequence as late as possible before the `4:00 PM UTC` deadline:

1. Refresh odds.
2. Rebuild no-vig consensus.
3. Recompute tournament-team pair probabilities.
4. Run calibration.
5. Generate final submission file.
6. Run QA checks.
7. Submit manually.
8. Confirm the correct submission is selected.

## QA Checklist

- Every row in `SampleSubmissionStage2.csv` has a prediction.
- Every prediction is strictly between `0` and `1`.
- Lower `TeamID` convention is respected.
- Men and women are never mixed.
- All actual tournament teams are correctly mapped.
- First Four losers do not accidentally drive downstream direct-game overrides.
- Posted market games reflect the latest intended consensus.
- Pair probabilities are symmetric by construction:
  - if internal model says `P(A beats B)`, submission row is consistent with lower-ID orientation.
- Final file row count matches sample exactly.

## Recommended Resource Allocation

If time is limited:

1. Build the late market ingestion and consensus pipeline first.
2. Build the bracket-aware tournament-pair override layer second.
3. Build women-specific site logic third.
4. Improve the generic full-grid prior model fourth.
5. Only after that, spend time on more exotic model stacking.

## What We Should Not Do

- Do not spend most effort squeezing tiny gains on non-tournament-team pairs.
- Do not rely on a naive season-long Elo as the main model.
- Do not use one shared model for men and women.
- Do not trust a single sportsbook blindly.
- Do not average viggy probabilities without normalizing.
- Do not submit early unless forced.
- Do not optimize for bracket-picking intuition over Brier calibration.

## Final Recommended Strategy

The most likely winning approach is:

- a separate men’s and women’s prior model,
- a late bracket-conditioned tournament override layer,
- direct use of multi-book no-vig Vegas consensus for posted games,
- latent rating inference for later-round tournament pairs,
- women-specific site/hosting treatment,
- and a disciplined final-day execution process.

If we execute that cleanly, this is the highest-probability path to actually winning the competition rather than just building a respectable generic model.
