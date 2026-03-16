# Rolling Tournament Backtests

## Men

- `prior` weighted Brier: `0.1970` via `('adj_margin_rating_diff', 'recent_margin_diff', 'conference_net_eff_diff')` with `C=0.01`; global shrink `1.00`
  2021: `0.2143` across 66 games
  2022: `0.2135` across 67 games
  2023: `0.2121` across 67 games
  2024: `0.1969` across 67 games
  2025: `0.1486` across 67 games

- `seeded` weighted Brier: `0.1935` via `('seed_matchup_prior', 'seed_num_diff', 'adj_margin_rating_diff', 'recent_margin_diff')` with `C=0.01` and seed prior; global shrink `1.00`
  2021: `0.2103` across 66 games
  2022: `0.2058` across 67 games
  2023: `0.2187` across 67 games
  2024: `0.1834` across 67 games
  2025: `0.1496` across 67 games

## Women

- `prior` weighted Brier: `0.1427` via `('adj_margin_rating_diff', 'recent_margin_diff', 'conference_net_eff_diff')` with `C=3.0`; global shrink `1.00`
  2021: `0.1350` across 63 games
  2022: `0.1581` across 67 games
  2023: `0.1744` across 67 games
  2024: `0.1236` across 67 games
  2025: `0.1221` across 67 games

- `seeded` weighted Brier: `0.1404` via `('seed_num_diff', 'net_eff_diff', 'recent_margin_diff', 'conference_net_eff_diff')` with `C=0.3`; global shrink `1.00`
  2021: `0.1517` across 63 games
  2022: `0.1642` across 67 games
  2023: `0.1621` across 67 games
  2024: `0.1124` across 67 games
  2025: `0.1121` across 67 games

