# Progress (for resuming in a fresh session)

Read CLAUDE.md, DECISIONS.md, then this file. Work on `main`, commit + push after each block.
Environment: `.venv` (gitignored), run with `.venv/Scripts/python`; dataset in `external/` at commit d10c80f.

- **Done:** data/target/features/leakage test/baselines (D1–D12); P1 LightGBM + purge-safe walk-forward on test
  (D13–D15, `reports/results_test*.csv`, `calibration_*.png`, `importance_*.png`); `python -m src.run [--test]`.
- **In progress:** P2 README (results tables, reproduce section verified from a fresh clone).
- **Next:** P3 importance text in README (plots and `reports/importance_test.csv` already exist; y_h1 gain share of
  `active` is 0.62); P4 2–7 Sep mini test; P5 volunteer-file run.
- **Open issues:** none. `gh` CLI is not installed; push directly to `main`.
