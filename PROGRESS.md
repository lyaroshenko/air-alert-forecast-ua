# Progress (for resuming in a fresh session)

Read CLAUDE.md, DECISIONS.md, then this file. Work on `main`, commit + push after each block.
Environment: `.venv` (gitignored), run with `.venv/Scripts/python`; dataset in `external/` at commit d10c80f.

- **Done:** data/target/features/leakage test/baselines (D1–D12); P1 LightGBM + purge-safe walk-forward on test
  (D13–D15, `reports/results_test*.csv`, plots); `python -m src.run [--test]`; P2 README with results and honest
  reading; P3 importance plots + README text (y_h1 gain share of `active` is 0.62).
- **In progress:** verifying the README reproduce commands from a fresh clone (run in a short path such as
  `%TEMP%/aaf-fresh`; a long path fails on Windows with "Filename too long").
- **Next:** P4 2–7 Sep mini test (`src/robustness.py`, D16); P5 volunteer-file run (D17); README sections for both.
- **Open issues:** none. `gh` CLI is not installed; push directly to `main`.
