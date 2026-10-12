# Progress (for resuming in a fresh session)

Read CLAUDE.md, DECISIONS.md, then this file. Work on `main`, commit + push after each block.
Environment: `.venv` (gitignored), run with `.venv/Scripts/python`; dataset in `external/` at commit d10c80f.

- **Done:** data/target/features/leakage test/baselines (D1–D12); P1 LightGBM + purge-safe walk-forward on test
  (D13–D15); P2 README; P3 importance (in README section 4); P4 September check and P5 volunteer-file run
  (`src/robustness.py`, D16–D17, README section 5, `reports/results_sep_check*.csv`, `reports/results_volunteer*.csv`).
- **Reproduce check:** done 2026-10-12 from a clean clone in `%TEMP%/aaf-fresh` (deleted afterwards): `src.run --test`,
  both `src.robustness` checks and `pytest -q` (27 passed) ran clean; regenerated `reports/` identical to committed.
- **In progress / next:** nothing. P1–P5 are complete and pushed to `main`.
- **Not done (outside the repo, user's task):** AI interaction log export and the reflection for the submission.
- **Open issues:** `gh` CLI is not installed; push directly to `main`. `.idea/` is untracked on purpose.
