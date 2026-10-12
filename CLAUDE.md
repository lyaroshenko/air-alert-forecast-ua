# CLAUDE.md — Air Raid Alert Forecasting (Ukraine)

## Context
Selection-pipeline pet project (Stage 2). **Hard deadline: Mon 12 Oct 2026, 23:59 Kyiv time.**
Graders evaluate the *process* (AI interaction log) as much as the result: they want to see
iteration, verification, and corrected mistakes — not blindly accepted output.
Deliverables: working Python code in a GitHub repo, clear README with results, and evaluation on the
**latest available period** with **zero data leakage**.

## Task definition
- **Unit:** oblast (start with 1–3 oblasts, e.g. Kharkiv, Kyiv city, Lviv — contrasting regimes; scale to all if time allows).
- **Frequency:** hourly grid, timestamps in UTC.
- **Forecast time `t`:** the start of an hour. A forecast made at `t` may only use information available strictly before `t`.
- **Primary target:** `y(t, H) = 1` if the oblast is under an air raid alert at any moment in `[t, t+H)`, with H = 1h (also report H = 3h and 6h if time allows).
- **Secondary (harder, more honest) target:** *alert onset* — `1` if a new alert starts in `[t, t+H)`, evaluated only on hours where no alert is active at `t`.
  Reason: for the primary target, "alert active now" makes persistence a very strong baseline; good-looking metrics may just mean the model copies current state.
- **Alert definition:** "oblast under alert" = any official alert at any level (oblast/raion/hromada) inside that oblast. Must be consistent across the whole series (see the reform in Findings).

## Data
- **Primary source:** `github.com/Vadimkin/ukrainian-air-raid-sirens-dataset` (MIT, CSV, README claims daily updates). Use the **official** dataset, not the volunteer one, unless analysis shows a reason to switch.
- Clone into `./external/` (gitignored). Record the commit hash and retrieval date in the README.
- Rejected: alerts.in.ua API (history limited to ~30 days via `week_ago`/`month_ago`, needs a token); Tyrrrz/RaidTrend (archived June 2022).

## Findings so far (status shown — verify UNVERIFIED items before relying on them)
1. UNVERIFIED — dataset freshness. "Updated daily" is a claim; check the actual max timestamp first.
2. UNVERIFIED (from a third-party project) — **raion-level reform around Aug 2025.** Before it, the oblast had a single siren; after, alerts are issued per raion (~7 independent sirens). An oblast-level target can change meaning at the reform date.
   Do NOT cap the data before the reform: the brief requires evaluating on the latest period. Instead, build a consistent "any alert in the oblast" target, and check for a structural break at the reform date (alert-hours per month, number of alerts, mean duration).
3. UNVERIFIED (third-party) — **imputed end times.** Alerts with a missing end signal reportedly get an imputed 30-minute duration; the documented `naive` flag may be missing from the `_en` CSV. Measure the share of exactly-30-min durations and decide how to treat them.
4. Timezone of raw timestamps: UNKNOWN — determine it from data and docs, don't assume.
5. Ukraine's DST ends 25 Oct 2026, after the latest data, so it is not in the test window — but earlier DST switches (Mar/Oct each year) are in the training data. Use UTC to avoid duplicate or missing local hours.

## Evaluation protocol
- **Test set:** the last 4 weeks of available data. **Validation set:** the 4 weeks before that. Train on everything earlier.
- Prefer a **walk-forward backtest** (e.g. weekly folds over the test period, retraining or at least re-featurizing with an expanding window).
- **Never** tune hyperparameters, select features, or pick thresholds while looking at test-set results. Touch the test set once, at the end.
- **Required baselines** (the model must be compared to all of them):
  1. Persistence: P(alert in next H) = current state at t (last known state before t).
  2. Seasonal naive: same hour one week earlier.
  3. Historical rate: P(alert) by hour-of-day (× weekday), estimated on training data only.
  4. Recent rate: alert frequency over the last 24h / 7d before t.
- **Main model:** LightGBM (or logistic regression as a sanity check). Don't use deep learning unless the baselines and GBM are done and there's a clear reason.
- **Metrics:** Brier score, log-loss, ROC-AUC, PR-AUC, and a calibration (reliability) plot. Accuracy alone is forbidden: the classes are imbalanced.
- Report results per oblast and per horizon in one table, including the baselines.
- Check for **regime drift:** compare model performance when trained on all history vs. recent history only (e.g. last 6–12 months).

## DO NOT — leakage and correctness rules
1. Do not use any data with timestamp >= t in features for the forecast at t.
2. Do not use `rolling()` / `expanding()` without `shift(1)` (or an equivalent explicit cutoff). The current hour must never feed its own feature.
3. Do not use the **end time or duration of an alert that is still ongoing at t** — at t, you only know that it started, not when it will end. "Alert active at t" must be computed only from alerts that started before t and had not ended before t, as known at t.
4. Do not use same-time status of neighbouring oblasts. Neighbour features must be lagged (state before t).
5. Do not fit scalers, encoders, or "average rate" features on data that includes validation or test periods.
6. Do not shuffle or use random train/test splits. Time-ordered splits only.
7. Do not resample in local time. Convert to UTC first, then build the hourly grid.
8. Do not assume column names, API endpoints, or library arguments — read the data, docs, or `help()`. If unsure, say so.
9. Do not claim something works, or report a number, without running the code and showing the actual output.
10. Do not silently drop rows. Log every filter with the row counts before and after.
11. Do not add complexity (extra models, text data, deep learning) before the baselines + LightGBM + backtest pipeline works end to end.

## Required tests (`tests/`)
- `test_leakage.py`: for a sample of forecast times t, assert that every feature can be recomputed from data truncated at t (truncate the raw alerts at t, rebuild features, compare to the full-pipeline values at t). This is the main leakage guard.
- `test_target.py`: hand-made toy alert intervals → expected hourly target values (alert crossing an hour boundary, alert shorter than an hour, overlapping alerts, alert spanning several hours).
- `test_splits.py`: train max time < val min time < test min time; no overlap.

## Project structure
```
external/                 # cloned dataset (gitignored)
data/processed/           # generated parquet files (gitignored)
src/
  data.py                 # load, clean, convert to UTC, build alert intervals per oblast
  target.py               # hourly grid + targets
  features.py             # leakage-safe features (all computed with cutoff < t)
  baselines.py
  train.py                # LightGBM
  evaluate.py             # metrics, walk-forward, plots
  run.py                  # CLI: python -m src.run --oblast ... --horizon ...
notebooks/01_eda.ipynb    # exploration only, not part of the pipeline
tests/
reports/                  # figures + results table (committed)
README.md
requirements.txt
```

## Workflow rules
- Small steps: implement → run → show output → then continue.
- Run `pytest` after every change to `target.py` or `features.py`.
- Fixed random seeds. Pin package versions in `requirements.txt`.
- Commit after each working milestone with a descriptive message.
- When you find a data issue or make a design decision, add it to `DECISIONS.md` (date, issue, options considered, choice, reason). This feeds the README and the reflection.

## Out of scope (README "future work" only, unless the core is done early)
- Telegram, news, or other text sources (high leakage risk: use publish time only, beware edited messages and posts *about* alerts that already started).
- Weapon-type modelling, raion-level forecasting.

## Time budget (today)
- by 14:00 — data verified, EDA, target built + tests
- by 17:00 — features + leakage test + baselines
- by 20:00 — LightGBM + walk-forward evaluation + results table
- by 22:30 — README, DECISIONS.md, cleanup, push
- by 23:30 — log export + reflection submitted
