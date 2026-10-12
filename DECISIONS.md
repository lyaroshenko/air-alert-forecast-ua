# Decisions and data issues

Format: date · issue · options considered · choice · reason.
All numbers below come from scripts run on 2026-10-12 against
`Vadimkin/ukrainian-air-raid-sirens-dataset` at commit `d10c80fa696bab42822b9f05b423fc8a30e79d20`.

## D1 · 2026-10-12 · Official dataset is not "updated daily"

- **Issue.** `official_data_en.csv` ends at `started_at` 2026-09-07 05:28:22 UTC / `finished_at` 05:39:33 UTC.
  The repo still commits daily, but since 2026-09-08 only the volunteer CSVs change.
  The source Telegram channel (`t.me/s/air_alert_ua`, public web preview) shows its last post at 09:09 Kyiv time
  on 7 Sep 2026, so the source stopped, not the scraper. (Web preview only; a move elsewhere is not ruled out.)
- **Options.** (A) official file, evaluate on its last clean weeks; (B) volunteer file, fresh to 2026-10-12.
- **Choice.** A. See D5 for why B was rejected.

## D2 · 2026-10-12 · 38% of official rows are exact duplicates

- **Issue.** 299,275 rows, 185,430 distinct. The history 2022-03-15 .. 2026-01-25 was appended a second time
  (file rows 113911–227831). Every duplicated row appears exactly twice. The file is not sorted by `started_at`.
- **Choice.** Exact de-duplication on all 7 columns, logged with before/after counts; sort by `started_at`.
- **Reason.** After exact de-duplication no two rows share (location, `started_at`), so nothing else needs merging.

## D3 · 2026-10-12 · Outage inside the tail of the official data

- **Issue.** Zero official rows between 2026-08-29 04:55 and 2026-09-02 13:48 UTC (the only zero-alert days in
  1,638 days; the volunteer file has 161–180 rows/day there). 31 alerts "span" the hole with ~113–115 h durations
  because their end messages were missed.
- **Options.** (a) test = 9 Aug – 6 Sep with 106 outage hours masked (16% of test) plus masking every lagged
  feature that looks into the hole; (b) test = last 4 full weeks before the outage.
- **Choice.** (b). Train < 2026-07-03 · validation 2026-07-03 → 2026-07-31 · test 2026-07-31 → 2026-08-28 (UTC,
  half-open). Data is cut at 2026-08-29 00:00 UTC.
- **Reason.** Clean labels, no masking logic. The test end leaves 29 h before the outage, so a 6 h horizon never
  reaches it. Cost: 2–7 Sep is unused and the test period is ~6 weeks old.

## D4 · 2026-10-12 · Timestamps are UTC

- **Evidence.** Every raw string ends in `+00:00`; `datasets/README.md` says "All times are in UTC"; the processor
  writes Telethon's `message.date` unconverted; Kyiv City 2022-10-10 is stored 03:47:06 → 09:24:32, i.e. 06:47 →
  12:24 Kyiv time (the morning mass-strike alert — reference from memory, not re-sourced); the channel's last
  all-clear "08:39" Kyiv time equals the file's max `finished_at` 05:39:33 UTC.
- **Choice.** Parse with `utc=True`, build the hourly grid in UTC, never resample in local time.

## D5 · 2026-10-12 · Volunteer dataset rejected as primary source

- **What passed.** UTC, no duplicates, no overlaps within a region, no outages, fresh to 2026-10-12 04:59 UTC.
- **What failed.** After the raion reform it files alerts under the wrong oblast. Post-reform (2025-12 .. 2026-08-28),
  590 of 661 volunteer "Lviv" alerts have no official Lviv start within ±120 s; 461 start together with the same
  three Odesa raions (Podilskyi, Rozdilnianskyi, Berezivskyi). 783 of 907 volunteer "Volyn" alerts coincide with
  Prylutskyi raion (Chernihiv). Mean hourly Jaccard vs the official file over 24 oblasts: 0.746 before the reform,
  0.578 after (Kyiv City 0.777 → 0.638, Lviv 0.821 → 0.122, Volyn 0.894 → 0.106).
- **Reason.** Its test window (13 Sep – 11 Oct) has no official data to check labels against, and the known error
  grows over time. A model scored on mislabelled targets says nothing.
- **Possible extra.** Run the finished pipeline on the volunteer file as a secondary "freshest, noisy labels" result.

## D6 · 2026-10-12 · The reform happened in two steps

- **Evidence (de-duplicated monthly rows).** Raion share 0.195 in 2025-07 → 0.582 in 2025-08 (13 oblasts) →
  0.893 in 2025-11 (23 oblasts). From 2025-12 only Kyiv City still issues oblast-level alerts. Monthly row counts
  roughly quadruple (≈2,000 → ≈8,000–11,000).
- **Choice.** Target = union of all alerts at any level inside the oblast, merged into non-overlapping intervals.
  Do not cap history at the reform; check regime drift in evaluation (all history vs recent only).

## D7 · 2026-10-12 · Imputed and implausible end times (official)

- **Issue.** The "30-minute imputation" finding belongs to the volunteer file (5,834 `naive=True` rows, all exactly
  1800 s). The official file has no `naive` column and only 38 exact-1800 s rows (chance level). Its real
  imputation, read from `processors/official_channel_processor.py`: if a place gets a new alert > 3 h after an
  unfinished one, the old one is closed at `started_at + 1 h`; if ≤ 3 h, the old start is silently overwritten.
  255 distinct rows are exactly 3600 s (0.14%). 897 distinct rows last > 24 h (max 604 days, a Kharkiv hromada).
  2 rows have zero duration. No row has a missing end: unfinished alerts are never written to the CSV.
- **Choice.** Keep exact-3600 s rows and flag them (`end_imputed`); drop zero-duration rows (logged); no duration
  cap for the three chosen oblasts unless EDA on them shows artefacts.

## D8 · 2026-10-12 · Oblast choice

- **Issue.** With the "any level" definition the frontline oblasts are under alert almost always (share of hours,
  2025-12 .. 2026-08: Dnipropetrovska 1.000, Donetska 0.980, Kharkivska 0.965, Sumska 0.900, Zaporizka 0.895), so the
  target is near-constant there.
- **Choice.** Kyiv City (0.126), Odeska (0.391), Poltavska (0.502). Kharkiv from the original plan is dropped.

## D9 · 2026-10-12 · Base rates shift sharply into the validation and test windows

- **Evidence (`python -m src.target`).** Share of hours with an alert, train → val → test: Kyiv City 0.119 → 0.126 →
  0.202; Odeska 0.176 → 0.574 → 0.640; Poltavska 0.334 → 0.592 → 0.616. Odeska steps up in the reform month
  (0.129 in 2025-10 → 0.361 in 2025-11). At H = 6 h the primary target is 0.96–0.98 positive in test for Odeska
  and Poltavska, i.e. nearly constant.
- **Consequence.** H = 1 h and the onset target are the informative ones; H = 6 h is reported but says little for
  those two oblasts. Historical-rate baselines fitted on all history will be badly calibrated, which makes the
  "all history vs recent only" comparison a required part of the evaluation, not an optional one.

## D10 · 2026-10-12 · What "alert active at t" means

- **Issue.** The first version of `target.py` used start < t < end. An alert whose end message is stamped exactly
  at t would then count as "not active", although at t that end is not yet known (rule: only information strictly
  before t). The leakage test's truncated view cannot reproduce that.
- **Choice.** active_at(t) = started strictly before t and not ended before t (end >= t). Target-side `active_at`
  and feature-side `active` are computed by different code and are equal on all 117,168 rows.

## D11 · 2026-10-12 · Feature design: closed-form functions instead of rolling windows

- **Options.** (a) hourly table + `shift(1).rolling()`; (b) evaluate F(x) = alert-seconds before x and
  N(x) = alert starts before x directly from the alert intervals.
- **Choice.** (b). Every feature at t evaluates F and N only at instants <= t, plus the running-alert state, which
  needs the start and whether the end is >= t but never the end itself. The same function builds one row or the
  whole grid, which is what makes the truncation test possible.
- **Leakage test.** `tests/test_leakage.py` truncates the raw alerts at t (later starts removed, ends >= t erased),
  rebuilds the features from that view, and compares with the full pipeline: 60 random hours on synthetic alerts,
  40 on the real data (15 of them in the test window), for all three oblasts. A negative control adds a feature
  that looks one hour ahead and checks that the same test rejects it.
- **Cross-oblast features.** Neighbour and nationwide features use the other oblasts' state as known at t (running
  alert, share of the last 1/6 h under alert, starts in the last 1/3 h). Nothing from [t, t+H) of any oblast is used.
  Frontline oblasts are near-constant in these features (D8); they are kept and left for the model to ignore.

## D12 · 2026-10-12 · Baseline definitions

- Persistence = `active` at t. For the onset target it is constant 0 on every scored row, so its ROC-AUC is
  undefined and its Brier score equals the base rate.
- Seasonal naive = primary target one week earlier (also for onset targets, which are undefined on hours with a
  running alert).
- Recent rate = mean of the same target over the last 24 h / 7 d of hours whose outcome window closed by t
  (`shift(H)` before `rolling`).
- Historical rate = mean by oblast x local weekday x local hour, fitted on rows before the scored split.
- All models are scored on identical rows; `evaluate.score` raises if a forecast is missing on a scored row.
- Log-loss clips probabilities to [0.001, 0.999] so that hard 0/1 baselines stay finite.
