"""Robustness checks outside the headline evaluation. Neither changes the default pipeline or its data cut.

python -m src.robustness --check sep        official file, 2026-09-03 -> 2026-09-06, after the outage (D16)
python -m src.robustness --check volunteer  volunteer file, 2026-09-13 -> 2026-10-11, unverifiable labels (D17)

Both reuse the model configuration chosen on the official validation split; nothing is re-tuned.
"""
import argparse
import logging

import pandas as pd

from src import baselines, features, target, train
from src.data import (OBLASTS, RAW_CSV, TEST_END, _log_filter, load_alerts, merge_intervals)
from src.evaluate import (REPORTS_DIR, TARGETS, best_baseline_on_validation, check_suspicious, compact_table, join,
                          run_window, score)

logger = logging.getLogger(__name__)


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


VOLUNTEER_CSV = RAW_CSV.with_name("volunteer_data_en.csv")
SEP_START, SEP_END = utc("2026-09-03 00:00"), utc("2026-09-06 00:00")
OUTAGE_SPAN = (utc("2026-08-30 00:00"), utc("2026-09-02 00:00"))  # alerts covering all of this are artefacts (D3)
VOL_VAL_START, VOL_TEST_START, VOL_TEST_END = utc("2026-08-16 00:00"), utc("2026-09-13 00:00"), utc("2026-10-11 00:00")
GRID_MARGIN = pd.Timedelta(hours=max(target.HORIZONS))  # so the longest target window of the last scored hour closes


def assemble(alerts: pd.DataFrame, feature_alerts: pd.DataFrame, grid_end: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """Targets from `alerts`, features from `feature_alerts`, on an hourly grid up to grid_end."""
    hourly = target.build(merge_intervals(alerts[alerts["oblast"].isin(OBLASTS)]), end=grid_end)
    grid = pd.DatetimeIndex(hourly["t"].drop_duplicates().sort_values())
    return join(features.build(feature_alerts, grid), hourly)


def label(df: pd.DataFrame, bounds: dict) -> pd.Series:
    """Split label from half-open [start, end) windows; everything else is 'other'."""
    out = pd.Series("other", index=df.index, dtype=object)
    for name, (a, b) in bounds.items():
        out[(df["t"] >= a) & (df["t"] < b)] = name
    return out


def sep_check(config: dict) -> pd.DataFrame:
    alerts = load_alerts(cutoff=SEP_END + GRID_MARGIN)
    spans = (alerts["started_at"] < OUTAGE_SPAN[0]) & (alerts["finished_at"] >= OUTAGE_SPAN[1])
    logger.info("alerts spanning the outage: %d (of which in the 3 scored oblasts: %d)", spans.sum(),
                (spans & alerts["oblast"].isin(OBLASTS)).sum())
    feature_alerts = alerts[~spans]
    _log_filter("outage-spanning alerts (features)", len(alerts), len(feature_alerts))
    df, feature_cols = assemble(alerts, feature_alerts, SEP_END + GRID_MARGIN)
    df["split"] = label(df, {"sep": (SEP_START, SEP_END)})
    logger.info("fit cut-off %s, scored window %s .. %s", TEST_END, SEP_START, SEP_END)
    res, _, _ = run_window(df, feature_cols, "sep", SEP_START, SEP_END, config, n_folds=1, fit_end=TEST_END)
    return res


def load_volunteer(cutoff: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(VOLUNTEER_CSV, dtype=str, keep_default_na=False)
    if list(df.columns) != ["region", "started_at", "finished_at", "naive"]:
        raise ValueError(f"unexpected columns: {list(df.columns)}")
    logger.info("%-34s %7d rows", "read " + VOLUNTEER_CSV.name, len(df))
    n = len(df)
    df = df.drop_duplicates()
    _log_filter("exact duplicates", n, len(df))
    for col in ["started_at", "finished_at"]:
        if not df[col].str.endswith("+00:00").all():
            raise ValueError(f"{col}: found a timestamp without an explicit +00:00 offset")
        df[col] = pd.to_datetime(df[col], utc=True)
    n = len(df)
    df = df[df["finished_at"] > df["started_at"]]
    _log_filter("non-positive duration", n, len(df))
    n = len(df)
    df = df[df["started_at"] < cutoff]
    _log_filter(f"started_at >= {cutoff:%Y-%m-%d %H:%M}", n, len(df))
    df = df.rename(columns={"region": "oblast"}).assign(end_imputed=df["naive"] == "True")
    return df.sort_values(["started_at", "oblast"]).reset_index(drop=True)


def volunteer_check(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    alerts = load_volunteer(VOL_TEST_END + GRID_MARGIN)
    df, feature_cols = assemble(alerts, alerts, VOL_TEST_END + GRID_MARGIN)
    df["split"] = label(df, {"val": (VOL_VAL_START, VOL_TEST_START), "test": (VOL_TEST_START, VOL_TEST_END)})
    in_test = df[df["split"] == "test"]
    logger.info("imputed (naive) ends among volunteer alerts starting in the test window, per oblast: %s",
                alerts[(alerts["started_at"] >= VOL_TEST_START) & (alerts["started_at"] < VOL_TEST_END)
                       & alerts["oblast"].isin(OBLASTS)].groupby("oblast")["end_imputed"].agg(["sum", "size"]).to_dict("index"))
    logger.info("test rows: %d", len(in_test))
    val = pd.concat([score(df, baselines.all_baselines(df, t, VOL_VAL_START), t, "val") for t in TARGETS], ignore_index=True)
    res, _, _ = run_window(df, feature_cols, "test", VOL_TEST_START, VOL_TEST_END, config)
    return res, best_baseline_on_validation(val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=["sep", "volunteer"], required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    config = train.load_config()
    REPORTS_DIR.mkdir(exist_ok=True)
    if args.check == "sep":
        res = sep_check(config)
        best = best_baseline_on_validation(pd.read_csv(REPORTS_DIR / "baselines_val.csv"))
        name = "results_sep_check"
        title = "SEP CHECK 2026-09-03 -> 2026-09-06 (lookback windows partially cover an outage)"
    else:
        res, best = volunteer_check(config)
        name = "results_volunteer"
        title = "VOLUNTEER FILE 2026-09-13 -> 2026-10-11 (labels unverifiable, oblast attribution unreliable)"
    check_suspicious(res)
    res.to_csv(REPORTS_DIR / f"{name}.csv", index=False, float_format="%.5f")
    table = compact_table(res, config, best)
    table.to_csv(REPORTS_DIR / f"{name}_compact.csv", index=False, float_format="%.5f")
    cols = ["oblast", "target", "n", "base_rate", "lgbm", "lgbm_brier", "lgbm_roc_auc", "lgbm_pr_auc",
            "baseline", "bl_brier", "bl_roc_auc", "bss"]
    print(f"\n{title}")
    print(table[cols].round(3).to_string(index=False))
