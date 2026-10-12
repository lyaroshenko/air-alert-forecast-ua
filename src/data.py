"""Load the official alerts CSV, clean it, and build merged alert intervals per oblast.

All timestamps are UTC (see DECISIONS.md D4). Every filter logs row counts before and after.
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "external" / "ukrainian-air-raid-sirens-dataset" / "datasets" / "official_data_en.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

OBLASTS = ["Kyiv City", "Odeska oblast", "Poltavska oblast"]

# Half-open UTC windows, see DECISIONS.md D3.
DATA_CUTOFF = pd.Timestamp("2026-08-29 00:00", tz="UTC")  # nothing at or after this is used
VAL_START = pd.Timestamp("2026-07-03 00:00", tz="UTC")
TEST_START = pd.Timestamp("2026-07-31 00:00", tz="UTC")
TEST_END = pd.Timestamp("2026-08-28 00:00", tz="UTC")

RAW_COLUMNS = ["oblast", "raion", "hromada", "level", "started_at", "finished_at", "source"]
IMPUTED_END_SECONDS = 3600  # the upstream processor closes orphaned alerts at start + 1 h (D7)


def assign_split(t: pd.Series) -> pd.Series:
    """Time-ordered split label for forecast times t: train / val / test / unused (>= TEST_END)."""
    out = pd.Series("unused", index=t.index, dtype=object)
    out[t < TEST_END] = "test"
    out[t < TEST_START] = "val"
    out[t < VAL_START] = "train"
    return out


def _log_filter(name: str, before: int, after: int) -> None:
    logger.info("%-34s %7d -> %7d  (dropped %d)", name, before, after, before - after)


def load_alerts(path: Path = RAW_CSV, oblasts: list[str] | None = None,
                cutoff: pd.Timestamp = DATA_CUTOFF) -> pd.DataFrame:
    """Return cleaned raw alerts (one row per official alert at any level), sorted by started_at."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != RAW_COLUMNS:
        raise ValueError(f"unexpected columns: {list(df.columns)}")
    logger.info("%-34s %7d rows", "read " + Path(path).name, len(df))

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

    if oblasts is not None:
        unknown = set(oblasts) - set(df["oblast"])
        if unknown:
            raise ValueError(f"oblasts not in data: {sorted(unknown)}")
        n = len(df)
        df = df[df["oblast"].isin(oblasts)]
        _log_filter(f"not in {len(oblasts)} selected oblasts", n, len(df))

    duration = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    df = df.assign(end_imputed=duration == IMPUTED_END_SECONDS)
    return df.sort_values(["started_at", "oblast", "raion", "hromada"]).reset_index(drop=True)


def merge_intervals(alerts: pd.DataFrame) -> pd.DataFrame:
    """Union of alerts at any level inside each oblast -> non-overlapping [started_at, finished_at) intervals.

    Touching or overlapping alerts are merged, so "oblast under alert" means the same thing before and
    after the raion reform (D6).
    """
    rows = []
    for oblast, g in alerts.sort_values("started_at").groupby("oblast", sort=True):
        cur_start = cur_end = None
        n_merged = 0
        for start, end in zip(g["started_at"], g["finished_at"]):
            if cur_start is not None and start <= cur_end:
                cur_end = max(cur_end, end)
                n_merged += 1
                continue
            if cur_start is not None:
                rows.append((oblast, cur_start, cur_end, n_merged))
            cur_start, cur_end, n_merged = start, end, 1
        if cur_start is not None:
            rows.append((oblast, cur_start, cur_end, n_merged))
    out = pd.DataFrame(rows, columns=["oblast", "started_at", "finished_at", "n_alerts"])
    logger.info("%-34s %7d -> %7d  (alerts -> merged oblast intervals)", "merge_intervals", len(alerts), len(out))
    return out


def build(oblasts: list[str] = OBLASTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write alerts.parquet (all oblasts, for cross-oblast features) and intervals.parquet (selected oblasts)."""
    alerts = load_alerts()
    unknown = set(oblasts) - set(alerts["oblast"])
    if unknown:
        raise ValueError(f"oblasts not in data: {sorted(unknown)}")
    selected = alerts[alerts["oblast"].isin(oblasts)]
    logger.info("%-34s %7d of %7d alerts are in %s", "selected oblasts", len(selected), len(alerts), oblasts)
    intervals = merge_intervals(selected)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    alerts.to_parquet(PROCESSED_DIR / "alerts.parquet", index=False)
    intervals.to_parquet(PROCESSED_DIR / "intervals.parquet", index=False)
    return selected, intervals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.width", 200)
    alerts, intervals = build()
    dur = (alerts["finished_at"] - alerts["started_at"]).dt.total_seconds()
    print("\nraw alerts per oblast and level:")
    print(alerts.groupby(["oblast", "level"]).size().unstack(fill_value=0))
    print("\nper oblast: first/last start, imputed ends, > 24 h, alerts ending after the cutoff")
    print(alerts.assign(gt_24h=dur > 86400, past_cutoff=alerts["finished_at"] > DATA_CUTOFF).groupby("oblast").agg(
        first=("started_at", "min"), last=("started_at", "max"), end_imputed=("end_imputed", "sum"),
        gt_24h=("gt_24h", "sum"), past_cutoff=("past_cutoff", "sum")))
    idur = (intervals["finished_at"] - intervals["started_at"]).dt.total_seconds() / 60
    print("\nmerged intervals per oblast (duration in minutes):")
    print(idur.groupby(intervals["oblast"]).describe(percentiles=[.5, .95, .99]).round(1))
    print("\n5 longest merged intervals:")
    print(intervals.assign(hours=(idur / 60).round(1)).nlargest(5, "hours").to_string(index=False))
