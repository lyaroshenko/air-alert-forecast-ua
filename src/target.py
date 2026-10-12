"""Hourly UTC grid and forecast targets built from merged alert intervals.

Conventions (t is the start of an hour, intervals are half-open [started_at, finished_at)):
- alert_in_hour(t) = 1 if any alert overlaps [t, t+1h).
- active_at(t)     = 1 if an alert started strictly before t and has not ended by t.
- y_h{H}(t)        = 1 if any alert overlaps [t, t+H). NaN when t+H runs past the end of the grid.
- onset_h{H}(t)    = y_h{H}(t) on hours where active_at(t) == 0, NaN elsewhere. When no alert is active
                     at t, "any alert in [t, t+H)" and "a new alert starts in [t, t+H)" are the same event.

alert_in_hour(t) and the targets describe the future of t; only active_at(t) is known at t.
"""
import logging

import numpy as np
import pandas as pd

from src.data import DATA_CUTOFF, PROCESSED_DIR

logger = logging.getLogger(__name__)

HORIZONS = (1, 3, 6)
HOUR = pd.Timedelta(hours=1)


def hourly_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Hour starts t with start <= t < end, in UTC."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("grid bounds must be timezone-aware (UTC)")
    return pd.date_range(start.tz_convert("UTC").ceil("h"), end.tz_convert("UTC"), freq="h", inclusive="left")


def _hours_since(ts: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return ((ts - origin) / HOUR).to_numpy(dtype=float)


def build_targets(intervals: pd.DataFrame, grid: pd.DatetimeIndex, horizons=HORIZONS) -> pd.DataFrame:
    """One row per hour of `grid` for a single oblast's alert intervals (they may overlap)."""
    n = len(grid)
    a = _hours_since(intervals["started_at"], grid[0])
    b = _hours_since(intervals["finished_at"], grid[0])
    if (b <= a).any():
        raise ValueError("intervals must have finished_at > started_at")

    def covered(first: np.ndarray, last_exclusive: np.ndarray) -> np.ndarray:
        diff = np.zeros(n + 1, dtype=np.int64)
        np.add.at(diff, np.clip(first, 0, n), 1)
        np.add.at(diff, np.clip(last_exclusive, 0, n), -1)
        return np.cumsum(diff)[:n] > 0

    # hour i covers [i, i+1): overlaps the alert iff floor(a) <= i < ceil(b)
    in_hour = covered(np.floor(a).astype(np.int64), np.ceil(b).astype(np.int64))
    # instant i is inside the alert iff a < i < b, i.e. floor(a)+1 <= i <= ceil(b)-1
    active = covered(np.floor(a).astype(np.int64) + 1, np.ceil(b).astype(np.int64))

    out = pd.DataFrame({"alert_in_hour": in_hour.astype(np.int8), "active_at": active.astype(np.int8)}, index=grid)
    out.index.name = "t"
    csum = np.concatenate([[0], np.cumsum(in_hour)])
    for horizon in horizons:
        y = np.full(n, np.nan)
        y[: n - horizon + 1] = (csum[horizon:] - csum[: n - horizon + 1]) > 0
        out[f"y_h{horizon}"] = y
        out[f"onset_h{horizon}"] = np.where(active, np.nan, y)
    return out


def build(intervals: pd.DataFrame, end: pd.Timestamp = DATA_CUTOFF) -> pd.DataFrame:
    """Targets for every oblast in `intervals` on a shared grid from the first alert hour to `end`."""
    grid = hourly_grid(intervals["started_at"].min().floor("h"), end)
    parts = [build_targets(g, grid).assign(oblast=oblast).reset_index()
             for oblast, g in intervals.groupby("oblast", sort=True)]
    out = pd.concat(parts, ignore_index=True)
    logger.info("hourly grid %s .. %s: %d hours x %d oblasts", grid[0], grid[-1], len(grid), len(parts))
    return out


if __name__ == "__main__":
    from src.data import assign_split

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.width", 200)
    intervals = pd.read_parquet(PROCESSED_DIR / "intervals.parquet")
    hourly = build(intervals)
    hourly.to_parquet(PROCESSED_DIR / "hourly.parquet", index=False)
    split = assign_split(hourly["t"])
    cols = ["alert_in_hour", "active_at"] + [f"{k}_h{h}" for h in HORIZONS for k in ("y", "onset")]
    print("\nmean of each column by oblast and split (onset_* averaged over hours with no active alert):")
    print(hourly.groupby(["oblast", split])[cols].mean().round(3))
    print("\nhours per oblast and split, and hours eligible for the onset target:")
    print(hourly.assign(onset_eligible=hourly["active_at"] == 0).groupby(["oblast", split])
          .agg(hours=("t", "size"), first=("t", "min"), last=("t", "max"), onset_eligible=("onset_eligible", "sum")))
