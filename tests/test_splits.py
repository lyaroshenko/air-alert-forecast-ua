"""Splits are time-ordered, non-overlapping, and stop before the data outage."""
import pandas as pd

from src.data import DATA_CUTOFF, TEST_END, TEST_START, VAL_START, assign_split
from src.target import HORIZONS, hourly_grid

OUTAGE_START = pd.Timestamp("2026-08-29 04:55", tz="UTC")  # DECISIONS.md D3


def _splits() -> pd.Series:
    grid = hourly_grid(pd.Timestamp("2022-03-15 16:00", tz="UTC"), DATA_CUTOFF)
    return pd.Series(assign_split(grid.to_series()).to_numpy(), index=grid)


def test_boundaries_are_ordered_utc_hour_starts():
    assert VAL_START < TEST_START < TEST_END <= DATA_CUTOFF
    for b in (VAL_START, TEST_START, TEST_END, DATA_CUTOFF):
        assert str(b.tz) == "UTC" and b == b.floor("h")


def test_train_before_val_before_test_without_overlap():
    s = _splits()
    t = {name: s.index[s == name] for name in ("train", "val", "test")}
    assert t["train"].max() < t["val"].min()
    assert t["val"].max() < t["test"].min()
    assert len(set(t["train"]) & set(t["val"])) == 0 and len(set(t["val"]) & set(t["test"])) == 0
    assert s.index.is_monotonic_increasing and s.index.is_unique


def test_val_and_test_are_four_full_weeks():
    s = _splits()
    assert (s == "val").sum() == 28 * 24
    assert (s == "test").sum() == 28 * 24


def test_longest_horizon_never_reaches_the_cutoff_or_the_outage():
    last_test_t = _splits().loc[lambda s: s == "test"].index.max()
    target_window_end = last_test_t + pd.Timedelta(hours=max(HORIZONS))
    assert target_window_end <= DATA_CUTOFF
    assert target_window_end < OUTAGE_START
