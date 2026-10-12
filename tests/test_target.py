"""Hand-made alert intervals -> expected hourly state and targets."""
import numpy as np
import pandas as pd
import pytest

from src.data import merge_intervals
from src.target import build_targets, hourly_grid


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(f"2026-01-01 {s}", tz="UTC")


def intervals(*pairs) -> pd.DataFrame:
    return pd.DataFrame({"started_at": [ts(a) for a, _ in pairs], "finished_at": [ts(b) for _, b in pairs]})


GRID = hourly_grid(ts("00:00"), ts("08:00"))  # t = 00:00 .. 07:00


def col(df: pd.DataFrame, name: str) -> list:
    return [None if np.isnan(x) else int(x) for x in df[name]]


def test_grid_is_utc_hour_starts():
    assert len(GRID) == 8
    assert str(GRID.tz) == "UTC"
    assert GRID[0] == ts("00:00") and GRID[-1] == ts("07:00")


def test_alert_shorter_than_an_hour():
    out = build_targets(intervals(("02:10", "02:40")), GRID)
    assert col(out, "alert_in_hour") == [0, 0, 1, 0, 0, 0, 0, 0]
    assert col(out, "active_at") == [0, 0, 0, 0, 0, 0, 0, 0]
    assert col(out, "y_h1") == [0, 0, 1, 0, 0, 0, 0, 0]
    assert col(out, "y_h3") == [1, 1, 1, 0, 0, 0, None, None]
    assert col(out, "y_h6") == [1, 1, 1, None, None, None, None, None]


def test_alert_crossing_an_hour_boundary():
    out = build_targets(intervals(("02:50", "03:10")), GRID)
    assert col(out, "alert_in_hour") == [0, 0, 1, 1, 0, 0, 0, 0]
    assert col(out, "active_at") == [0, 0, 0, 1, 0, 0, 0, 0]
    assert col(out, "y_h1") == [0, 0, 1, 1, 0, 0, 0, 0]
    # at 03:00 the alert is already active, so that hour is excluded from the onset target
    assert col(out, "onset_h1") == [0, 0, 1, None, 0, 0, 0, 0]


def test_alert_spanning_several_hours():
    out = build_targets(intervals(("01:30", "04:30")), GRID)
    assert col(out, "alert_in_hour") == [0, 1, 1, 1, 1, 0, 0, 0]
    assert col(out, "active_at") == [0, 0, 1, 1, 1, 0, 0, 0]
    assert col(out, "y_h3") == [1, 1, 1, 1, 1, 0, None, None]
    assert col(out, "onset_h3") == [1, 1, None, None, None, 0, None, None]


def test_overlapping_alerts_count_once():
    out = build_targets(intervals(("01:10", "02:20"), ("02:00", "03:30"), ("01:40", "01:50")), GRID)
    assert col(out, "alert_in_hour") == [0, 1, 1, 1, 0, 0, 0, 0]
    assert col(out, "active_at") == [0, 0, 1, 1, 0, 0, 0, 0]


def test_alert_on_exact_hour_boundaries():
    # starts exactly at 02:00: not yet "active before" 02:00; ends exactly at 04:00: hour 04 is clear
    out = build_targets(intervals(("02:00", "04:00")), GRID)
    assert col(out, "alert_in_hour") == [0, 0, 1, 1, 0, 0, 0, 0]
    assert col(out, "active_at") == [0, 0, 0, 1, 0, 0, 0, 0]
    assert col(out, "onset_h1") == [0, 0, 1, None, 0, 0, 0, 0]


def test_alerts_outside_the_grid_are_clipped():
    out = build_targets(pd.DataFrame({
        "started_at": [pd.Timestamp("2025-12-31 22:00", tz="UTC"), ts("06:30")],
        "finished_at": [ts("01:15"), pd.Timestamp("2026-01-02 03:00", tz="UTC")]}), GRID)
    assert col(out, "alert_in_hour") == [1, 1, 0, 0, 0, 0, 1, 1]
    assert col(out, "active_at") == [1, 1, 0, 0, 0, 0, 0, 1]


def test_onset_equals_primary_target_on_inactive_hours():
    out = build_targets(intervals(("00:20", "00:40"), ("02:50", "05:10"), ("06:59", "07:01")), GRID)
    for h in (1, 3, 6):
        inactive = out["active_at"] == 0
        assert out.loc[inactive, f"onset_h{h}"].equals(out.loc[inactive, f"y_h{h}"])
        assert out.loc[~inactive, f"onset_h{h}"].isna().all()


def test_non_positive_duration_is_rejected():
    with pytest.raises(ValueError):
        build_targets(intervals(("02:00", "02:00")), GRID)


def test_merge_intervals_unions_levels_within_an_oblast():
    alerts = pd.DataFrame({
        "oblast": ["A", "A", "A", "A", "B"],
        "started_at": [ts("01:00"), ts("01:30"), ts("02:00"), ts("05:00"), ts("01:10")],
        "finished_at": [ts("02:00"), ts("01:45"), ts("03:00"), ts("05:30"), ts("01:20")]})
    merged = merge_intervals(alerts)
    a = merged[merged["oblast"] == "A"]
    assert list(a["started_at"]) == [ts("01:00"), ts("05:00")]
    assert list(a["finished_at"]) == [ts("03:00"), ts("05:30")]
    assert list(a["n_alerts"]) == [3, 1]
    assert len(merged[merged["oblast"] == "B"]) == 1
