"""Splits are time-ordered, non-overlapping, and stop before the data outage."""
import numpy as np
import pandas as pd
import pytest

from src.data import DATA_CUTOFF, TEST_END, TEST_START, VAL_START, assign_split
from src.evaluate import fold_edges
from src.target import HORIZONS, hourly_grid
from src.train import RECENT_START, make_X, training_mask

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


def _toy_frame() -> pd.DataFrame:
    t = pd.date_range("2026-01-01", periods=24 * 40, freq="h", tz="UTC")
    active = (np.arange(len(t)) % 7 == 0).astype(float)
    frames = []
    for oblast in ("Kyiv City", "Odeska oblast"):
        df = pd.DataFrame({"oblast": oblast, "t": t, "active": active})
        for h in HORIZONS:
            df[f"y_h{h}"] = 1.0
            df[f"onset_h{h}"] = np.where(active == 1, np.nan, 1.0)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_purge_rule_training_outcomes_close_before_the_fit_cutoff():
    df = _toy_frame()
    for a, _ in fold_edges(pd.Timestamp("2026-01-20", tz="UTC"), pd.Timestamp("2026-02-09", tz="UTC"), 4):
        for h in HORIZONS:
            for kind in ("y", "onset"):
                target = f"{kind}_h{h}"
                mask = training_mask(df, target, a, "all")
                window_close = df.loc[mask, "t"] + pd.Timedelta(hours=h)
                assert window_close.max() <= a
                if kind == "y":
                    assert window_close.max() == a  # the last allowed row closes exactly at the cut-off
                # every row left out either has an open outcome window or is not a row this model is asked about
                dropped = df[~mask]
                open_window = dropped["t"] + pd.Timedelta(hours=h) > a
                not_asked = (dropped["active"] == 1) if kind == "onset" else pd.Series(False, index=dropped.index)
                assert (open_window | not_asked).all()
                if kind == "onset":
                    assert (df.loc[mask, "active"] == 0).all()


def test_recent_variant_is_a_subset_starting_at_recent_start():
    df = _toy_frame()
    df["t"] = df["t"] + (RECENT_START - pd.Timestamp("2026-01-15", tz="UTC"))
    cut = df["t"].max()
    recent, full = training_mask(df, "y_h3", cut, "recent"), training_mask(df, "y_h3", cut, "all")
    assert (full | ~recent).all() and recent.sum() < full.sum()
    assert df.loc[recent, "t"].min() == RECENT_START


def test_folds_tile_the_test_window_once():
    folds = fold_edges(TEST_START, TEST_END, 4)
    assert folds[0][0] == TEST_START and folds[-1][1] == TEST_END
    assert all(b - a == pd.Timedelta(days=7) for a, b in folds)
    assert all(folds[i][1] == folds[i + 1][0] for i in range(3))


def test_model_inputs_refuse_target_columns():
    df = _toy_frame()
    with pytest.raises(AssertionError):
        make_X(df, ["active", "y_h1"])
    with pytest.raises(AssertionError):
        make_X(df, ["active", "onset_h3"])
    with pytest.raises(AssertionError):
        make_X(df, ["active", "t"])
    assert list(make_X(df, ["active"]).columns) == ["active", "oblast"]
