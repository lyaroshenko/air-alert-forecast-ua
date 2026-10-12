"""Baselines only read outcomes that were fully observed before the forecast time."""
import numpy as np
import pandas as pd

from src import baselines
from src.evaluate import metrics


def toy(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    frames = []
    for oblast in ("A", "B"):
        in_hour = (rng.random(n) < 0.3).astype(float)
        df = pd.DataFrame({"oblast": oblast, "t": t, "active": (rng.random(n) < 0.2).astype(float)})
        csum = np.concatenate([[0], np.cumsum(in_hour)])
        for h in (1, 3):
            y = np.full(n, np.nan)
            y[: n - h + 1] = (csum[h:] - csum[: n - h + 1]) > 0
            df[f"y_h{h}"] = y
            df[f"onset_h{h}"] = np.where(df["active"] == 1, np.nan, y)
        local = t.tz_convert("Europe/Kyiv")
        df["hour_local"], df["dow_local"] = local.hour.astype(float), local.dayofweek.astype(float)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_baselines_at_t_ignore_outcomes_not_yet_observed():
    df = toy()
    cut = df["t"].iloc[400]
    fit_end = df["t"].iloc[300]
    for target in ("y_h1", "y_h3", "onset_h1", "onset_h3"):
        horizon = baselines.horizon_of(target)
        full = baselines.all_baselines(df, target, fit_end)
        # erase every outcome whose window [s, s+H) had not closed by `cut`
        hidden = df.copy()
        hidden.loc[hidden["t"] + pd.Timedelta(hours=horizon) > cut, [f"y_h{horizon}", f"onset_h{horizon}"]] = np.nan
        rebuilt = baselines.all_baselines(hidden, target, fit_end)
        at_cut = df["t"] == cut
        pd.testing.assert_frame_equal(full[at_cut], rebuilt[at_cut])


def test_seasonal_naive_is_last_weeks_value_within_the_same_oblast():
    df = toy()
    p = baselines.seasonal_naive(df, "y_h1")
    b = df[df["oblast"] == "B"]
    assert p[b.index[:168]].isna().all()
    assert (p[b.index[168:]].to_numpy() == b["y_h1"].to_numpy()[:-168]).all()


def test_historical_rate_uses_only_rows_before_fit_end():
    df = toy()
    fit_end = df["t"].iloc[300]
    changed = df.copy()
    changed.loc[changed["t"] >= fit_end, "y_h1"] = 1.0
    pd.testing.assert_series_equal(baselines.historical_rate(df, "y_h1", fit_end),
                                   baselines.historical_rate(changed, "y_h1", fit_end))


def test_metrics_on_known_values():
    m = metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert m["n"] == 4 and m["base_rate"] == 0.5
    assert np.isclose(m["brier"], (0.01 + 0.16 + 0.4225 + 0.04) / 4)
    assert np.isclose(m["roc_auc"], 0.75)
    assert np.isnan(metrics([0, 0, 1], [0.0, 0.0, 0.0])["roc_auc"])
    assert metrics([0, 1, np.nan], [0.2, np.nan, 0.9])["n"] == 1
