"""Main leakage guard: features at t must be reproducible from the alerts as known at t.

For sampled forecast times t, the raw alerts are truncated at t (alerts starting at or after t removed,
end times >= t erased), features are rebuilt from that view alone, and compared with the full pipeline.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import DATA_CUTOFF, OBLASTS, PROCESSED_DIR, TEST_START
from src.features import AlertState, build_features, truncate_alerts
from src.target import build_targets, hourly_grid

RNG_SEED = 20261012


def synthetic_alerts(n: int = 4000) -> pd.DataFrame:
    """Random overlapping alerts in 6 oblasts over 120 days, on whole seconds, some on exact hour marks."""
    rng = np.random.default_rng(RNG_SEED)
    names = OBLASTS + ["Kyivska oblast", "Mykolaivska oblast", "Kharkivska oblast"]
    start_s = rng.integers(0, 120 * 86400, n)
    start_s[: n // 20] -= start_s[: n // 20] % 3600  # exact hour starts
    dur_s = rng.choice([60, 900, 1800, 3600, 7200, 20000, 90000], n) + rng.integers(0, 600, n)
    dur_s[n // 20: n // 10] = 3600 - (start_s[n // 20: n // 10] % 3600)  # ends exactly on an hour mark
    origin = pd.Timestamp("2026-01-01", tz="UTC")
    return pd.DataFrame({
        "oblast": rng.choice(names, n),
        "started_at": origin + pd.to_timedelta(start_s, unit="s"),
        "finished_at": origin + pd.to_timedelta(start_s + dur_s, unit="s"),
    })


def assert_features_reproducible(alerts: pd.DataFrame, grid: pd.DatetimeIndex, sample: pd.DatetimeIndex, builder):
    for oblast in OBLASTS:
        full = builder(alerts, oblast, grid)
        for t in sample:
            known_at_t = truncate_alerts(alerts, t)
            assert (known_at_t["started_at"] < t).all()
            assert not (known_at_t["finished_at"] >= t).any()
            rebuilt = builder(known_at_t, oblast, pd.DatetimeIndex([t]))
            pd.testing.assert_series_equal(rebuilt.iloc[0], full.loc[t], check_names=False, rtol=1e-9, atol=1e-9,
                                           obj=f"{oblast} @ {t}")


def test_features_reproducible_from_truncated_synthetic_alerts():
    alerts = synthetic_alerts()
    grid = hourly_grid(pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-05-05", tz="UTC"))
    sample = grid[np.random.default_rng(RNG_SEED).choice(len(grid), 60, replace=False)]
    assert_features_reproducible(alerts, grid, sample, build_features)


def test_the_check_catches_a_leaky_feature():
    """Negative control: a feature that looks one hour ahead must fail the same check."""

    def leaky_builder(alerts, oblast, grid):
        out = build_features(alerts, oblast, grid)
        g = alerts[alerts["oblast"] == oblast]
        state = AlertState(g["started_at"], g["finished_at"])
        x = np.asarray((grid - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1), dtype=float)
        out["leak_frac_next_hour"] = state.frac(x, x + 3600)
        return out

    alerts = synthetic_alerts()
    grid = hourly_grid(pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-05-05", tz="UTC"))
    sample = grid[np.random.default_rng(RNG_SEED).choice(len(grid), 60, replace=False)]
    with pytest.raises(AssertionError):
        assert_features_reproducible(alerts, grid, sample, leaky_builder)


def test_truncation_erases_everything_at_or_after_t():
    alerts = synthetic_alerts(500)
    t = pd.Timestamp("2026-02-01 00:00", tz="UTC")
    known = truncate_alerts(alerts, t)
    assert known["started_at"].max() < t
    assert known["finished_at"].dropna().max() < t
    running = alerts[(alerts["started_at"] < t) & (alerts["finished_at"] >= t)]
    assert known["finished_at"].isna().sum() == len(running) > 0


def test_active_feature_matches_target_side_state():
    alerts = synthetic_alerts()
    grid = hourly_grid(pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-05-05", tz="UTC"))
    for oblast in OBLASTS:
        feats = build_features(alerts, oblast, grid)
        targets = build_targets(alerts[alerts["oblast"] == oblast], grid)
        assert (feats["active"].to_numpy() == targets["active_at"].to_numpy()).all()
        # frac_1h at t+1 is the share of hour [t, t+1) under alert: > 0 exactly when the target saw an alert
        assert ((feats["frac_1h"].to_numpy()[1:] > 0) == (targets["alert_in_hour"].to_numpy()[:-1] == 1)).all()


@pytest.mark.skipif(not (PROCESSED_DIR / "alerts.parquet").exists(), reason="run `python -m src.data` first")
def test_features_reproducible_from_truncated_real_alerts():
    alerts = pd.read_parquet(PROCESSED_DIR / "alerts.parquet")
    grid = hourly_grid(alerts["started_at"].min().floor("h"), DATA_CUTOFF)
    rng = np.random.default_rng(RNG_SEED)
    recent = grid[grid >= TEST_START]
    sample = pd.DatetimeIndex(list(grid[rng.choice(len(grid), 25, replace=False)])
                              + list(recent[rng.choice(len(recent), 15, replace=False)]))
    assert_features_reproducible(alerts, grid, sample, build_features)
