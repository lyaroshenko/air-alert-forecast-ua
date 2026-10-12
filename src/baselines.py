"""Baseline probability forecasts. Each returns a Series aligned to `df` (one row per oblast and hour t).

`df` must be sorted by (oblast, t) on a gap-free hourly grid and hold the feature and target columns.
Everything a baseline reads at t was observable before t:
- persistence: the alert state at t,
- seasonal naive: the target one week earlier, whose window [t-168h, t-168h+H) closed long before t,
- recent rate: targets at s <= t-H, whose windows [s, s+H) close at or before t (shift(H) before rolling),
- historical rate: a lookup table fitted on rows strictly before `fit_end`.
"""
import pandas as pd

WEEK_H = 168


def horizon_of(target: str) -> int:
    return int(target.rsplit("_h", 1)[1])


def _check(df: pd.DataFrame) -> None:
    steps = df.groupby("oblast")["t"].diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise ValueError("df must be sorted by (oblast, t) on a gap-free hourly grid")


def persistence(df: pd.DataFrame, target: str) -> pd.Series:
    """P = 1 if an alert is running at t, else 0. For onset targets this is constant 0 by construction."""
    return df["active"].astype(float)


def seasonal_naive(df: pd.DataFrame, target: str) -> pd.Series:
    """Was there an alert in the same window one week earlier?

    Onset targets are undefined on hours with a running alert, so they also use last week's primary target.
    """
    _check(df)
    return df.groupby("oblast")[f"y_h{horizon_of(target)}"].shift(WEEK_H)


def recent_rate(df: pd.DataFrame, target: str, window_h: int) -> pd.Series:
    """Mean of the target over the `window_h` most recent hours whose outcome is fully known at t."""
    _check(df)
    horizon = horizon_of(target)

    def rate(col: str) -> pd.Series:
        return df.groupby("oblast")[col].transform(lambda s: s.shift(horizon).rolling(window_h, min_periods=1).mean())

    # an onset target has no defined values in a window spent entirely under alert: fall back to the primary rate
    return rate(target).fillna(rate(f"y_h{horizon}"))


def historical_rate(df: pd.DataFrame, target: str, fit_end: pd.Timestamp) -> pd.Series:
    """Mean target by oblast x local weekday x local hour, estimated on rows whose outcome closed before fit_end."""
    horizon = horizon_of(target)
    fit = df[df["t"] + pd.Timedelta(hours=horizon) <= fit_end]
    keys = ["oblast", "dow_local", "hour_local"]
    table = fit.groupby(keys)[target].mean().rename("p").reset_index()
    fallback = fit.groupby("oblast")[target].mean()
    p = df[keys].merge(table, on=keys, how="left")["p"].to_numpy()
    out = pd.Series(p, index=df.index)
    return out.fillna(df["oblast"].map(fallback))


def all_baselines(df: pd.DataFrame, target: str, fit_end: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({
        "persistence": persistence(df, target),
        "seasonal_naive_7d": seasonal_naive(df, target),
        "historical_rate": historical_rate(df, target, fit_end),
        "recent_rate_24h": recent_rate(df, target, 24),
        "recent_rate_7d": recent_rate(df, target, WEEK_H),
    }, index=df.index)
