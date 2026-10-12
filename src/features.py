"""Leakage-safe features for a forecast made at hour start t.

Everything is computed from raw alerts through two closed-form functions of time, so no rolling windows
and no hourly resampling are involved:
- F(x): seconds of alert before instant x (the part of each alert after x is clipped away),
- N(x): number of alert starts strictly before x.

A feature at t only evaluates F and N at instants <= t, plus "is an alert running at t", which needs the
start (< t) and whether the end is >= t, never the end itself. An alert whose end is unknown at t is
represented by finished_at = NaT and treated as open-ended; `truncate_alerts` produces exactly that view,
and tests/test_leakage.py checks that features rebuilt from it match the full pipeline.
"""
import logging

import numpy as np
import pandas as pd

from src.data import OBLASTS, PROCESSED_DIR

logger = logging.getLogger(__name__)

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
HOUR_S = 3600.0
LOCAL_TZ = "Europe/Kyiv"

NEIGHBOURS = {
    "Kyiv City": ["Kyivska oblast", "Chernihivska oblast", "Zhytomyrska oblast", "Cherkaska oblast"],
    "Odeska oblast": ["Mykolaivska oblast", "Vinnytska oblast", "Kirovohradska oblast", "Khersonska oblast"],
    "Poltavska oblast": ["Kharkivska oblast", "Sumska oblast", "Dnipropetrovska oblast", "Kyivska oblast",
                         "Cherkaska oblast", "Chernihivska oblast", "Kirovohradska oblast"],
}
FRAC_WINDOWS_H = (1, 3, 6, 24, 168, 720)
START_WINDOWS_H = (3, 24, 168)


def _seconds(ts) -> np.ndarray:
    """Seconds since epoch as float; NaT -> NaN."""
    if len(ts) == 0:
        return np.zeros(0)
    return np.asarray((pd.Series(ts) - EPOCH) / pd.Timedelta(seconds=1), dtype=float)


def truncate_alerts(alerts: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """The alerts as they are known at t: nothing that starts at or after t, and no end time >= t."""
    known = alerts[alerts["started_at"] < t].copy()
    known.loc[known["finished_at"] >= t, "finished_at"] = pd.NaT
    return known


class AlertState:
    """Merged alert intervals of one area (touching/overlapping alerts unioned), with F, N and state lookups."""

    def __init__(self, started_at, finished_at):
        s, e = _seconds(started_at), _seconds(finished_at)
        e = np.where(np.isnan(e), np.inf, e)  # end unknown -> still running
        order = np.argsort(s, kind="stable")
        s, e = s[order], e[order]
        self.raw_s, self.raw_e = s, np.sort(e)
        if len(s):
            run_end = np.maximum.accumulate(e)
            first = np.concatenate([[True], s[1:] > run_end[:-1]])
            last = np.concatenate([first[1:], [True]])
            s, e = s[first], run_end[last]
        self.s, self.e = s, e
        self.prefix = np.concatenate([[0.0], np.cumsum((e - s)[:-1])]) if len(s) else np.zeros(0)

    def _last_started_before(self, x):
        return np.searchsorted(self.s, x, side="left") - 1

    def alert_seconds_before(self, x: np.ndarray) -> np.ndarray:
        """F(x)."""
        if not len(self.s):
            return np.zeros_like(x)
        k = self._last_started_before(x)
        kk = np.maximum(k, 0)
        return np.where(k >= 0, self.prefix[kk] + np.minimum(self.e[kk], x) - self.s[kk], 0.0)

    def starts_before(self, x: np.ndarray) -> np.ndarray:
        """N(x)."""
        return np.searchsorted(self.s, x, side="left").astype(float)

    def frac(self, x_from: np.ndarray, x_to: np.ndarray) -> np.ndarray:
        """Share of [x_from, x_to) spent under alert."""
        return (self.alert_seconds_before(x_to) - self.alert_seconds_before(x_from)) / (x_to - x_from)

    def active(self, x: np.ndarray) -> np.ndarray:
        """An alert started before x and had not ended before x."""
        if not len(self.s):
            return np.zeros(len(x), dtype=bool)
        k = self._last_started_before(x)
        return (k >= 0) & (self.e[np.maximum(k, 0)] >= x)

    def n_running(self, x: np.ndarray) -> np.ndarray:
        """Number of raw (unmerged) alerts that started before x and had not ended before x."""
        return (np.searchsorted(self.raw_s, x, side="left") - np.searchsorted(self.raw_e, x, side="left")).astype(float)

    def current_alert_age(self, x: np.ndarray) -> np.ndarray:
        """Seconds since the running alert started; 0 when no alert is running."""
        if not len(self.s):
            return np.zeros_like(x)
        k = np.maximum(self._last_started_before(x), 0)
        return np.where(self.active(x), x - self.s[k], 0.0)

    def last_finished(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(seconds since the last alert that ended before x, its duration); NaN if there is none."""
        if not len(self.s):
            return np.full(len(x), np.nan), np.full(len(x), np.nan)
        j = np.searchsorted(self.e, x, side="left") - 1
        jj = np.maximum(j, 0)
        return np.where(j >= 0, x - self.e[jj], np.nan), np.where(j >= 0, self.e[jj] - self.s[jj], np.nan)


MAX_NEIGHBOUR_SLOTS = 8  # slot 0 is reserved for "rest of the own oblast" (v2 capital-raion target)
SHORT_WINDOWS_S = {"10m": 600.0, "30m": 1800.0}


def build_features(alerts: pd.DataFrame, oblast: str, grid: pd.DatetimeIndex, extended: bool = False,
                   neighbours: dict | None = None) -> pd.DataFrame:
    """Features for `oblast` at every forecast time in `grid`, from alerts of all oblasts.

    extended=True adds the v2 columns: each neighbour separately, 10/30-minute start counts, running-alert counts.
    """
    neighbours = NEIGHBOURS if neighbours is None else neighbours
    t = _seconds(grid)
    states = {name: AlertState(g["started_at"], g["finished_at"]) for name, g in alerts.groupby("oblast")}
    empty = AlertState([], [])
    own = states.get(oblast, empty)
    out = pd.DataFrame(index=grid)
    out.index.name = "t"

    out["active"] = own.active(t).astype(float)
    out["cur_alert_age_min"] = own.current_alert_age(t) / 60
    since_end, last_dur = own.last_finished(t)
    out["mins_since_last_end"] = np.where(out["active"] == 1, 0.0, since_end / 60)
    out["last_alert_dur_min"] = last_dur / 60
    for w in FRAC_WINDOWS_H:
        out[f"frac_{w}h"] = own.frac(t - w * HOUR_S, t)
    for w in START_WINDOWS_H:
        out[f"starts_{w}h"] = own.starts_before(t) - own.starts_before(t - w * HOUR_S)
    out["frac_hour_lag24h"] = own.frac(t - 24 * HOUR_S, t - 23 * HOUR_S)
    out["frac_hour_lag168h"] = own.frac(t - 168 * HOUR_S, t - 167 * HOUR_S)

    local = grid.tz_convert(LOCAL_TZ)
    out["hour_utc"] = grid.hour.astype(float)
    out["hour_local"] = local.hour.astype(float)
    out["dow_local"] = local.dayofweek.astype(float)

    others = sorted(set(states) - {oblast})
    for prefix, names in (("nb", [n for n in neighbours.get(oblast, []) if n]), ("nat", others)):
        group = [states.get(name, empty) for name in names]
        if not group:
            continue
        active = np.array([st.active(t) for st in group], dtype=float)
        frac_1h = np.array([st.frac(t - HOUR_S, t) for st in group])
        out[f"{prefix}_active_share"] = active.mean(axis=0)
        out[f"{prefix}_frac_1h_mean"] = frac_1h.mean(axis=0)
        for w in (1, 3):
            out[f"{prefix}_starts_{w}h"] = sum(st.starts_before(t) - st.starts_before(t - w * HOUR_S) for st in group)
        if prefix == "nb":
            out["nb_frac_1h_max"] = frac_1h.max(axis=0)
            out["nb_frac_6h_mean"] = np.mean([st.frac(t - 6 * HOUR_S, t) for st in group], axis=0)
        if extended:
            for label, w in SHORT_WINDOWS_S.items():
                out[f"{prefix}_starts_{label}"] = sum(st.starts_before(t) - st.starts_before(t - w) for st in group)
            out[f"{prefix}_n_running"] = sum(st.n_running(t) for st in group)
    if extended:
        out["own_n_running"] = own.n_running(t)
        slots = list(neighbours.get(oblast, []))[:MAX_NEIGHBOUR_SLOTS]
        slots += [None] * (MAX_NEIGHBOUR_SLOTS - len(slots))
        for i, name in enumerate(slots):
            st = states.get(name, empty) if name else None
            out[f"nb{i}_active"] = st.active(t).astype(float) if st else np.nan
            out[f"nb{i}_age_min"] = st.current_alert_age(t) / 60 if st else np.nan
            out[f"nb{i}_frac_1h"] = st.frac(t - HOUR_S, t) if st else np.nan
            out[f"nb{i}_n_running"] = st.n_running(t) if st else np.nan
    return out


def build(alerts: pd.DataFrame, grid: pd.DatetimeIndex, oblasts: list[str] = OBLASTS) -> pd.DataFrame:
    parts = [build_features(alerts, oblast, grid).assign(oblast=oblast).reset_index() for oblast in oblasts]
    out = pd.concat(parts, ignore_index=True)
    logger.info("features: %d rows x %d feature columns", len(out), out.shape[1] - 2)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    alerts = pd.read_parquet(PROCESSED_DIR / "alerts.parquet")
    hourly = pd.read_parquet(PROCESSED_DIR / "hourly.parquet")
    grid = pd.DatetimeIndex(hourly["t"].drop_duplicates().sort_values())
    feats = build(alerts, grid)
    feats.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    print(feats.drop(columns=["t"]).groupby("oblast").mean().T.round(3))
    print("\nNaN counts:", feats.isna().sum().loc[lambda s: s > 0].to_dict())
    merged = feats.merge(hourly, on=["oblast", "t"])
    print("\nfeature 'active' equals target-side 'active_at' on every row:", bool((merged["active"] == merged["active_at"]).all()))
