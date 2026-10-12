"""Second model ("v2"), kept apart from the headline pipeline (D18).

Differences from v1: extended features (each neighbour separately, 10/30-minute start counts, running-alert
counts), recency weighting of training rows, and a second target definition:
- `oblast`:  same target as v1 (any alert at any level in the oblast),
- `capital`: alert covering the oblast capital's raion (oblast-level alerts, or alerts in that raion).
  The rest of the oblast becomes an extra "neighbour" in the features.

python -m src.v2 --split val    choose recency half-life and rounds on validation only
python -m src.v2 --split test   walk-forward on the test window (a SECOND look at it, after v1)
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd

from src import baselines, features, target, train
from src.data import DATA_CUTOFF, OBLASTS, PROCESSED_DIR, ROOT, TEST_END, TEST_START, VAL_START, assign_split, merge_intervals
from src.evaluate import (METRIC_COLS, N_FOLDS, REPORTS_DIR, TARGETS, best_baseline_on_validation, check_suspicious,
                          fold_edges, join, metrics, score)

logger = logging.getLogger(__name__)

CONFIG_PATH = ROOT / "reports" / "v2_config.json"
TARGET_DEFS = ("oblast", "capital")
CAPITAL_RAION = {"Kyiv City": None, "Odeska oblast": "Odeskyi raion", "Poltavska oblast": "Poltavskyi raion"}
REST = " (rest)"
HALF_LIVES = (None, 180, 60)  # days; None = no recency weighting
ROUND_GRID = (50, 100, 200, 400)
MODEL = "lgbm_v2"


def capital_view(alerts: pd.DataFrame) -> pd.DataFrame:
    """Relabel alerts so that `oblast` means "the capital raion of that oblast".

    Rows of a scored oblast that do not cover its capital raion (other raions, hromadas in other raions) are
    moved to "<oblast> (rest)". Row-wise, so it commutes with truncating the alerts at t.
    """
    out = alerts.copy()
    for oblast, raion in CAPITAL_RAION.items():
        if raion is None:
            continue
        own = out["oblast"] == oblast
        covers = (out["level"] == "oblast") | (out["raion"] == raion)
        out.loc[own & ~covers, "oblast"] = oblast + REST
    return out


def neighbours_for(target_def: str) -> dict:
    """Slot 0 = rest of the own oblast (capital definition only), then the v1 neighbours."""
    rest = {o: (o + REST if target_def == "capital" and CAPITAL_RAION[o] else None) for o in OBLASTS}
    return {o: [rest[o]] + features.NEIGHBOURS[o] for o in OBLASTS}


def build_dataset(alerts: pd.DataFrame, target_def: str) -> tuple[pd.DataFrame, list[str]]:
    view = capital_view(alerts) if target_def == "capital" else alerts
    hourly = target.build(merge_intervals(view[view["oblast"].isin(OBLASTS)]), end=DATA_CUTOFF)
    grid = pd.DatetimeIndex(hourly["t"].drop_duplicates().sort_values())
    nb = neighbours_for(target_def)
    feats = pd.concat([features.build_features(view, o, grid, extended=True, neighbours=nb).assign(oblast=o).reset_index()
                       for o in OBLASTS], ignore_index=True)
    df, feature_cols = join(feats, hourly)
    df["split"] = assign_split(df["t"])
    return df, feature_cols


def select_on_validation(df: pd.DataFrame, feature_cols: list[str], params: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    for tgt in TARGETS:
        val = df[(df["split"] == "val") & train.prediction_rows(df, tgt) & df[tgt].notna()]
        X_val = train.make_X(val, feature_cols)
        for half_life in HALF_LIVES:
            booster = train.fit(df, feature_cols, tgt, VAL_START, "all", max(ROUND_GRID), params, half_life_days=half_life)
            for rounds in ROUND_GRID:
                m = metrics(val[tgt], booster.predict(X_val, num_iteration=rounds))
                rows.append({"target": tgt, "half_life_days": half_life or 0, "rounds": rounds, "val_log_loss": m["log_loss"],
                             "val_brier": m["brier"], "val_roc_auc": m["roc_auc"]})
    tried = pd.DataFrame(rows)
    best = tried.loc[tried.groupby("target")["val_log_loss"].idxmin()].set_index("target")
    chosen = {t: {"half_life_days": int(best.loc[t, "half_life_days"]) or None, "rounds": int(best.loc[t, "rounds"])} for t in TARGETS}
    return tried, chosen


def walk_forward(df: pd.DataFrame, feature_cols: list[str], tgt: str, start, end, params: dict, chosen: dict) -> pd.DataFrame:
    preds = baselines.all_baselines(df, tgt, start)
    preds.loc[:, "historical_rate"] = np.nan
    preds[MODEL] = np.nan
    wanted = train.prediction_rows(df, tgt)
    for a, b in fold_edges(start, end, N_FOLDS):
        fold = (df["t"] >= a) & (df["t"] < b)
        preds.loc[fold, "historical_rate"] = baselines.historical_rate(df, tgt, a)[fold]
        booster = train.fit(df, feature_cols, tgt, a, "all", chosen[tgt]["rounds"], params,
                            half_life_days=chosen[tgt]["half_life_days"])
        rows = fold & wanted
        preds.loc[rows, MODEL] = booster.predict(train.make_X(df[rows], feature_cols))
    preds[~((df["t"] >= start) & (df["t"] < end))] = np.nan
    return preds


def compact(res: pd.DataFrame, best: pd.Series, v1: pd.DataFrame | None) -> pd.DataFrame:
    r = res.set_index(["oblast", "target", "model"])
    rows = []
    for (oblast, tgt), baseline in best.items():
        m, b = r.loc[(oblast, tgt, MODEL)], r.loc[(oblast, tgt, baseline)]
        row = {"oblast": oblast, "target": tgt, "n": int(m["n"]), "base_rate": m["base_rate"]} \
            | {f"v2_{c}": m[c] for c in METRIC_COLS} | {"baseline": baseline, "bl_brier": b["brier"], "bl_roc_auc": b["roc_auc"],
                                                         "bss_v2": 1 - m["brier"] / b["brier"]}
        if v1 is not None:
            one = v1[(v1["oblast"] == oblast) & (v1["target"] == tgt)].iloc[0]
            row |= {"v1_brier": one["lgbm_brier"], "v1_roc_auc": one["lgbm_roc_auc"], "v1_pr_auc": one["lgbm_pr_auc"]}
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {t: i for i, t in enumerate(TARGETS)}
    return out.sort_values(["oblast", "target"], key=lambda s: s.map(order) if s.name == "target" else s).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    params = train.load_config()["params"]  # tree settings stay as chosen for v1 (D13)
    alerts = pd.read_parquet(PROCESSED_DIR / "alerts.parquet")

    if args.split == "val":
        config, tried_all, val_scores = {}, [], []
        for target_def in TARGET_DEFS:
            df, feature_cols = build_dataset(alerts, target_def)
            tried, chosen = select_on_validation(df, feature_cols, params)
            config[target_def] = chosen
            tried_all.append(tried.assign(target_def=target_def))
            base = pd.concat([score(df, baselines.all_baselines(df, t, VAL_START), t, "val") for t in TARGETS], ignore_index=True)
            val_scores.append(base.assign(target_def=target_def))
            rates = df.groupby("split")["alert_in_hour"].mean().round(3).to_dict()
            print(f"\n[{target_def}] {len(feature_cols)} features | share of hours with an alert by split: {rates}")
            print(tried.pivot_table(index=["target", "half_life_days"], columns="rounds", values="val_log_loss", sort=False).round(4))
            print("chosen:", chosen)
        pd.concat(tried_all).to_csv(REPORTS_DIR / "v2_selection_val.csv", index=False, float_format="%.5f")
        pd.concat(val_scores).to_csv(REPORTS_DIR / "v2_baselines_val.csv", index=False, float_format="%.5f")
        CONFIG_PATH.write_text(json.dumps(config, indent=2))
    else:
        config = json.loads(CONFIG_PATH.read_text())
        val_base = pd.read_csv(REPORTS_DIR / "v2_baselines_val.csv")
        v1 = pd.read_csv(REPORTS_DIR / "results_test_compact.csv")
        cols = ["oblast", "target", "n", "base_rate", "v2_brier", "v2_roc_auc", "v2_pr_auc", "baseline", "bl_brier", "bl_roc_auc", "bss_v2"]
        for target_def in TARGET_DEFS:
            df, feature_cols = build_dataset(alerts, target_def)
            parts = [score(df, walk_forward(df, feature_cols, t, TEST_START, TEST_END, params, config[target_def]), t, "test")
                     for t in TARGETS]
            res = pd.concat(parts, ignore_index=True)
            check_suspicious(res)
            res.to_csv(REPORTS_DIR / f"results_test_v2_{target_def}.csv", index=False, float_format="%.5f")
            best = best_baseline_on_validation(val_base[val_base["target_def"] == target_def])
            table = compact(res, best, v1 if target_def == "oblast" else None)
            table.to_csv(REPORTS_DIR / f"results_test_v2_{target_def}_compact.csv", index=False, float_format="%.5f")
            extra = ["v1_brier", "v1_roc_auc"] if target_def == "oblast" else []
            print(f"\nTEST (second look), v2, target definition: {target_def}")
            print(table[cols + extra].round(3).to_string(index=False))
