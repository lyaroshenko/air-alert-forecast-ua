"""Metrics and evaluation tables. The test split is only scored when asked for explicitly."""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src import baselines
from src.data import PROCESSED_DIR, ROOT, TEST_END, TEST_START, VAL_START, assign_split
from src.target import HORIZONS

logger = logging.getLogger(__name__)

REPORTS_DIR = ROOT / "reports"
TARGETS = [f"{kind}_h{h}" for h in HORIZONS for kind in ("y", "onset")]
PROB_CLIP = 1e-3  # log-loss only: hard 0/1 forecasts (persistence, seasonal naive) would otherwise be infinite
SPLIT_START = {"val": VAL_START, "test": TEST_START}
SPLIT_END = {"val": TEST_START, "test": TEST_END}


def metrics(y_true, p) -> dict:
    """Brier, log-loss, ROC-AUC, PR-AUC on rows where both the outcome and the forecast exist."""
    y_true, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    ok = ~np.isnan(y_true) & ~np.isnan(p)
    y, p = y_true[ok].astype(int), p[ok]
    out = {"n": int(ok.sum()), "base_rate": float(y.mean()) if len(y) else np.nan}
    if len(y) == 0:
        return out | {"brier": np.nan, "log_loss": np.nan, "roc_auc": np.nan, "pr_auc": np.nan}
    out["brier"] = brier_score_loss(y, p)
    out["log_loss"] = log_loss(y, np.clip(p, PROB_CLIP, 1 - PROB_CLIP), labels=[0, 1])
    both_classes = 0 < y.sum() < len(y)
    constant = np.ptp(p) == 0
    out["roc_auc"] = roc_auc_score(y, p) if both_classes and not constant else np.nan
    out["pr_auc"] = average_precision_score(y, p) if both_classes else np.nan
    return out


def load_dataset() -> pd.DataFrame:
    """Features joined with targets, one row per (oblast, t), sorted."""
    feats = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    hourly = pd.read_parquet(PROCESSED_DIR / "hourly.parquet")
    df = feats.merge(hourly, on=["oblast", "t"], validate="one_to_one")
    if len(df) != len(feats) or len(df) != len(hourly):
        raise ValueError(f"join changed row count: features {len(feats)}, hourly {len(hourly)}, joined {len(df)}")
    df = df.sort_values(["oblast", "t"]).reset_index(drop=True)
    df["split"] = assign_split(df["t"])
    return df


def score(df: pd.DataFrame, preds: pd.DataFrame, target: str, split: str) -> pd.DataFrame:
    """One row of metrics per (oblast, model) for `target` on `split`."""
    rows = []
    mask = df["split"] == split
    missing = preds[mask & df[target].notna()].isna().sum()
    if missing.any():
        raise ValueError(f"{target}: forecasts missing on scored rows: {missing[missing > 0].to_dict()}")
    for oblast, idx in df[mask].groupby("oblast").groups.items():
        for model in preds.columns:
            rows.append({"oblast": oblast, "target": target, "model": model}
                        | metrics(df.loc[idx, target], preds.loc[idx, model]))
    return pd.DataFrame(rows)


def evaluate_baselines(df: pd.DataFrame, split: str) -> pd.DataFrame:
    fit_end = SPLIT_START[split]  # historical rate sees nothing from the scored split
    parts = [score(df, baselines.all_baselines(df, target, fit_end), target, split) for target in TARGETS]
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 20)

    df = load_dataset()
    print(f"dataset: {len(df)} rows | split sizes: {df['split'].value_counts().to_dict()}")
    print(f"scoring split '{args.split}': {SPLIT_START[args.split]} .. {SPLIT_END[args.split]}")
    res = evaluate_baselines(df, args.split)
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"baselines_{args.split}.csv"
    res.to_csv(path, index=False, float_format="%.5f")
    print(f"wrote {path.relative_to(ROOT)}\n")
    for target in ("y_h1", "onset_h1"):
        print(f"--- {target} ---")
        print(res[res["target"] == target].drop(columns="target").round(4).to_string(index=False))
    print("\n--- Brier score, all targets (rows: oblast x model) ---")
    print(res.pivot_table(index=["oblast", "model"], columns="target", values="brier", sort=False)[TARGETS].round(4))
