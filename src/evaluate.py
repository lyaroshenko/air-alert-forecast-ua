"""Metrics, baseline scoring on validation, and the walk-forward evaluation.

`python -m src.evaluate --split val`  scores the baselines on validation.
`python -m src.evaluate --split test` runs the walk-forward on the test window. Nothing else reads test outcomes.
"""
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src import baselines, train
from src.data import PROCESSED_DIR, ROOT, TEST_END, TEST_START, VAL_START, assign_split
from src.target import HORIZONS

logger = logging.getLogger(__name__)

REPORTS_DIR = ROOT / "reports"
TARGETS = [f"{kind}_h{h}" for h in HORIZONS for kind in ("y", "onset")]
PROB_CLIP = 1e-3  # log-loss only: hard 0/1 forecasts (persistence, seasonal naive) would otherwise be infinite
SPLIT_START = {"val": VAL_START, "test": TEST_START}
SPLIT_END = {"val": TEST_START, "test": TEST_END}
N_FOLDS = 4
METRIC_COLS = ["brier", "log_loss", "roc_auc", "pr_auc"]


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


def join(feats: pd.DataFrame, hourly: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Features joined with targets, one row per (oblast, t), sorted. Returns (frame, feature column names)."""
    feature_cols = [c for c in feats.columns if c not in ("t", "oblast")]
    df = feats.merge(hourly, on=["oblast", "t"], validate="one_to_one")
    if len(df) != len(feats) or len(df) != len(hourly):
        raise ValueError(f"join changed row count: features {len(feats)}, hourly {len(hourly)}, joined {len(df)}")
    return df.sort_values(["oblast", "t"]).reset_index(drop=True), feature_cols


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    df, feature_cols = join(pd.read_parquet(PROCESSED_DIR / "features.parquet"),
                            pd.read_parquet(PROCESSED_DIR / "hourly.parquet"))
    df["split"] = assign_split(df["t"])
    return df, feature_cols


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


def fold_edges(start: pd.Timestamp, end: pd.Timestamp, n_folds: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Equal half-open folds [a, b) that tile [start, end)."""
    edges = pd.date_range(start, end, periods=n_folds + 1)
    return list(zip(edges[:-1], edges[1:]))


def walk_forward(df: pd.DataFrame, feature_cols: list[str], target: str, start: pd.Timestamp, end: pd.Timestamp,
                 config: dict, n_folds: int = N_FOLDS, fit_end: pd.Timestamp | None = None,
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecasts for [start, end): each fold is predicted by models refitted on rows whose outcome closed
    before the fold starts (expanding window). The historical-rate baseline is refitted with the same cut-off.

    `fit_end` freezes the cut-off for every fold (used when the window comes after a data gap).
    Returns (forecasts aligned to df, importance of the headline variant per feature).
    """
    preds = baselines.all_baselines(df, target, start)
    preds.loc[:, "historical_rate"] = np.nan
    for variant in train.VARIANTS:
        preds[f"lgbm_{variant}"] = np.nan
    wanted = train.prediction_rows(df, target)
    gain, contrib, n_contrib = 0.0, 0.0, 0
    for a, b in fold_edges(start, end, n_folds):
        cut = fit_end if fit_end is not None else a
        if cut > a:
            raise ValueError("fit cut-off after the start of the fold")
        fold = (df["t"] >= a) & (df["t"] < b)
        preds.loc[fold, "historical_rate"] = baselines.historical_rate(df, target, cut)[fold]
        rows = fold & wanted
        X = train.make_X(df[rows], feature_cols)
        for variant in train.VARIANTS:
            booster = train.fit(df, feature_cols, target, cut, variant, config["rounds"][target][variant], config["params"])
            preds.loc[rows, f"lgbm_{variant}"] = booster.predict(X)
            if variant == config["headline"][target]:
                g = booster.feature_importance("gain")
                gain = gain + g / g.sum() / n_folds
                contrib = contrib + np.abs(booster.predict(X, pred_contrib=True)[:, :-1]).sum(axis=0)
                n_contrib += len(X)
    in_window = (df["t"] >= start) & (df["t"] < end)
    preds[~in_window] = np.nan
    importance = pd.DataFrame({"target": target, "feature": list(X.columns), "gain_share": gain,
                               "mean_abs_shap": contrib / max(n_contrib, 1)})
    return preds, importance


def best_baseline_on_validation(val_scores: pd.DataFrame) -> pd.Series:
    """Baseline with the lowest validation Brier per (oblast, target)."""
    base = val_scores[~val_scores["model"].str.startswith("lgbm_")]
    return base.loc[base.groupby(["oblast", "target"])["brier"].idxmin()].set_index(["oblast", "target"])["model"]


def compact_table(res: pd.DataFrame, config: dict, best_baseline: pd.Series) -> pd.DataFrame:
    """Per oblast x target: headline LightGBM, the validation-chosen baseline, and the Brier skill score."""
    rows = []
    r = res.set_index(["oblast", "target", "model"])
    for (oblast, target), baseline in best_baseline.items():
        variant = config["headline"][target]
        m, b = r.loc[(oblast, target, f"lgbm_{variant}")], r.loc[(oblast, target, baseline)]
        rows.append({"oblast": oblast, "target": target, "n": int(m["n"]), "base_rate": m["base_rate"], "lgbm": variant}
                    | {f"lgbm_{c}": m[c] for c in METRIC_COLS} | {"baseline": baseline}
                    | {f"bl_{c}": b[c] for c in METRIC_COLS} | {"bss": 1 - m["brier"] / b["brier"]})
    out = pd.DataFrame(rows)
    order = {t: i for i, t in enumerate(TARGETS)}
    return out.sort_values(["oblast", "target"], key=lambda s: s.map(order) if s.name == "target" else s).reset_index(drop=True)


def reliability_plot(df: pd.DataFrame, preds: pd.DataFrame, target: str, split: str, models: dict, path, n_bins: int = 10):
    """One panel per oblast; `models` maps oblast -> {label: forecast column}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    oblasts = sorted(models)
    fig, axes = plt.subplots(1, len(oblasts), figsize=(4.6 * len(oblasts), 4.4), squeeze=False)
    bins = np.linspace(0, 1, n_bins + 1)
    for ax, oblast in zip(axes[0], oblasts):
        mask = (df["split"] == split) & (df["oblast"] == oblast) & df[target].notna()
        y = df.loc[mask, target].to_numpy()
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect")
        for label, col in models[oblast].items():
            p = preds.loc[mask, col].to_numpy()
            which = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
            xs = [p[which == i].mean() for i in range(n_bins) if (which == i).sum() >= 5]
            ys = [y[which == i].mean() for i in range(n_bins) if (which == i).sum() >= 5]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.5, label=label)
        ax.set_title(f"{oblast} (n={int(mask.sum())}, base rate {y.mean():.2f})", fontsize=10)
        ax.set_xlabel("forecast probability")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
    axes[0][0].set_ylabel("observed frequency")
    fig.suptitle(f"Reliability, {target}, {split} window (bins with < 5 forecasts hidden)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def importance_plot(importance: pd.DataFrame, target: str, path, top: int = 15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imp = importance[importance["target"] == target]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col, title in zip(axes, ["mean_abs_shap", "gain_share"],
                              ["mean |SHAP| on test folds (log-odds)", "share of total split gain"]):
        top_rows = imp.nlargest(top, col).iloc[::-1]
        ax.barh(top_rows["feature"], top_rows[col], color="#3b6ea5")
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Top {top} features, {target}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_window(df: pd.DataFrame, feature_cols: list[str], split: str, start: pd.Timestamp, end: pd.Timestamp,
               config: dict, n_folds: int = N_FOLDS, fit_end: pd.Timestamp | None = None, targets=None):
    """Walk-forward for every target. Returns (long metrics table, {target: forecasts}, importance table)."""
    scores, all_preds, importances = [], {}, []
    for target in targets or TARGETS:
        preds, importance = walk_forward(df, feature_cols, target, start, end, config, n_folds, fit_end)
        scores.append(score(df, preds, target, split))
        all_preds[target] = preds
        importances.append(importance)
        logger.info("walk-forward done: %s", target)
    return pd.concat(scores, ignore_index=True), all_preds, pd.concat(importances, ignore_index=True)


def check_suspicious(res: pd.DataFrame) -> None:
    sus = res[res["target"].str.startswith("onset_") & (res["roc_auc"] > 0.95)]
    if len(sus):
        logger.warning("SUSPICIOUS: onset ROC-AUC > 0.95, check for leakage before reporting:\n%s", sus.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)

    df, feature_cols = load_dataset()
    print(f"dataset: {len(df)} rows | split sizes: {df['split'].value_counts().to_dict()}")
    print(f"scoring split '{args.split}': {SPLIT_START[args.split]} .. {SPLIT_END[args.split]}")
    REPORTS_DIR.mkdir(exist_ok=True)

    if args.split == "val":
        res = evaluate_baselines(df, "val")
        path = REPORTS_DIR / "baselines_val.csv"
        res.to_csv(path, index=False, float_format="%.5f")
        print(f"wrote {path.relative_to(ROOT)}\n")
        for target in ("y_h1", "onset_h1"):
            print(f"--- {target} ---")
            print(res[res["target"] == target].drop(columns="target").round(4).to_string(index=False))
        print("\n--- Brier score, all targets (rows: oblast x model) ---")
        print(res.pivot_table(index=["oblast", "model"], columns="target", values="brier", sort=False)[TARGETS].round(4))
    else:
        config = train.load_config()
        best = best_baseline_on_validation(pd.read_csv(REPORTS_DIR / "baselines_val.csv"))
        print(f"model config (chosen on validation): {config['param_set']} | headline: {config['headline']}")
        print("folds:", [(f"{a:%m-%d}", f"{b:%m-%d}") for a, b in fold_edges(TEST_START, TEST_END, N_FOLDS)])
        res, preds, importance = run_window(df, feature_cols, "test", TEST_START, TEST_END, config)
        check_suspicious(res)
        res.to_csv(REPORTS_DIR / "results_test.csv", index=False, float_format="%.5f")
        importance.to_csv(REPORTS_DIR / "importance_test.csv", index=False, float_format="%.6f")
        table = compact_table(res, config, best)
        table.to_csv(REPORTS_DIR / "results_test_compact.csv", index=False, float_format="%.5f")
        print("\nTEST, walk-forward: headline LightGBM vs the baseline that was best on VALIDATION")
        print(table.round(3).to_string(index=False))
        print("\nTEST Brier, both training windows and every baseline:")
        print(res.pivot_table(index=["oblast", "model"], columns="target", values="brier", sort=False)[TARGETS].round(4))
        for target in ("y_h1", "onset_h1"):
            variant = config["headline"][target]
            models = {o: {f"LightGBM ({variant})": f"lgbm_{variant}", best[(o, target)]: best[(o, target)]}
                      for o in sorted(df["oblast"].unique())}
            reliability_plot(df, preds[target], target, "test", models, REPORTS_DIR / f"calibration_{target}.png")
            importance_plot(importance, target, REPORTS_DIR / f"importance_{target}.png")
            top = importance[importance["target"] == target].nlargest(8, "mean_abs_shap")
            print(f"\ntop features by mean |SHAP|, {target}:")
            print(top[["feature", "mean_abs_shap", "gain_share"]].round(4).to_string(index=False))
        print("\nwrote reports/results_test.csv, results_test_compact.csv, importance_test.csv, calibration_*.png, importance_*.png")
