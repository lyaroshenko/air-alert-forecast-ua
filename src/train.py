"""LightGBM models: one binary model per target, pooled over oblasts.

`python -m src.train` runs model selection on the VALIDATION split only and writes the chosen
configuration to reports/model_config.json. It never reads test-window outcomes.
"""
import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.baselines import horizon_of
from src.data import OBLASTS, ROOT

logger = logging.getLogger(__name__)

SEED = 20261012
CONFIG_PATH = ROOT / "reports" / "model_config.json"
RECENT_START = pd.Timestamp("2025-11-01", tz="UTC")  # month the raion reform reached almost all oblasts (D6)
VARIANTS = ("all", "recent")
FIXED_PARAMS = dict(objective="binary", metric="binary_logloss", seed=SEED, deterministic=True,
                    force_row_wise=True, num_threads=4, verbose=-1)
PARAM_SETS = {
    "base": dict(num_leaves=31, learning_rate=0.05, min_child_samples=100, feature_fraction=0.8),
    "small": dict(num_leaves=15, learning_rate=0.05, min_child_samples=200, feature_fraction=0.8),
    "large": dict(num_leaves=63, learning_rate=0.05, min_child_samples=50, feature_fraction=0.8),
}
ROUND_GRID = (25, 50, 100, 200, 400)
FORBIDDEN_COLUMNS = {"t", "split", "alert_in_hour", "active_at"}


def make_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Model inputs: the feature columns plus `oblast` as a categorical. Refuses anything target-like."""
    X = df[feature_cols].copy()
    X["oblast"] = pd.Categorical(df["oblast"], categories=sorted(OBLASTS))
    if X["oblast"].isna().any():
        raise ValueError("unknown oblast in model input")
    leaked = [c for c in X.columns if c.startswith(("y_", "onset_")) or c in FORBIDDEN_COLUMNS]
    assert not leaked, f"target or bookkeeping columns in X: {leaked}"
    return X


def prediction_rows(df: pd.DataFrame, target: str) -> pd.Series:
    """Rows a model of `target` is asked about: onset models only see hours with no running alert."""
    if target.startswith("onset_"):
        return df["active"] == 0
    return pd.Series(True, index=df.index)


def training_mask(df: pd.DataFrame, target: str, fit_end: pd.Timestamp, variant: str) -> pd.Series:
    """Rows allowed in a fit that may be used from `fit_end` on.

    Purge rule: a row at time s is allowed only if s + H <= fit_end, i.e. its outcome window [s, s+H)
    had closed. `recent` additionally drops everything before RECENT_START.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    closed = df["t"] + pd.Timedelta(hours=horizon_of(target)) <= fit_end
    mask = closed & prediction_rows(df, target) & df[target].notna()
    if variant == "recent":
        mask &= df["t"] >= RECENT_START
    return mask


def fit(df: pd.DataFrame, feature_cols: list[str], target: str, fit_end: pd.Timestamp, variant: str,
        num_boost_round: int, params: dict, valid: pd.DataFrame | None = None, record: dict | None = None):
    mask = training_mask(df, target, fit_end, variant)
    train_set = lgb.Dataset(make_X(df[mask], feature_cols), df.loc[mask, target].astype(int), categorical_feature=["oblast"])
    kwargs = {}
    if valid is not None:
        valid_set = lgb.Dataset(make_X(valid, feature_cols), valid[target].astype(int), reference=train_set)
        kwargs = dict(valid_sets=[valid_set], valid_names=["val"], callbacks=[lgb.record_evaluation(record)])
    return lgb.train(FIXED_PARAMS | params, train_set, num_boost_round=num_boost_round, **kwargs)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("reports/model_config.json missing: run `python -m src.train` first")
    return json.loads(CONFIG_PATH.read_text())


def select_on_validation(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Try every (param set, target, variant, rounds) on validation; pick by pooled validation log-loss."""
    from src.data import VAL_START
    from src.evaluate import TARGETS, metrics

    rows = []
    for name, params in PARAM_SETS.items():
        for target in TARGETS:
            val = df[(df["split"] == "val") & prediction_rows(df, target) & df[target].notna()]
            X_val = make_X(val, feature_cols)
            for variant in VARIANTS:
                booster = fit(df, feature_cols, target, VAL_START, variant, max(ROUND_GRID), params)
                n_train = int(training_mask(df, target, VAL_START, variant).sum())
                for rounds in ROUND_GRID:
                    m = metrics(val[target], booster.predict(X_val, num_iteration=rounds))
                    rows.append({"param_set": name, "target": target, "variant": variant, "rounds": rounds,
                                 "n_train": n_train, "n_val": m["n"], "val_log_loss": m["log_loss"],
                                 "val_brier": m["brier"], "val_roc_auc": m["roc_auc"], "val_pr_auc": m["pr_auc"]})
        logger.info("validated param set %s", name)
    tried = pd.DataFrame(rows)

    best_rounds = tried.loc[tried.groupby(["param_set", "target", "variant"])["val_log_loss"].idxmin()]
    per_set = best_rounds.groupby("param_set")["val_log_loss"].mean()
    chosen_set = per_set.idxmin()
    chosen = best_rounds[best_rounds["param_set"] == chosen_set].set_index(["target", "variant"])
    config = {"param_set": chosen_set, "params": PARAM_SETS[chosen_set], "rounds": {}, "headline": {},
              "param_set_mean_val_log_loss": per_set.round(5).to_dict()}
    for target in TARGETS:
        config["rounds"][target] = {v: int(chosen.loc[(target, v), "rounds"]) for v in VARIANTS}
        config["headline"][target] = min(VARIANTS, key=lambda v: chosen.loc[(target, v), "val_log_loss"])
    return tried, config


if __name__ == "__main__":
    from src.evaluate import REPORTS_DIR, TARGETS, load_dataset, score

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    df, feature_cols = load_dataset()
    print(f"{len(feature_cols)} features + oblast: {feature_cols}")
    tried, config = select_on_validation(df, feature_cols)
    REPORTS_DIR.mkdir(exist_ok=True)
    tried.to_csv(REPORTS_DIR / "model_selection_val.csv", index=False, float_format="%.5f")
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    print("\nmean over 6 targets x 2 variants of the best validation log-loss, per param set:")
    print(pd.Series(config["param_set_mean_val_log_loss"]).to_string())
    print(f"chosen param set: {config['param_set']} {config['params']}")
    print("\nvalidation log-loss by rounds for the chosen param set (pooled over oblasts):")
    sel = tried[tried["param_set"] == config["param_set"]]
    print(sel.pivot_table(index=["target", "variant"], columns="rounds", values="val_log_loss", sort=False).round(4))
    print("\nchosen rounds and headline variant per target:")
    summary = []
    for target in TARGETS:
        for variant in VARIANTS:
            r = sel[(sel["target"] == target) & (sel["variant"] == variant) & (sel["rounds"] == config["rounds"][target][variant])].iloc[0]
            summary.append({"target": target, "variant": variant, "rounds": int(r["rounds"]), "n_train": int(r["n_train"]),
                            "val_log_loss": r["val_log_loss"], "val_brier": r["val_brier"], "val_roc_auc": r["val_roc_auc"],
                            "val_pr_auc": r["val_pr_auc"], "headline": config["headline"][target] == variant})
    print(pd.DataFrame(summary).round(4).to_string(index=False))

    from src.data import VAL_START
    parts = []
    for target in TARGETS:
        preds = pd.DataFrame(index=df.index)
        rows = (df["split"] == "val") & prediction_rows(df, target)
        for variant in VARIANTS:
            booster = fit(df, feature_cols, target, VAL_START, variant, config["rounds"][target][variant], config["params"])
            preds.loc[rows, f"lgbm_{variant}"] = booster.predict(make_X(df[rows], feature_cols))
        parts.append(score(df, preds, target, "val"))
    val = pd.concat(parts, ignore_index=True)
    val.to_csv(REPORTS_DIR / "lgbm_val.csv", index=False, float_format="%.5f")
    base = pd.read_csv(REPORTS_DIR / "baselines_val.csv")
    best = base.loc[base.groupby(["oblast", "target"])["brier"].idxmin()].set_index(["oblast", "target"])
    val["headline"] = [config["headline"][t] == m.removeprefix("lgbm_") for t, m in zip(val["target"], val["model"])]
    head = val[val["headline"]].set_index(["oblast", "target"])
    cmp = pd.DataFrame({"lgbm": head["model"], "brier": head["brier"], "roc_auc": head["roc_auc"], "pr_auc": head["pr_auc"],
                        "best_baseline": best["model"], "bl_brier": best["brier"], "bl_roc_auc": best["roc_auc"]})
    cmp["bss"] = 1 - cmp["brier"] / cmp["bl_brier"]
    print("\nVALIDATION, per oblast: headline LightGBM vs best baseline (by validation Brier):")
    print(cmp.round(4).to_string())
