import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

LABEL_CONFIGS = [
    ("relevance",    "relevance"),
    ("booking_bool", "booking_bool"),
    ("click_bool",   "click_bool"),
]
SEEDS = [42, 123, 2024]   # same seeds as LightGBM / neural ensembles

DROP_COLS = {
    "srch_id", "date_time", "prop_id",
    "click_bool", "booking_bool", "gross_bookings_usd",
    "position", "random_bool",
    "relevance", "score",
}

# Auto-detect GPU — falls back to CPU hist if unavailable
try:
    xgb.train({"tree_method": "hist", "device": "cuda"},
              xgb.DMatrix(np.zeros((2, 2)), label=[0, 1]),
              num_boost_round=1, verbose_eval=False)
    DEVICE = "cuda"
except Exception:
    DEVICE = "cpu"
print(f"XGBoost device: {DEVICE}")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def make_group_array(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("srch_id", sort=False).size().to_numpy()


def to_dmatrix(df: pd.DataFrame, features: list[str], label_col: str | None):
    dm = xgb.DMatrix(df[features], label=df[label_col] if label_col else None)
    dm.set_group(make_group_array(df))
    return dm


def rank_normalize(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("srch_id")[col].rank(pct=True)


def train_xgb_ranker(
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    label_col: str,
    features: list[str],
    seed: int,
) -> xgb.Booster:
    train_fold = train_fold.sort_values("srch_id").reset_index(drop=True)
    val_fold   = val_fold.sort_values("srch_id").reset_index(drop=True)

    dtrain = to_dmatrix(train_fold, features, label_col)
    dval   = to_dmatrix(val_fold,   features, label_col)

    params = {
        "objective":        "rank:ndcg",
        "eval_metric":      ["ndcg@10", "ndcg@5"],
        "tree_method":      "hist",
        "device":           DEVICE,
        "learning_rate":    0.05,
        "max_depth":        8,
        "min_child_weight": 10,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,  # extra stochasticity per depth level
        "lambda":           1.0,
        "alpha":            0.1,   # matched from LightGBM working config
        "seed":             seed,
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        callbacks=[xgb.callback.EvaluationMonitor(period=50)],
    )

    print(f"    Best iteration: {booster.best_iteration}  "
          f"val NDCG@5: {booster.best_score:.5f}")
    return booster


def ndcg_at_k(group: pd.DataFrame, k: int = 5) -> float:
    ranked    = group.sort_values("pred_score", ascending=False).head(k)
    gains     = 2 ** ranked["relevance"].to_numpy() - 1
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg       = (gains * discounts).sum()

    ideal  = group.sort_values("relevance", ascending=False).head(k)
    igains = 2 ** ideal["relevance"].to_numpy() - 1
    idcg   = (igains / np.log2(np.arange(2, len(igains) + 2))).sum()
    return dcg / idcg if idcg > 0 else 0.0


def plot_importance(booster, label_col: str, seed: int, top: int = 25):
    score = booster.get_score(importance_type="gain")
    imp   = pd.Series(score).sort_values(ascending=True).tail(top)

    _, ax = plt.subplots(figsize=(8, 0.3 * len(imp) + 1))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Gain")
    ax.set_title(f"XGBoost importance — {label_col} seed{seed} (top {top}, gain)")
    plt.tight_layout()
    path = OUT / f"xgb_importance_{label_col}_seed{seed}.png"
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    train = pd.read_parquet("prepared_train.parquet")
    val   = pd.read_parquet("prepared_val.parquet")
    test  = pd.read_parquet("prepared_test.parquet")

    train["relevance"] = train["relevance"].clip(upper=5)
    val["relevance"]   = val["relevance"].clip(upper=5)

    train = train.sort_values("srch_id").reset_index(drop=True)
    val   = val.sort_values("srch_id").reset_index(drop=True)
    test  = test.sort_values("srch_id").reset_index(drop=True)

    features = feature_columns(train)
    n_models = len(LABEL_CONFIGS) * len(SEEDS)
    print(f"Features : {len(features)}")
    print(f"Models   : {len(LABEL_CONFIGS)} labels × {len(SEEDS)} seeds = {n_models}")

    val_acc  = np.zeros(len(val))
    test_acc = np.zeros(len(test))

    for label_name, label_col in LABEL_CONFIGS:
        for seed in SEEDS:
            val_cache  = f"xgb_{label_name}_seed{seed}_val.parquet"
            test_cache = f"xgb_{label_name}_seed{seed}_test.parquet"

            if os.path.exists(val_cache) and os.path.exists(test_cache):
                print(f"\n[{label_name} seed={seed}] loading from cache")
                val_scores_df  = pd.read_parquet(val_cache)
                test_scores_df = pd.read_parquet(test_cache)
            else:
                print(f"\n[{label_name} seed={seed}] training ...")
                booster = train_xgb_ranker(train, val, label_col, features, seed)
                plot_importance(booster, label_name, seed)

                irange = (0, booster.best_iteration + 1)
                dval   = to_dmatrix(val,  features, label_col)
                dtest  = to_dmatrix(test, features, label_col=None)

                val_scores_df  = val[["srch_id",  "prop_id"]].copy()
                test_scores_df = test[["srch_id", "prop_id"]].copy()
                val_scores_df["rank_score"]  = booster.predict(dval,  iteration_range=irange)
                test_scores_df["rank_score"] = booster.predict(dtest, iteration_range=irange)

                val_scores_df.to_parquet(val_cache,   index=False)
                test_scores_df.to_parquet(test_cache, index=False)
                print(f"    Cached {val_cache} and {test_cache}")

            val_acc  += rank_normalize(val_scores_df,  "rank_score").values
            test_acc += rank_normalize(test_scores_df, "rank_score").values

    val["pred_score"]  = val_acc  / n_models
    test["pred_score"] = test_acc / n_models

    per_query = val.groupby("srch_id").apply(ndcg_at_k, include_groups=False)
    print(f"\nEnsemble val NDCG@5 ({len(per_query):,} queries): {per_query.mean():.5f}")

    val[["srch_id",  "prop_id", "pred_score"]].to_csv("xgboost_val_scores.csv",  index=False)
    test[["srch_id", "prop_id", "pred_score"]].to_csv("xgboost_test_scores.csv", index=False)
    print("Exported xgboost_val_scores.csv and xgboost_test_scores.csv")
