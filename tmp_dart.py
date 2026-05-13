import os
import numpy as np
import pandas as pd
import lightgbm as lgb

LABEL_CONFIGS = [
    ("relevance",    "relevance"),
    ("booking_bool", "booking_bool"),
    ("click_bool",   "click_bool"),
]
SEEDS = [42, 123, 2024]

DROP_COLS = {"srch_id", "date_time", "relevance", "score",
             "click_bool", "booking_bool", "gross_bookings_usd",
             "position", "random_bool"}


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def rank_normalize(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("srch_id")[col].rank(pct=True)


train = pd.read_parquet("prepared_train.parquet").sort_values("srch_id").reset_index(drop=True)
val   = pd.read_parquet("prepared_val.parquet").sort_values("srch_id").reset_index(drop=True)
test  = pd.read_parquet("prepared_test.parquet").sort_values("srch_id").reset_index(drop=True)

train["relevance"] = train["relevance"].clip(upper=5)
val["relevance"]   = val["relevance"].clip(upper=5)

FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
n_models = len(LABEL_CONFIGS) * len(SEEDS)
print(f"Features : {len(FEATURE_COLS)}")
print(f"Models   : {len(LABEL_CONFIGS)} labels × {len(SEEDS)} seeds = {n_models}")
print("DART — no early stopping (incompatible with dropout)\n")

g_train = train.groupby("srch_id", sort=False)["srch_id"].count().values
g_val   = val.groupby("srch_id",   sort=False)["srch_id"].count().values

train_ds = lgb.Dataset(train[FEATURE_COLS], label=train["relevance"], group=g_train, free_raw_data=False)
val_ds   = lgb.Dataset(val[FEATURE_COLS],   label=val["relevance"],   group=g_val,   reference=train_ds, free_raw_data=False)

val_acc  = np.zeros(len(val))
test_acc = np.zeros(len(test))

for label_name, label_col in LABEL_CONFIGS:
    g_train_l = train.groupby("srch_id", sort=False)["srch_id"].count().values
    g_val_l   = val.groupby("srch_id",   sort=False)["srch_id"].count().values

    ds_train = lgb.Dataset(train[FEATURE_COLS], label=train[label_col], group=g_train_l, free_raw_data=False)
    ds_val   = lgb.Dataset(val[FEATURE_COLS],   label=val[label_col],   group=g_val_l,   reference=ds_train, free_raw_data=False)

    for seed in SEEDS:
        val_cache  = f"dart_{label_name}_seed{seed}_val.parquet"
        test_cache = f"dart_{label_name}_seed{seed}_test.parquet"

        if os.path.exists(val_cache) and os.path.exists(test_cache):
            print(f"[{label_name} seed={seed}] loading from cache")
            val_scores_df  = pd.read_parquet(val_cache)
            test_scores_df = pd.read_parquet(test_cache)
        else:
            print(f"[{label_name} seed={seed}] training ...")
            params = {
                "boosting_type":               "dart",
                "objective":                   "lambdarank",
                "metric":                      "ndcg",
                "eval_at":                     [5],
                "label_gain":                  [0, 1, 3, 7, 15, 31],
                "lambdarank_truncation_level": 10,
                "drop_rate":                   0.1,
                "max_drop":                    50,
                "num_leaves":                  63,
                "learning_rate":               0.05,
                "min_data_in_leaf":            100,
                "feature_fraction":            0.7,
                "bagging_fraction":            0.7,
                "bagging_freq":                1,
                "seed":                        seed,
                "verbosity":                   -1,
            }
            model = lgb.train(
                params, ds_train,
                num_boost_round=2000,
                valid_sets=[ds_val], valid_names=["val"],
                callbacks=[lgb.log_evaluation(period=100)],
            )

            val_scores_df  = val[["srch_id",  "prop_id"]].copy()
            test_scores_df = test[["srch_id", "prop_id"]].copy()
            val_scores_df["rank_score"]  = model.predict(val[FEATURE_COLS])
            test_scores_df["rank_score"] = model.predict(test[FEATURE_COLS])

            val_scores_df.to_parquet(val_cache,   index=False)
            test_scores_df.to_parquet(test_cache, index=False)
            print(f"  Cached {val_cache} and {test_cache}")

        val_acc  += rank_normalize(val_scores_df,  "rank_score").values
        test_acc += rank_normalize(test_scores_df, "rank_score").values

val["pred_score"]  = val_acc  / n_models
test["pred_score"] = test_acc / n_models

per_query = val.groupby("srch_id").apply(
    lambda g: ndcg_at_k(g["relevance"].values, g["pred_score"].values),
    include_groups=False,
)
print(f"\nDART 9-model ensemble val NDCG@5 ({len(per_query):,} queries): {per_query.mean():.6f}")
print(f"Baseline LightGBM: 0.385000  delta: {per_query.mean() - 0.385:+.6f}")

val[["srch_id",  "prop_id", "pred_score"]].to_csv("dart_val_scores.csv",  index=False)
test[["srch_id", "prop_id", "pred_score"]].to_csv("dart_test_scores.csv", index=False)
print("Exported dart_val_scores.csv and dart_test_scores.csv")
