import pandas as pd
import numpy as np
import lightgbm as lgb

train_full = pd.read_parquet("prepared_train.parquet")
val        = pd.read_parquet("prepared_val.parquet")
test       = pd.read_parquet("prepared_test.parquet")

train = train_full[train_full["random_bool"] == 1].copy()
print(f"Random-only train: {len(train):,} rows  {train['srch_id'].nunique():,} queries"
      f"  ({100*len(train)/len(train_full):.1f}% of full train)")

DROP_COLS    = ["srch_id", "date_time", "relevance", "score",
                "click_bool", "booking_bool", "gross_bookings_usd", "random_bool"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
GROUP_COL    = "srch_id"

for df in [train, val, test]:
    df.sort_values(GROUP_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

train["relevance"] = train["relevance"].clip(upper=5)
val["relevance"]   = val["relevance"].clip(upper=5)

g_train = train.groupby(GROUP_COL, sort=False)[GROUP_COL].count().values
g_val   = val.groupby(GROUP_COL,   sort=False)[GROUP_COL].count().values

train_ds = lgb.Dataset(train[FEATURE_COLS], label=train["relevance"],
                       group=g_train, free_raw_data=False)
val_ds   = lgb.Dataset(val[FEATURE_COLS],   label=val["relevance"],
                       group=g_val, reference=train_ds, free_raw_data=False)

params = {
    "objective":                   "lambdarank",
    "metric":                      "ndcg",
    "eval_at":                     [5],
    "label_gain":                  [0, 1, 3, 7, 15, 31],
    "lambdarank_truncation_level": 10,
    "num_leaves":                  127,
    "learning_rate":               0.05,
    "min_child_samples":           10,
    "feature_fraction":            0.8,
    "bagging_fraction":            0.8,
    "bagging_freq":                1,
    "reg_alpha":                   0.1,
    "reg_lambda":                  1.0,
    "num_iterations":              2000,
    "seed":                        42,
    "verbosity":                  -1,
}

model = lgb.train(
    params, train_ds,
    valid_sets=[val_ds], valid_names=["val"],
    callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)],
)

def ndcg_at_k(relevance, scores, k=5):
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

vp   = model.predict(val[FEATURE_COLS])
ndcg = val.assign(p=vp).groupby(GROUP_COL).apply(
    lambda g: ndcg_at_k(g["relevance"].values, g["p"].values)
).mean()
print(f"\nVal NDCG@5: {ndcg:.6f}  (best iter: {model.best_iteration})")

def rank_pct(df, col):
    return df.groupby(GROUP_COL)[col].rank(pct=True).values

vr = rank_pct(val.assign(p=vp), "p")
tr = rank_pct(test.assign(p=model.predict(test[FEATURE_COLS])), "p")

val[["srch_id",  "prop_id"]].assign(pred_score=vr).to_csv("lightgbm_random_val_scores.csv",  index=False)
test[["srch_id", "prop_id"]].assign(pred_score=tr).to_csv("lightgbm_random_test_scores.csv", index=False)
print("Exported lightgbm_random_val_scores.csv and lightgbm_random_test_scores.csv")
