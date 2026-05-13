import pandas as pd
import numpy as np
import lightgbm as lgb

train = pd.read_parquet("prepared_train.parquet")
val   = pd.read_parquet("prepared_val.parquet")
test  = pd.read_parquet("prepared_test.parquet")

DROP_COLS    = ["srch_id", "date_time", "relevance", "score",
                "click_bool", "booking_bool", "gross_bookings_usd", "random_bool"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
GROUP_COL    = "srch_id"

train = train.sort_values(GROUP_COL).reset_index(drop=True)
val   = val.sort_values(GROUP_COL).reset_index(drop=True)
test  = test.sort_values(GROUP_COL).reset_index(drop=True)

train["relevance"] = train["relevance"].clip(upper=5)
val["relevance"]   = val["relevance"].clip(upper=5)

g_train = train.groupby(GROUP_COL, sort=False)[GROUP_COL].count().values
g_val   = val.groupby(GROUP_COL,   sort=False)[GROUP_COL].count().values
g_test  = test.groupby(GROUP_COL,  sort=False)[GROUP_COL].count().values

train_ds = lgb.Dataset(
    train[FEATURE_COLS], label=train["relevance"], group=g_train,
    free_raw_data=False,
)
val_ds = lgb.Dataset(
    val[FEATURE_COLS], label=val["relevance"], group=g_val,
    reference=train_ds, free_raw_data=False,
)

params = {
    "objective":                 "lambdarank",
    "metric":                    "ndcg",
    "eval_at":                   [5],
    "label_gain":                [0, 1, 3, 7, 15, 31],
    "lambdarank_truncation_level": 10,
    "num_leaves":                 255,
    "max_depth":                 -1,
    "learning_rate":              0.05,
    "num_iterations":             2000,
    "min_child_samples":          20,
    "feature_fraction":           0.8,
    "bagging_fraction":           0.8,
    "bagging_freq":               1,
    "reg_alpha":                  0.1,
    "reg_lambda":                 1.0,
    "verbosity":                 -1,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=100),
]

print(f"Features: {len(FEATURE_COLS)}")
print(f"Train: {len(train):,} rows  {train[GROUP_COL].nunique():,} queries")
print(f"Val:   {len(val):,} rows  {val[GROUP_COL].nunique():,} queries")
print()

model = lgb.train(
    params,
    train_ds,
    valid_sets=[val_ds],
    valid_names=["val"],
    callbacks=callbacks,
)

print(f"\nBest iteration: {model.best_iteration}  "
      f"val NDCG@5: {model.best_score['val']['ndcg@5']:.6f}")


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


val["pred_score"]  = model.predict(val[FEATURE_COLS])
test["pred_score"] = model.predict(test[FEATURE_COLS])

per_query = val.groupby(GROUP_COL).apply(
    lambda g: ndcg_at_k(g["relevance"].values, g["pred_score"].values)
)
ndcg = per_query.mean()
print(f"Overall val NDCG@5: {ndcg:.6f}")

bucket = pd.cut(per_query, bins=[-0.001, 0.0, 0.9999, 1.0001],
                labels=["fail", "partial", "perfect"])
bucket_df = pd.DataFrame({"ndcg": per_query, "bucket": bucket})
stats = bucket_df.groupby("bucket")["ndcg"].agg(count="count", mean_ndcg="mean")
stats["pct"] = 100 * stats["count"] / stats["count"].sum()

BASELINE = {"fail": 41.8, "partial": 38.9, "perfect": 19.3}
print(f"\n{'Bucket':8s}  {'Count':>7s}  {'%New':>6s}  {'%Base':>6s}  {'Δ%':>6s}  {'MeanNDCG':>9s}")
print("-" * 55)
for bkt in ["fail", "partial", "perfect"]:
    row   = stats.loc[bkt]
    delta = row["pct"] - BASELINE[bkt]
    print(f"{bkt:8s}  {int(row['count']):>7,}  {row['pct']:>5.1f}%  "
          f"{BASELINE[bkt]:>5.1f}%  {delta:>+5.1f}%  {row['mean_ndcg']:>9.4f}")

importance = pd.Series(
    model.feature_importance(importance_type="gain"),
    index=FEATURE_COLS,
).sort_values(ascending=False)
print("\nTop 20 features by gain:")
print(importance.head(20).to_string())

val[["srch_id",  "prop_id", "pred_score"]].to_csv("lightgbm_val_scores.csv",  index=False)
test[["srch_id", "prop_id", "pred_score"]].to_csv("lightgbm_test_scores.csv", index=False)
print("\nExported lightgbm_val_scores.csv and lightgbm_test_scores.csv")
