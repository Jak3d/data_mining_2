import os
import time
import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
import catboost
from catboost.utils import get_gpu_device_count

print(f"CatBoost version: {catboost.__version__}")
n_gpu = get_gpu_device_count()
print(f"GPU devices: {n_gpu}")
if n_gpu == 0:
    raise RuntimeError("No GPU found — YetiRank on CPU is not supported.")

SEEDS = [42, 123, 2024]
DROP_COLS = {
    "srch_id", "date_time", "relevance", "score",
    "click_bool", "booking_bool", "gross_bookings_usd",
    "position", "random_bool",
}
BASELINE = {"fail": 41.8, "partial": 38.9, "perfect": 19.3}

train = pd.read_parquet("prepared_train.parquet").sort_values("srch_id").reset_index(drop=True)
val   = pd.read_parquet("prepared_val.parquet").sort_values("srch_id").reset_index(drop=True)
test  = pd.read_parquet("prepared_test.parquet").sort_values("srch_id").reset_index(drop=True)

train["relevance"] = train["relevance"].clip(upper=5)
val["relevance"]   = val["relevance"].clip(upper=5)

assert (train["srch_id"].values[1:] >= train["srch_id"].values[:-1]).all()
assert (val["srch_id"].values[1:]   >= val["srch_id"].values[:-1]).all()
assert (test["srch_id"].values[1:]  >= test["srch_id"].values[:-1]).all()

FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
print(f"Features : {len(FEATURE_COLS)}")
print(f"Train    : {len(train):,} rows  {train['srch_id'].nunique():,} queries")
print(f"Val      : {len(val):,} rows  {val['srch_id'].nunique():,} queries")
print(f"Test     : {len(test):,} rows  {test['srch_id'].nunique():,} queries")

train_pool = Pool(train[FEATURE_COLS], label=train["relevance"], group_id=train["srch_id"])
val_pool   = Pool(val[FEATURE_COLS],   label=val["relevance"],   group_id=val["srch_id"])
test_pool  = Pool(test[FEATURE_COLS],  group_id=test["srch_id"])


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def bucket_report(per_query: pd.Series, model_label: str) -> None:
    bucket = pd.cut(per_query, bins=[-0.001, 0.0, 0.9999, 1.0001],
                    labels=["fail", "partial", "perfect"])
    stats  = pd.DataFrame({"ndcg": per_query, "bucket": bucket}) \
               .groupby("bucket")["ndcg"].agg(count="count", mean_ndcg="mean")
    stats["pct"] = 100 * stats["count"] / stats["count"].sum()
    print(f"\n{'Bucket':8s}  {'Count':>7s}  {model_label:>8s}  {'%Base':>6s}  {'Δ%':>6s}  {'MeanNDCG':>9s}")
    print("-" * 55)
    for bkt in ["fail", "partial", "perfect"]:
        row   = stats.loc[bkt]
        delta = row["pct"] - BASELINE[bkt]
        print(f"{bkt:8s}  {int(row['count']):>7,}  {row['pct']:>7.1f}%  "
              f"{BASELINE[bkt]:>5.1f}%  {delta:>+5.1f}%  {row['mean_ndcg']:>9.4f}")


def rank_normalize_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("srch_id")[col].rank(pct=True)


val_acc    = np.zeros(len(val))
test_acc   = np.zeros(len(test))
seed_times = []

for seed in SEEDS:
    val_cache  = f"catboost_yeti_seed{seed}_val.parquet"
    test_cache = f"catboost_yeti_seed{seed}_test.parquet"
    t0 = time.time()

    if os.path.exists(val_cache) and os.path.exists(test_cache):
        print(f"\n[seed={seed}] loading from cache")
        val_preds  = pd.read_parquet(val_cache)["pred_score"].values
        test_preds = pd.read_parquet(test_cache)["pred_score"].values
    else:
        print(f"\n[seed={seed}] training ...")
        params = {
            "loss_function":  "YetiRank",
            "iterations":     1505,
            "learning_rate":  0.05,
            "depth":          8,
            "l2_leaf_reg":    3.0,
            "random_seed":    seed,
            "task_type":      "GPU",
            "devices":        "0",
            "verbose":        100,
            "bootstrap_type": "Bernoulli",
            "subsample":      0.8,
        }
        model = CatBoost(params)
        model.fit(train_pool)

        val_preds  = model.predict(val_pool)
        test_preds = model.predict(test_pool)

        val_df  = val[["srch_id",  "prop_id"]].copy()
        test_df = test[["srch_id", "prop_id"]].copy()
        val_df["pred_score"]  = val_preds
        test_df["pred_score"] = test_preds
        val_df.to_parquet(val_cache,   index=False)
        test_df.to_parquet(test_cache, index=False)
        val_df.to_csv(f"catboost_yeti_seed{seed}_val_scores.csv",  index=False)
        test_df.to_csv(f"catboost_yeti_seed{seed}_test_scores.csv", index=False)

        imp = pd.Series(
            model.get_feature_importance(train_pool, type="PredictionValuesChange"),
            index=FEATURE_COLS,
        ).sort_values(ascending=False)
        print(f"\n  Top 20 features seed={seed}:")
        for rank, (feat, gain) in enumerate(imp.head(20).items(), 1):
            print(f"    {rank:2d}. {feat:<42s}  {gain:>10.2f}")

    elapsed = (time.time() - t0) / 60
    seed_times.append(elapsed)
    print(f"  Elapsed: {elapsed:.1f} min")

    val["pred_score"] = val_preds
    per_query = val.groupby("srch_id").apply(
        lambda g: ndcg_at_k(g["relevance"].values, g["pred_score"].values)
    )
    print(f"  Val NDCG@5 seed={seed}: {per_query.mean():.6f}")
    bucket_report(per_query, f"seed{seed}")

    val_acc  += rank_normalize_series(val,  "pred_score").values
    test["pred_score"] = test_preds
    test_acc += rank_normalize_series(test, "pred_score").values

n_models = len(SEEDS)
val["pred_score"]  = val_acc  / n_models
test["pred_score"] = test_acc / n_models

per_query_ens = val.groupby("srch_id").apply(
    lambda g: ndcg_at_k(g["relevance"].values, g["pred_score"].values)
)
print(f"\nCatBoost 3-seed ensemble val NDCG@5: {per_query_ens.mean():.6f}")
bucket_report(per_query_ens, "%Ens")

print(f"\nPer-seed runtimes:")
for seed, t in zip(SEEDS, seed_times):
    print(f"  seed={seed}: {t:.1f} min")
print(f"  Total: {sum(seed_times):.1f} min")

val[["srch_id",  "prop_id", "pred_score"]].to_csv("catboost_val_scores.csv",  index=False)
test[["srch_id", "prop_id", "pred_score"]].to_csv("catboost_test_scores.csv", index=False)
print("Exported catboost_val_scores.csv and catboost_test_scores.csv")
