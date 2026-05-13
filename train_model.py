import pandas as pd
import numpy as np
from catboost import CatBoost, Pool
import subprocess

result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else "No GPU detected")

train = pd.read_parquet("prepared_train.parquet")
val   = pd.read_parquet("prepared_val.parquet")
test  = pd.read_parquet("prepared_test.parquet")

DROP_COLS    = ["srch_id", "date_time", "relevance", "score",
                "click_bool", "booking_bool", "gross_bookings_usd"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
TARGET_COL   = "relevance"
GROUP_COL    = "srch_id"

X_train = train[FEATURE_COLS]
y_train = train[TARGET_COL]
g_train = train[GROUP_COL]

X_val   = val[FEATURE_COLS]
y_val   = val[TARGET_COL]
g_val   = val[GROUP_COL]

X_test  = test[FEATURE_COLS]

train_pool = Pool(X_train, label=y_train, group_id=g_train)
val_pool   = Pool(X_val,   label=y_val,   group_id=g_val)
test_pool  = Pool(X_test,  group_id=test[GROUP_COL])

model = CatBoost(params={
    "iterations":      1000,
    "depth":           6,
    "learning_rate":   0.1,
    "loss_function":   "YetiRank",
    "l2_leaf_reg":     3.0,
    "random_strength": 1.0,
    "metric_period":   100,
    "task_type":       "GPU",
    "verbose":         100,
})
model.fit(train_pool)

feature_importance = pd.Series(
    model.get_feature_importance(train_pool),
    index=FEATURE_COLS
).sort_values(ascending=False)
print(feature_importance.head(20))

preds = model.predict(test_pool)
test["pred_score"] = preds
first_search = test[test["srch_id"] == test["srch_id"].iloc[0]].copy()
first_search["rank"] = first_search["pred_score"].rank(ascending=False).astype(int)
print(first_search[["rank", "prop_id", "pred_score"]].sort_values("rank"))

def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

val["pred_score"] = model.predict(val_pool)
ndcg = val.groupby("srch_id").apply(
    lambda g: ndcg_at_k(g["relevance"].values, g["pred_score"].values)
).mean()
print(f"Validation NDCG@5: {ndcg:.6f}")

first_search = val[val["srch_id"] == val["srch_id"].iloc[0]].copy()
first_search["rank"] = first_search["pred_score"].rank(ascending=False).astype(int)
print(first_search[["rank", "prop_id", "pred_score", "relevance"]].sort_values("rank"))

val[["srch_id", "prop_id", "pred_score"]].to_csv("catboost_val_scores.csv",  index=False)
test[["srch_id", "prop_id", "pred_score"]].to_csv("catboost_test_scores.csv", index=False)
print("Exported catboost_val_scores.csv and catboost_test_scores.csv")
