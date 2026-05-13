import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

train = pd.read_parquet("prepared_train.parquet")
val   = pd.read_parquet("prepared_val.parquet")
test  = pd.read_parquet("prepared_test.parquet")

DROP_COLS    = ["srch_id", "date_time", "relevance", "score",
                "click_bool", "booking_bool", "gross_bookings_usd"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
GROUP_COL    = "srch_id"

X_train = train[FEATURE_COLS].fillna(0).values
y_train = train["score"].values

X_val   = val[FEATURE_COLS].fillna(0).values
y_val   = val["score"].values

X_test  = test[FEATURE_COLS].fillna(0).values

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

model = Ridge(alpha=10.0)
model.fit(X_train, y_train)

def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

val["pred_score"] = model.predict(X_val)
ndcg = val.groupby(GROUP_COL).apply(
    lambda g: ndcg_at_k(g["relevance"].values, g["pred_score"].values)
).mean()
print(f"Validation NDCG@5: {ndcg:.6f}")

test["pred_score"] = model.predict(X_test)

val[["srch_id", "prop_id", "pred_score"]].to_csv("linear_val_scores.csv",  index=False)
test[["srch_id", "prop_id", "pred_score"]].to_csv("linear_test_scores.csv", index=False)
print("Exported linear_val_scores.csv and linear_test_scores.csv")
