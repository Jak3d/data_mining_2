import pandas as pd
import numpy as np

def load_scores(prefix):
    cb  = pd.read_csv(f"catboost_{prefix}_scores.csv").rename(columns={"pred_score": "catboost"})
    lin = pd.read_csv(f"linear_{prefix}_scores.csv").rename(columns={"pred_score": "linear"})
    neu = pd.read_csv(f"neural_{prefix}_scores.csv").rename(columns={"pred_score": "neural"})
    # xgb = pd.read_csv(f"xgboost_{prefix}_scores.csv").rename(columns={"pred_score": "xgboost"})
    return cb.merge(lin, on=["srch_id", "prop_id"]) \
             .merge(neu, on=["srch_id", "prop_id"])

SCORE_COLS = ["catboost", "linear", "neural"]
# SCORE_COLS = ["catboost", "linear", "neural", "xgboost"]

def normalize_and_ensemble(df):
    df = df.copy()
    for col in SCORE_COLS:
        lo, hi = df[col].min(), df[col].max()
        df[col] = (df[col] - lo) / (hi - lo + 1e-9)
    df["ensemble_score"] = df[SCORE_COLS].mean(axis=1)
    return df

def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

# VAL — evaluate each model and the ensemble
val_scores = normalize_and_ensemble(load_scores("val"))
val_labels  = pd.read_parquet("prepared_val.parquet")[["srch_id", "prop_id", "relevance"]]
val_scores  = val_scores.merge(val_labels, on=["srch_id", "prop_id"])

print("Validation NDCG@5:")
for col in SCORE_COLS + ["ensemble_score"]:
    ndcg = val_scores.groupby("srch_id").apply(
        lambda g: ndcg_at_k(g["relevance"].values, g[col].values)
    ).mean()
    print(f"  {col:20s}  {ndcg:.6f}")

# TEST — build submission ranked by ensemble score within each query
test_scores = normalize_and_ensemble(load_scores("test"))

submission = (
    test_scores[["srch_id", "prop_id", "ensemble_score"]]
    .sort_values(["srch_id", "ensemble_score"], ascending=[True, False])
    .drop(columns="ensemble_score")
    .reset_index(drop=True)
)

submission.to_csv("submission.csv", index=False)
print(f"\nsubmission.csv  ({len(submission):,} rows, {submission['srch_id'].nunique():,} queries)")
