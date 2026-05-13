import pandas as pd
import numpy as np

def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

preds  = pd.read_csv("neural_val_scores.csv")
labels = pd.read_parquet("prepared_val.parquet")[["srch_id", "prop_id", "relevance"]]
df     = preds.merge(labels, on=["srch_id", "prop_id"])

ndcg = df.groupby("srch_id").apply(
    lambda g: ndcg_at_k(g["relevance"].values, g["pred_score"].values)
).mean()

print(f"Neural val NDCG@5: {ndcg:.6f}")
print(f"Queries: {df['srch_id'].nunique():,}   Rows: {len(df):,}")
