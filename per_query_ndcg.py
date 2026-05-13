import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUP_COL = "srch_id"

def ndcg_at_k(relevance, scores, k=5):
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

labels = pd.read_parquet("prepared_val.parquet")[["srch_id", "prop_id", "relevance"]]
lgbm   = pd.read_csv("lightgbm_val_scores.csv").rename(columns={"pred_score": "lgbm"})
nn     = pd.read_csv("neural_val_scores.csv").rename(columns={"pred_score": "nn"})

val = labels.merge(lgbm, on=["srch_id","prop_id"]).merge(nn, on=["srch_id","prop_id"])

if os.path.exists("lightgbm_random_val_scores.csv") and os.path.exists("neural_random_val_scores.csv"):
    lgbm_r = pd.read_csv("lightgbm_random_val_scores.csv").rename(columns={"pred_score": "lgbm_r"})
    nn_r   = pd.read_csv("neural_random_val_scores.csv").rename(columns={"pred_score": "nn_r"})
    val    = val.merge(lgbm_r, on=["srch_id","prop_id"]).merge(nn_r, on=["srch_id","prop_id"])
    score_cols = ["lgbm", "nn", "lgbm_r", "nn_r"]
    label = "20-model ensemble"
else:
    score_cols = ["lgbm", "nn"]
    label = "18-model ensemble"

val["ensemble"] = val[score_cols].mean(axis=1)

per_query = val.groupby(GROUP_COL).apply(
    lambda g: ndcg_at_k(g["relevance"].values, g["ensemble"].values)
).values

mean_ndcg = per_query.mean()
frac_zero = (per_query == 0).mean()
frac_one  = (per_query == 1).mean()
frac_mid  = ((per_query > 0) & (per_query < 1)).mean()

print(f"Ensemble: {label}")
print(f"Mean NDCG@5:        {mean_ndcg:.6f}")
print(f"Fraction NDCG = 0:  {frac_zero:.3f}  ({frac_zero*100:.1f}%)")
print(f"Fraction NDCG = 1:  {frac_one:.3f}  ({frac_one*100:.1f}%)")
print(f"Fraction NDCG in (0,1): {frac_mid:.3f}  ({frac_mid*100:.1f}%)")

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(per_query, bins=50, edgecolor="black", linewidth=0.4)
ax.set_xlabel("NDCG@5")
ax.set_ylabel("Number of queries")
ax.set_title(f"Per-query NDCG@5 distribution — {label}\n"
             f"mean={mean_ndcg:.4f}  zero={frac_zero:.1%}  perfect={frac_one:.1%}")
plt.tight_layout()
plt.savefig("figures/per_query_ndcg.png", dpi=150)
plt.close()
print("Saved figures/per_query_ndcg.png")
