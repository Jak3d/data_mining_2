import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ["relevance", "booking_bool", "click_bool"]
SEEDS  = [42, 123, 777]
ABBREV = {"relevance": "rel", "booking_bool": "bk", "click_bool": "cl"}

val_ref = pd.read_parquet("prepared_val.parquet")[["srch_id", "prop_id"]]
scores  = val_ref.copy()
order   = []

for prefix, tag in [("lgbm", "lgb"), ("neural", "nn")]:
    for label in LABELS:
        for seed in SEEDS:
            path = f"{prefix}_{label}_seed{seed}_val.parquet"
            if not os.path.exists(path):
                print(f"Missing: {path} — run the ensemble scripts first")
                raise SystemExit(1)
            col = f"{tag}_{ABBREV[label]}_{seed}"
            df  = pd.read_parquet(path)[["srch_id", "prop_id", "rank_score"]]
            scores = scores.merge(df.rename(columns={"rank_score": col}),
                                  on=["srch_id", "prop_id"])
            order.append(col)

corr = scores[order].corr()

os.makedirs("figures", exist_ok=True)
corr.to_csv("correlation_matrix.csv")

fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=0.5, vmax=1.0, square=True, linewidths=0.4,
            xticklabels=order, yticklabels=order, ax=ax)
ax.set_title("Pairwise Pearson correlation — val scores (18 models)")
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png", dpi=150)
plt.close()
print("Saved figures/correlation_matrix.png and correlation_matrix.csv")

lgb_cols = [c for c in order if c.startswith("lgb")]
nn_cols  = [c for c in order if c.startswith("nn")]

def mean_upper(m): return m.values[np.triu_indices(len(m), k=1)].mean()

w_lgb   = mean_upper(corr.loc[lgb_cols, lgb_cols])
w_nn    = mean_upper(corr.loc[nn_cols,  nn_cols])
cross   = corr.loc[lgb_cols, nn_cols].values.mean()

print(f"\nWithin-LGB mean ρ : {w_lgb:.4f}")
print(f"Within-NN  mean ρ : {w_nn:.4f}")
print(f"Cross-class mean ρ : {cross:.4f}")
print(f"Diversity gap (within - cross): {((w_lgb+w_nn)/2 - cross):.4f}")
