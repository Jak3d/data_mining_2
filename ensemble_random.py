import pandas as pd
import numpy as np

GROUP_COL = "srch_id"

def ndcg_at_k(relevance, scores, k=5):
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

labels  = pd.read_parquet("prepared_val.parquet")[["srch_id", "prop_id", "relevance"]]

lgbm = pd.read_csv("lightgbm_val_scores.csv").rename(columns={"pred_score": "lgbm"})
nn   = pd.read_csv("neural_val_scores.csv").rename(columns={"pred_score": "nn"})
lgbm_r = pd.read_csv("lightgbm_random_val_scores.csv").rename(columns={"pred_score": "lgbm_r"})
nn_r   = pd.read_csv("neural_random_val_scores.csv").rename(columns={"pred_score": "nn_r"})

val = labels.merge(lgbm, on=["srch_id","prop_id"]) \
            .merge(nn,   on=["srch_id","prop_id"]) \
            .merge(lgbm_r, on=["srch_id","prop_id"]) \
            .merge(nn_r,   on=["srch_id","prop_id"])

def score(cols):
    combined = val[cols].mean(axis=1)
    return val.assign(p=combined).groupby(GROUP_COL).apply(
        lambda g: ndcg_at_k(g["relevance"].values, g["p"].values)
    ).mean()

base   = score(["lgbm", "nn"])
new    = score(["lgbm_r", "nn_r"])
all20  = score(["lgbm", "nn", "lgbm_r", "nn_r"])

print(f"18 models (base):          {base:.6f}")
print(f"2 random models only:      {new:.6f}")
print(f"All 20:                    {all20:.6f}")
print(f"Gain from adding random:   {all20 - base:+.6f}")

if all20 - base >= 0.002:
    print("\nGain ≥ 0.002 — generating submission_with_random.csv")
    test_lgbm   = pd.read_csv("lightgbm_test_scores.csv").rename(columns={"pred_score": "lgbm"})
    test_nn     = pd.read_csv("neural_test_scores.csv").rename(columns={"pred_score": "nn"})
    test_lgbm_r = pd.read_csv("lightgbm_random_test_scores.csv").rename(columns={"pred_score": "lgbm_r"})
    test_nn_r   = pd.read_csv("neural_random_test_scores.csv").rename(columns={"pred_score": "nn_r"})

    test = test_lgbm.merge(test_nn, on=["srch_id","prop_id"]) \
                    .merge(test_lgbm_r, on=["srch_id","prop_id"]) \
                    .merge(test_nn_r,   on=["srch_id","prop_id"])
    test["ensemble"] = test[["lgbm","nn","lgbm_r","nn_r"]].mean(axis=1)

    submission = (test[["srch_id","prop_id","ensemble"]]
                  .sort_values(["srch_id","ensemble"], ascending=[True, False])
                  .drop(columns="ensemble")
                  .reset_index(drop=True))
    submission.to_csv("submission_with_random.csv", index=False)
    print(f"submission_with_random.csv  ({len(submission):,} rows)")
else:
    print(f"\nGain < 0.002 — skip new submission.")
