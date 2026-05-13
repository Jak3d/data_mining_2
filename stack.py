import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# add / remove models here — missing score files are skipped automatically
MODELS = {
    "lightgbm": "lightgbm",
    "neural":   "neural",
    "aug":      "lightgbm_aug",
    "dm":       "lightgbm_dm",
    "xgb":      "xgboost",
    "catboost": "catboost",
    "dart":     "dart",
}

_DISC = 1.0 / np.log2(np.arange(2, 7))


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def mean_ndcg(df: pd.DataFrame, score_col: str) -> float:
    return df.groupby("srch_id").apply(
        lambda g: ndcg_at_k(g["relevance"].values, g[score_col].values)
    ).mean()


def load_scores(prefix: str, models: dict) -> tuple[pd.DataFrame, list[str]]:
    import os
    dfs, loaded = [], []
    for name, file_prefix in models.items():
        path = f"{file_prefix}_{prefix}_scores.csv"
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(path).rename(columns={"pred_score": name})
        dfs.append(df)
        loaded.append(name)
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["srch_id", "prop_id"])
    return merged, loaded


def rank_normalize(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in score_cols:
        df[col] = df.groupby("srch_id")[col].rank(pct=True)
    return df


def fit_weights(val_df: pd.DataFrame, score_cols: list[str]) -> np.ndarray:
    n = len(score_cols)

    sorted_df  = val_df.sort_values("srch_id").reset_index(drop=True)
    srch_ids   = sorted_df["srch_id"].values
    relevance  = sorted_df["relevance"].values
    score_mat  = sorted_df[score_cols].values

    _, first_occ = np.unique(srch_ids, return_index=True)
    boundaries   = np.append(first_occ, len(srch_ids))
    lengths      = np.diff(boundaries)
    n_queries    = int(len(lengths))
    max_len      = int(lengths.max())

    pad_mask     = np.ones((n_queries, max_len), dtype=bool)
    rel_padded   = np.zeros((n_queries, max_len))
    score_padded = np.zeros((n_queries, max_len, n))
    for q in range(n_queries):
        L  = int(lengths[q])
        sl = slice(int(boundaries[q]), int(boundaries[q + 1]))
        pad_mask[q, :L]     = False
        rel_padded[q, :L]   = relevance[sl]
        score_padded[q, :L] = score_mat[sl]

    ideal_rel  = np.sort(rel_padded, axis=1)[:, ::-1][:, :5]
    ideal_gain = 2.0 ** ideal_rel - 1.0
    idcg       = (ideal_gain * _DISC[None, :ideal_rel.shape[1]]).sum(axis=1)

    def neg_ndcg(logits):
        w                 = np.exp(logits); w /= w.sum()
        stacked           = score_padded @ w
        stacked[pad_mask] = -np.inf
        top_idx           = np.argsort(-stacked, axis=1)[:, :5]
        top_rel           = np.take_along_axis(rel_padded, top_idx, axis=1)
        gains             = (2.0 ** top_rel - 1.0) * _DISC[None, :top_idx.shape[1]]
        dcg               = gains.sum(axis=1)
        return -float(np.where(idcg > 0, dcg / idcg, 0.0).mean())

    result = differential_evolution(
        neg_ndcg,
        bounds=[(-2, 2)] * n,
        seed=42,
        maxiter=100,
        tol=1e-8,
        popsize=20,
    )
    w = np.exp(result.x); w /= w.sum()
    return w, result


print("Loading scores...")
val_scores,  score_cols = load_scores("val",  MODELS)
test_scores, _          = load_scores("test", MODELS)
print(f"  Loaded {len(score_cols)} models: {score_cols}")

val_labels = pd.read_parquet("prepared_val.parquet")[["srch_id", "prop_id", "relevance"]]
val_scores = val_scores.merge(val_labels, on=["srch_id", "prop_id"])

val_scores  = rank_normalize(val_scores,  score_cols)
test_scores = rank_normalize(test_scores, score_cols)

val_scores["avg_score"]  = val_scores[score_cols].mean(axis=1)
test_scores["avg_score"] = test_scores[score_cols].mean(axis=1)

print("\nValidation NDCG@5 — individual models:")
for col in score_cols:
    print(f"  {col:20s}  {mean_ndcg(val_scores, col):.6f}")

avg_ndcg = mean_ndcg(val_scores, "avg_score")
print(f"\n  {'equal average':20s}  {avg_ndcg:.6f}")

print("\nFitting stacker weights...")
weights, opt_result = fit_weights(val_scores, score_cols)

print(f"  Success:        {opt_result.success}")
print(f"  Iterations:     {opt_result.nit}")
print(f"  Function evals: {opt_result.nfev}")
print(f"  Final logits:   {np.round(opt_result.x, 4)}")

print("\nLearned weights:")
for col, w in zip(score_cols, weights):
    print(f"  {col:20s}  {w:.4f}")

val_scores["stacked"]  = sum(weights[i] * val_scores[col]  for i, col in enumerate(score_cols))
test_scores["stacked"] = sum(weights[i] * test_scores[col] for i, col in enumerate(score_cols))

stacked_ndcg = mean_ndcg(val_scores, "stacked")
print(f"\n  {'stacked':20s}  {stacked_ndcg:.6f}")
print(f"  gain over avg:  {stacked_ndcg - avg_ndcg:+.6f}")

submission = (
    test_scores[["srch_id", "prop_id", "stacked"]]
    .sort_values(["srch_id", "stacked"], ascending=[True, False])
    .drop(columns="stacked")
    .reset_index(drop=True)
)
submission.to_csv("submission.csv", index=False)
print(f"\nsubmission.csv  ({len(submission):,} rows, {submission['srch_id'].nunique():,} queries)")

np.random.seed(42)
unique_queries = val_scores["srch_id"].unique()
np.random.shuffle(unique_queries)
half         = len(unique_queries) // 2
val_fit      = val_scores[val_scores["srch_id"].isin(set(unique_queries[:half]))]
val_eval     = val_scores[val_scores["srch_id"].isin(set(unique_queries[half:]))]

weights_ho, _ = fit_weights(val_fit, score_cols)
val_eval = val_eval.copy()
val_eval["stacked_ho"] = sum(weights_ho[i] * val_eval[col] for i, col in enumerate(score_cols))
honest_ndcg = mean_ndcg(val_eval, "stacked_ho")

print(f"\n── Overfit check ──────────────────────────────────────────────")
print(f"  In-sample  NDCG (full val, full weights):  {stacked_ndcg:.6f}")
print(f"  Held-out   NDCG (half val, half weights):  {honest_ndcg:.6f}")
print(f"  Optimism:                                  {stacked_ndcg - honest_ndcg:+.6f}")
if stacked_ndcg - honest_ndcg > 0.002:
    print("  *** optimism > 0.002 — use equal-weight average for submission")
else:
    print("  optimism within noise — stacked weights are trustworthy")
