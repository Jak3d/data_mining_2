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

BASE_PARAMS = {
    "objective":                   "lambdarank",
    "metric":                      "ndcg",
    "eval_at":                     [5],
    "lambdarank_truncation_level": 10,
    "num_leaves":                  255,
    "learning_rate":               0.05,
    "min_child_samples":           20,
    "feature_fraction":            0.8,
    "bagging_fraction":            0.8,
    "bagging_freq":                1,
    "reg_alpha":                   0.1,
    "reg_lambda":                  1.0,
    "num_iterations":              2000,
    "verbosity":                  -1,
}

SEEDS = [42, 123, 777]

LABEL_CONFIGS = [
    ("relevance",    [0, 1, 3, 7, 15, 31]),
    ("booking_bool", [0, 1]),
    ("click_bool",   [0, 1]),
]


def ndcg_at_k(relevance, scores, k=5):
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def rank_pct(df, col):
    return df.groupby(GROUP_COL)[col].rank(pct=True).values


print(f"Features: {len(FEATURE_COLS)}")
print(f"Train: {len(train):,} rows  {train[GROUP_COL].nunique():,} queries")
print(f"Val:   {len(val):,} rows  {val[GROUP_COL].nunique():,} queries")
print(f"Models: {len(LABEL_CONFIGS)} label configs × {len(SEEDS)} seeds = "
      f"{len(LABEL_CONFIGS) * len(SEEDS)} total\n")

label_val_acc  = {label_col: np.zeros(len(val))  for label_col, _ in LABEL_CONFIGS}
label_test_acc = {label_col: np.zeros(len(test)) for label_col, _ in LABEL_CONFIGS}

for label_col, label_gain in LABEL_CONFIGS:
    print(f"\n=== label: {label_col} ===")
    for seed in SEEDS:
        print(f"  seed {seed}")
        params = {
            **BASE_PARAMS,
            "label_gain":            label_gain,
            "seed":                  seed,
            "feature_fraction_seed": seed,
            "bagging_seed":          seed,
        }

        train_ds = lgb.Dataset(
            train[FEATURE_COLS], label=train[label_col],
            group=g_train, free_raw_data=False,
        )
        val_ds = lgb.Dataset(
            val[FEATURE_COLS], label=val[label_col],
            group=g_val, reference=train_ds, free_raw_data=False,
        )

        model = lgb.train(
            params, train_ds,
            valid_sets=[val_ds], valid_names=["val"],
            callbacks=[
                lgb.early_stopping(50, verbose=True),
                lgb.log_evaluation(100),
            ],
        )

        vp = model.predict(val[FEATURE_COLS])
        tp = model.predict(test[FEATURE_COLS])

        vr = rank_pct(val.assign(p=vp),  "p")
        tr = rank_pct(test.assign(p=tp), "p")

        label_val_acc[label_col]  += vr
        label_test_acc[label_col] += tr

        val[["srch_id",  "prop_id"]].assign(rank_score=vr).to_parquet(
            f"lgbm_{label_col}_seed{seed}_val.parquet",  index=False)
        test[["srch_id", "prop_id"]].assign(rank_score=tr).to_parquet(
            f"lgbm_{label_col}_seed{seed}_test.parquet", index=False)

        ndcg = val.assign(p=vp).groupby(GROUP_COL).apply(
            lambda g: ndcg_at_k(g["relevance"].values, g["p"].values)
        ).mean()
        print(f"    iter={model.best_iteration}  ndcg@5={ndcg:.6f}")

    label_val_acc[label_col]  /= len(SEEDS)
    label_test_acc[label_col] /= len(SEEDS)

all_labels = [label_col for label_col, _ in LABEL_CONFIGS]


def score_ensemble(labels):
    combined = sum(label_val_acc[l] for l in labels) / len(labels)
    return val.assign(p=combined).groupby(GROUP_COL).apply(
        lambda g: ndcg_at_k(g["relevance"].values, g["p"].values)
    ).mean()


print("\n--- Per-label NDCG@5 ---")
for label_col in all_labels:
    print(f"  {label_col:20s}  {score_ensemble([label_col]):.6f}")

print("\n--- Ablation: drop one label type ---")
for label_col in all_labels:
    remaining = [l for l in all_labels if l != label_col]
    print(f"  drop {label_col:16s}  {score_ensemble(remaining):.6f}")

full_ndcg = score_ensemble(all_labels)
print(f"\n  all three:            {full_ndcg:.6f}")

val_acc  = sum(label_val_acc[l]  for l in all_labels) / len(all_labels)
test_acc = sum(label_test_acc[l] for l in all_labels) / len(all_labels)

val["pred_score"]  = val_acc
test["pred_score"] = test_acc

val[["srch_id",  "prop_id", "pred_score"]].to_csv("lightgbm_val_scores.csv",  index=False)
test[["srch_id", "prop_id", "pred_score"]].to_csv("lightgbm_test_scores.csv", index=False)
print("Exported lightgbm_val_scores.csv and lightgbm_test_scores.csv")
