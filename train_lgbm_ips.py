import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

CLAMP = 0.05   # minimum propensity — caps IPS weights at 20× after smoothing

# ── STEP 1: ESTIMATE POSITION PROPENSITIES FROM RANDOM IMPRESSIONS ───────────
# position is dropped during prepare_data.py, so we go back to the raw CSV.
# random_bool=1 rows had their display order randomised by Expedia, which
# breaks the feedback loop and gives an unbiased view of position effects.

print("Loading raw CSV for propensity estimation ...")
raw = pd.read_csv(
    "training_set_VU_DM.csv",
    na_values="NULL",
    usecols=["srch_id", "prop_id", "position", "click_bool", "random_bool"],
)

rand = raw[raw["random_bool"] == 1].copy()
print(f"  Random-display rows: {len(rand):,}  "
      f"({100 * len(rand) / len(raw):.1f}% of raw train)")

# clicks per position (raw counts — under random display, impressions are
# roughly uniform across positions, so counts are proportional to click rate)
pos_clicks = rand.groupby("position")["click_bool"].sum().sort_index()
max_clicks  = pos_clicks.max()
raw_propensity = (pos_clicks / max_clicks).to_dict()   # normalized to [0, 1]

# ── SMOOTH THE PROPENSITY CURVE ───────────────────────────────────────────────
# The Expedia dataset has a well-known anomaly at position 5 (propensity ≈ 0.007)
# caused by a UI element (fold/ad slot) that suppresses clicks regardless of
# item quality.  Fitting isotonic regression with a decreasing constraint
# recovers a monotone curve and fixes the outlier without manual heuristics.
positions_arr   = np.array(sorted(raw_propensity))
raw_values      = np.array([raw_propensity[p] for p in positions_arr])

iso = IsotonicRegression(increasing=False)
smoothed_values = iso.fit_transform(positions_arr, raw_values)
# Re-normalize so the smoothed curve still peaks at 1.0
smoothed_values = smoothed_values / smoothed_values.max()

propensity = dict(zip(positions_arr.tolist(), smoothed_values.tolist()))

print(f"\nTop 10 position propensities (raw → smoothed):")
for pos in sorted(raw_propensity)[:10]:
    print(f"  position {pos:3d}:  raw={raw_propensity[pos]:.4f}  "
          f"smoothed={propensity[pos]:.4f}")

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 5))
positions_sorted = sorted(propensity)
ax.bar(positions_sorted,
       [raw_propensity.get(p, np.nan) for p in positions_sorted],
       width=0.8, color="lightsteelblue", edgecolor="white",
       linewidth=0.3, label="raw")
ax.plot(positions_sorted,
        [propensity[p] for p in positions_sorted],
        color="steelblue", linewidth=1.8, label="isotonic smoothed")
ax.axhline(CLAMP, color="tomato", linewidth=1.2, linestyle="--",
           label=f"clamp floor = {CLAMP}")
ax.set_xlabel("Position in search results")
ax.set_ylabel("Propensity  (relative click rate, max = 1.0)")
ax.set_title("Position propensities — raw vs isotonic-smoothed\n"
             "(position 5 anomaly corrected; used for IPS training weights)")
ax.set_xlim(0.5, min(max(positions_sorted), 50) + 0.5)
ax.legend()
plt.tight_layout()
plt.savefig("figures/position_propensities.png", dpi=150)
plt.close()
print("Saved figures/position_propensities.png")

# ── STEP 2: COMPUTE SAMPLE WEIGHTS ALIGNED TO PREPARED PARQUET ───────────────
# Alignment strategy: merge position onto the prepared parquet via srch_id +
# prop_id (unique per search in the Expedia dataset).  Adding position as a
# temporary column before sort_values keeps weights correctly aligned.

print("\nLoading prepared parquets ...")
train = pd.read_parquet("prepared_train.parquet")
val   = pd.read_parquet("prepared_val.parquet")
test  = pd.read_parquet("prepared_test.parquet")

raw_pos = (raw[["srch_id", "prop_id", "position"]]
           .drop_duplicates(subset=["srch_id", "prop_id"]))

n_before = len(train)
train = train.merge(raw_pos, on=["srch_id", "prop_id"], how="left")
assert len(train) == n_before, (
    f"Row count changed after merge ({n_before} → {len(train)}); "
    "srch_id + prop_id is not unique in the raw CSV"
)

n_missing_pos = train["position"].isna().sum()
if n_missing_pos:
    print(f"  Warning: {n_missing_pos:,} train rows had no position in raw CSV "
          "— assigned weight 1.0")

def pos_to_weight(pos):
    if pd.isna(pos):
        return 1.0
    p = propensity.get(int(pos), CLAMP)
    return 1.0 / max(p, CLAMP)

train["_ips_w"] = train["position"].map(pos_to_weight)

# normalize so mean weight = 1.0 (preserves overall loss magnitude)
train["_ips_w"] /= train["_ips_w"].mean()

# ── STEP 5 PRE-TRAIN DIAGNOSTICS ─────────────────────────────────────────────
top5_mask          = train["position"] <= 5
frac_top5_rows     = top5_mask.mean()
w_top5             = train.loc[top5_mask,  "_ips_w"].sum()
w_total            = train["_ips_w"].sum()
frac_top5_eff      = w_top5 / w_total

print("\n── IPS diagnostics ──────────────────────────────────────────")
print(f"  Rows at position ≤ 5 (raw):       {frac_top5_rows:.3f}  "
      f"({100*frac_top5_rows:.1f}%)")
print(f"  Effective weight fraction pos ≤ 5: {frac_top5_eff:.3f}  "
      f"({100*frac_top5_eff:.1f}%)")
print(f"  Weight min / max / mean:           "
      f"{train['_ips_w'].min():.3f} / {train['_ips_w'].max():.3f} / "
      f"{train['_ips_w'].mean():.3f}")
print(f"  Top 5 position propensities:")
for pos in range(1, 6):
    print(f"    position {pos}: {propensity.get(pos, 0.0):.4f}  "
          f"→ IPS weight (before norm) = {1/max(propensity.get(pos, CLAMP), CLAMP):.3f}")
print("─────────────────────────────────────────────────────────────")

# ── LGBM SETUP ────────────────────────────────────────────────────────────────
DROP_COLS    = ["srch_id", "date_time", "relevance", "score",
                "click_bool", "booking_bool", "gross_bookings_usd",
                "random_bool", "position", "_ips_w"]
GROUP_COL    = "srch_id"
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]

# sort by group AFTER weights are attached — same permutation applied to both
train = train.sort_values(GROUP_COL).reset_index(drop=True)
val   = val.sort_values(GROUP_COL).reset_index(drop=True)
test  = test.sort_values(GROUP_COL).reset_index(drop=True)

train["relevance"] = train["relevance"].clip(upper=5)
val["relevance"]   = val["relevance"].clip(upper=5)

weights = train["_ips_w"].values.astype(np.float64)

g_train = train.groupby(GROUP_COL, sort=False)[GROUP_COL].count().values
g_val   = val.groupby(GROUP_COL,   sort=False)[GROUP_COL].count().values
g_test  = test.groupby(GROUP_COL,  sort=False)[GROUP_COL].count().values

# ── STEP 3: TRAIN WITH IPS WEIGHTS ───────────────────────────────────────────
train_ds = lgb.Dataset(
    train[FEATURE_COLS], label=train["relevance"],
    weight=weights, group=g_train,
    free_raw_data=False,
)
val_ds = lgb.Dataset(
    val[FEATURE_COLS], label=val["relevance"], group=g_val,
    reference=train_ds, free_raw_data=False,
)

params = {
    "objective":                    "lambdarank",
    "metric":                       "ndcg",
    "eval_at":                      [5],
    "label_gain":                   [0, 1, 3, 7, 15, 31],
    "lambdarank_truncation_level":  10,
    "num_leaves":                   255,
    "max_depth":                    -1,
    "learning_rate":                0.05,
    "num_iterations":               2000,
    "min_child_samples":            20,
    "feature_fraction":             0.8,
    "bagging_fraction":             0.8,
    "bagging_freq":                 1,
    "reg_alpha":                    0.1,
    "reg_lambda":                   1.0,
    "seed":                         42,
    "verbosity":                    -1,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=100),
]

print(f"\nFeatures : {len(FEATURE_COLS)}")
print(f"Train    : {len(train):,} rows  {train[GROUP_COL].nunique():,} queries")
print(f"Val      : {len(val):,} rows  {val[GROUP_COL].nunique():,} queries")
print()

model = lgb.train(
    params,
    train_ds,
    valid_sets=[val_ds],
    valid_names=["val"],
    callbacks=callbacks,
)

print(f"\nBest iteration: {model.best_iteration}  "
      f"val NDCG@5: {model.best_score['val']['ndcg@5']:.6f}")

# ── STEP 4: EVALUATION ───────────────────────────────────────────────────────

def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


val["pred_score"]  = model.predict(val[FEATURE_COLS])
test["pred_score"] = model.predict(test[FEATURE_COLS])

per_query = val.groupby(GROUP_COL).apply(
    lambda g: ndcg_at_k(g["relevance"].values, g["pred_score"].values)
)
ndcg = per_query.mean()
print(f"Overall val NDCG@5: {ndcg:.6f}")

bucket = pd.cut(per_query, bins=[-0.001, 0.0, 0.9999, 1.0001],
                labels=["fail", "partial", "perfect"])
bucket_df = pd.DataFrame({"ndcg": per_query, "bucket": bucket})
stats = bucket_df.groupby("bucket")["ndcg"].agg(count="count", mean_ndcg="mean")
stats["pct"] = 100 * stats["count"] / stats["count"].sum()

BASELINE = {"fail": 41.8, "partial": 38.9, "perfect": 19.3}
print(f"\n{'Bucket':8s}  {'Count':>7s}  {'%IPS':>6s}  {'%Base':>6s}  {'Δ%':>6s}  {'MeanNDCG':>9s}")
print("-" * 55)
for bkt in ["fail", "partial", "perfect"]:
    row   = stats.loc[bkt]
    delta = row["pct"] - BASELINE[bkt]
    print(f"{bkt:8s}  {int(row['count']):>7,}  {row['pct']:>5.1f}%  "
          f"{BASELINE[bkt]:>5.1f}%  {delta:>+5.1f}%  {row['mean_ndcg']:>9.4f}")

# ── STEP 5 POST-TRAIN: FAIL-BUCKET COMPARISON ─────────────────────────────────
fail_pct = stats.loc["fail", "pct"]
fail_delta = fail_pct - BASELINE["fail"]
print(f"\n── Ethics / bias section summary ────────────────────────────")
print(f"  Position bias correction via IPS (CLAMP={CLAMP}, isotonic-smoothed propensities)")
print(f"  Fail-bucket shift vs baseline: {fail_delta:+.1f}%  "
      f"({'✓ improvement' if fail_delta < -2 else '— below 2% threshold' if fail_delta < 0 else '✗ regression'})")
print(f"  Overall NDCG shift: {ndcg - 0.385:+.4f}  "
      f"({'≥ real-lift threshold' if ndcg >= 0.388 else 'below 0.388 threshold'})")
print("─────────────────────────────────────────────────────────────")

importance = pd.Series(
    model.feature_importance(importance_type="gain"),
    index=FEATURE_COLS,
).sort_values(ascending=False)
print("\nTop 20 features by gain:")
print(importance.head(20).to_string())

val[["srch_id",  "prop_id", "pred_score"]].to_csv("lightgbm_ips_val_scores.csv",  index=False)
test[["srch_id", "prop_id", "pred_score"]].to_csv("lightgbm_ips_test_scores.csv", index=False)
print("\nExported lightgbm_ips_val_scores.csv and lightgbm_ips_test_scores.csv")
