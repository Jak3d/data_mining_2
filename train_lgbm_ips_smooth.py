import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

CLAMP_NAIVE  = 0.01   # clamp used in the naive run (for "weight_before" column)
CLAMP_SMOOTH = 0.20   # clamp after isotonic smoothing → max weight 5×

# ── STEP 1: PROPENSITIES — RAW THEN ISOTONIC-SMOOTHED ────────────────────────

print("Loading raw CSV for propensity estimation ...")
raw = pd.read_csv(
    "training_set_VU_DM.csv",
    na_values="NULL",
    usecols=["srch_id", "prop_id", "position", "click_bool", "random_bool"],
)

rand = raw[raw["random_bool"] == 1].copy()
print(f"  Random-display rows: {len(rand):,}  ({100*len(rand)/len(raw):.1f}% of raw train)")

pos_clicks      = rand.groupby("position")["click_bool"].sum().sort_index()
max_clicks      = pos_clicks.max()
empirical_prop  = (pos_clicks / max_clicks).to_dict()   # normalized: position 1 = 1.0

positions_arr   = np.array(sorted(empirical_prop), dtype=float)
raw_values      = np.array([empirical_prop[p] for p in positions_arr])

# Isotonic regression — monotone-decreasing (propensity should fall with position)
iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
smoothed_values = iso.fit_transform(positions_arr, raw_values)
smoothed_values = smoothed_values / smoothed_values.max()   # keep peak = 1.0

smoothed_prop = {int(p): float(s) for p, s in zip(positions_arr, smoothed_values)}

# ── SANITY CHECK: did smoothing flatten everything? ───────────────────────────
final_weights_unnorm = np.array([
    1.0 / max(smoothed_prop.get(int(p), CLAMP_SMOOTH), CLAMP_SMOOTH)
    for p in positions_arr
])
w_range = final_weights_unnorm.max() - final_weights_unnorm.min()
if w_range < 0.2:
    print("\n[STOP] Isotonic smoothing collapsed all weights to a narrow band "
          f"({final_weights_unnorm.min():.3f}–{final_weights_unnorm.max():.3f}). "
          "IPS signal is destroyed — aborting.")
    sys.exit(1)

# ── COMPARISON TABLE: positions 1-15 ─────────────────────────────────────────
print(f"\n{'Pos':>4}  {'Raw prop':>9}  {'Smooth prop':>11}  "
      f"{'W_naive':>9}  {'W_smooth':>9}  {'Anomaly?':>8}")
print("─" * 60)
for pos in range(1, 16):
    rp  = empirical_prop.get(pos, np.nan)
    sp  = smoothed_prop.get(pos, np.nan)
    wn  = 1.0 / max(rp,  CLAMP_NAIVE)  if not np.isnan(rp) else np.nan
    ws  = 1.0 / max(sp,  CLAMP_SMOOTH) if not np.isnan(sp) else np.nan
    flag = " ← ANOMALY" if (not np.isnan(rp) and not np.isnan(sp)
                             and abs(rp - sp) > 0.1) else ""
    rp_s = f"{rp:.4f}" if not np.isnan(rp) else "  n/a "
    sp_s = f"{sp:.4f}" if not np.isnan(sp) else "  n/a "
    wn_s = f"{wn:8.3f}" if not np.isnan(wn) else "    n/a "
    ws_s = f"{ws:8.3f}" if not np.isnan(ws) else "    n/a "
    print(f"{pos:>4}  {rp_s:>9}  {sp_s:>11}  {wn_s:>9}  {ws_s:>9}{flag}")

# ── PLOT: RAW vs SMOOTHED ─────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

plot_positions = [p for p in sorted(empirical_prop) if p <= 50]
raw_p   = [empirical_prop[p] for p in plot_positions]
smooth_p = [smoothed_prop.get(p, np.nan) for p in plot_positions]

ax = axes[0]
ax.bar(plot_positions, raw_p, width=0.8, color="lightsteelblue",
       edgecolor="white", linewidth=0.3, label="raw empirical")
ax.plot(plot_positions, smooth_p, color="steelblue", linewidth=2.0,
        label="isotonic smoothed")
ax.axhline(CLAMP_SMOOTH, color="tomato", linewidth=1.2, linestyle="--",
           label=f"clamp = {CLAMP_SMOOTH}")
ax.set_xlabel("Position")
ax.set_ylabel("Propensity (max = 1.0)")
ax.set_title("Propensities: raw vs smoothed")
ax.legend()

ax = axes[1]
raw_w    = [1.0 / max(empirical_prop.get(p, CLAMP_NAIVE),  CLAMP_NAIVE)
            for p in plot_positions]
smooth_w = [1.0 / max(smoothed_prop.get(p, CLAMP_SMOOTH), CLAMP_SMOOTH)
            for p in plot_positions]
ax.plot(plot_positions, raw_w,    color="lightsteelblue", linewidth=1.5,
        label=f"naive IPS (clamp={CLAMP_NAIVE})")
ax.plot(plot_positions, smooth_w, color="steelblue",      linewidth=2.0,
        label=f"smooth IPS (clamp={CLAMP_SMOOTH})")
ax.set_xlabel("Position")
ax.set_ylabel("IPS weight (pre-normalization)")
ax.set_title("IPS weights: naive vs smooth")
ax.legend()

plt.suptitle("Position propensities and IPS weights — anomaly correction\n"
             "(position 5 raw ≈ 0.007, likely sponsored-slot artifact)",
             fontsize=11)
plt.tight_layout()
plt.savefig("figures/position_propensities_smoothed.png", dpi=150)
plt.close()
print("\nSaved figures/position_propensities_smoothed.png")

# ── STEP 2: ALIGN WEIGHTS TO PREPARED TRAIN PARQUET ──────────────────────────
print("\nLoading prepared parquets ...")
train = pd.read_parquet("prepared_train.parquet")
val   = pd.read_parquet("prepared_val.parquet")
test  = pd.read_parquet("prepared_test.parquet")

raw_pos = raw[["srch_id", "prop_id", "position"]].drop_duplicates(["srch_id", "prop_id"])
n_before = len(train)
train    = train.merge(raw_pos, on=["srch_id", "prop_id"], how="left")
assert len(train) == n_before, "Row count changed after merge — srch_id+prop_id not unique"

n_missing = train["position"].isna().sum()
if n_missing:
    print(f"  Warning: {n_missing:,} rows had no position info → assigned weight 1.0")

def row_weight(pos):
    if pd.isna(pos):
        return 1.0
    p = smoothed_prop.get(int(pos), CLAMP_SMOOTH)
    return 1.0 / max(p, CLAMP_SMOOTH)

train["_ips_w"] = train["position"].map(row_weight)
train["_ips_w"] /= train["_ips_w"].mean()   # normalize: mean = 1.0

# ── DIAGNOSTICS ───────────────────────────────────────────────────────────────
w_arr      = train["_ips_w"].values
top5_mask  = train["position"] <= 5
frac_raw   = top5_mask.mean()
frac_eff   = train.loc[top5_mask, "_ips_w"].sum() / w_arr.sum()

print(f"\n── IPS smooth diagnostics ───────────────────────────────────────")
print(f"  Weight min / max / mean / median:  "
      f"{w_arr.min():.3f} / {w_arr.max():.3f} / "
      f"{w_arr.mean():.3f} / {np.median(w_arr):.3f}")
print(f"  Weight p10 / p50 / p90:            "
      f"{np.percentile(w_arr,10):.3f} / "
      f"{np.percentile(w_arr,50):.3f} / "
      f"{np.percentile(w_arr,90):.3f}")
print(f"  Rows at position ≤ 5 (raw):        {frac_raw:.3f}  ({100*frac_raw:.1f}%)")
print(f"  Effective weight fraction pos ≤ 5: {frac_eff:.3f}  ({100*frac_eff:.1f}%)")
print(f"\n  Top 5 positions → smoothed prop → final weight:")
for pos in range(1, 6):
    sp = smoothed_prop.get(pos, np.nan)
    if np.isnan(sp):
        continue
    w_unnorm = 1.0 / max(sp, CLAMP_SMOOTH)
    # representative normalized weight (multiply by inverse of mean unnorm weight)
    mean_unnorm = np.mean([1.0/max(smoothed_prop.get(int(p), CLAMP_SMOOTH), CLAMP_SMOOTH)
                           for p in positions_arr])
    w_norm = w_unnorm / mean_unnorm
    print(f"    pos {pos}:  prop={sp:.4f}  →  weight={w_norm:.3f}")
print("─────────────────────────────────────────────────────────────────")

# ── STEP 3: LGBM WITH SMOOTHED IPS WEIGHTS ───────────────────────────────────
DROP_COLS    = ["srch_id", "date_time", "relevance", "score",
                "click_bool", "booking_bool", "gross_bookings_usd",
                "random_bool", "position", "_ips_w"]
GROUP_COL    = "srch_id"
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]

train = train.sort_values(GROUP_COL).reset_index(drop=True)
val   = val.sort_values(GROUP_COL).reset_index(drop=True)
test  = test.sort_values(GROUP_COL).reset_index(drop=True)

train["relevance"] = train["relevance"].clip(upper=5)
val["relevance"]   = val["relevance"].clip(upper=5)

weights = train["_ips_w"].values.astype(np.float64)

g_train = train.groupby(GROUP_COL, sort=False)[GROUP_COL].count().values
g_val   = val.groupby(GROUP_COL,   sort=False)[GROUP_COL].count().values
g_test  = test.groupby(GROUP_COL,  sort=False)[GROUP_COL].count().values

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
    params, train_ds,
    valid_sets=[val_ds], valid_names=["val"],
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
ndcg_val = per_query.mean()
print(f"Overall val NDCG@5: {ndcg_val:.6f}")

bucket = pd.cut(per_query, bins=[-0.001, 0.0, 0.9999, 1.0001],
                labels=["fail", "partial", "perfect"])
bucket_df = pd.DataFrame({"ndcg": per_query, "bucket": bucket})
stats = bucket_df.groupby("bucket")["ndcg"].agg(count="count", mean_ndcg="mean")
stats["pct"] = 100 * stats["count"] / stats["count"].sum()

BL_BASE  = {"fail": 41.8, "partial": 38.9, "perfect": 19.3}
BL_NAIVE = {"fail": 42.3, "partial": 39.0, "perfect": 18.8}

print(f"\n{'Bucket':8s}  {'Count':>7s}  {'%Smooth':>8s}  "
      f"{'%Base':>6s}  {'ΔBase':>6s}  {'%Naive':>7s}  {'ΔNaive':>7s}  {'MeanNDCG':>9s}")
print("─" * 72)
for bkt in ["fail", "partial", "perfect"]:
    row     = stats.loc[bkt]
    d_base  = row["pct"] - BL_BASE[bkt]
    d_naive = row["pct"] - BL_NAIVE[bkt]
    print(f"{bkt:8s}  {int(row['count']):>7,}  {row['pct']:>7.1f}%  "
          f"{BL_BASE[bkt]:>5.1f}%  {d_base:>+5.1f}%  "
          f"{BL_NAIVE[bkt]:>6.1f}%  {d_naive:>+6.1f}%  {row['mean_ndcg']:>9.4f}")

importance = pd.Series(
    model.feature_importance(importance_type="gain"),
    index=FEATURE_COLS,
).sort_values(ascending=False)
print("\nTop 20 features by gain:")
print(importance.head(20).to_string())

val[["srch_id",  "prop_id", "pred_score"]].to_csv(
    "lightgbm_ips_smooth_val_scores.csv",  index=False)
test[["srch_id", "prop_id", "pred_score"]].to_csv(
    "lightgbm_ips_smooth_test_scores.csv", index=False)
print("\nExported lightgbm_ips_smooth_val_scores.csv and lightgbm_ips_smooth_test_scores.csv")

# ── STEP 5: ETHICS / BIAS WRITEUP BLOCK ──────────────────────────────────────
fail_delta  = stats.loc["fail",  "pct"] - BL_BASE["fail"]
ndcg_delta  = ndcg_val - 0.385
w_std_naive = np.std([1.0 / max(empirical_prop.get(int(p), CLAMP_NAIVE),  CLAMP_NAIVE)
                      for p in train["position"].dropna().astype(int)])
w_std_smooth = weights.std()
w_max_naive  = max(1.0 / max(empirical_prop.get(int(p), CLAMP_NAIVE), CLAMP_NAIVE)
                   for p in positions_arr)
w_max_smooth = weights.max()

pos5_raw     = empirical_prop.get(5, np.nan)
pos5_smooth  = smoothed_prop.get(5, np.nan)
neighbor_avg = np.mean([empirical_prop.get(4, 0), empirical_prop.get(6, 0)])
anomaly_factor = neighbor_avg / pos5_raw if pos5_raw and pos5_raw > 0 else float("inf")

lift_str = ("real lift" if ndcg_val >= 0.388 and fail_delta <= -1.0
            else "marginal lift" if ndcg_val >= 0.385 and fail_delta <= 0
            else "neutral / regression")

print(f"""
── Bias mitigation summary ──────────────────────────────────────────────
Bias mechanism:  position bias in click_bool labels
Detection:       position-stratified click rates from random_bool=1 rows
                 ({len(rand):,} random-display impressions, {100*len(rand)/len(raw):.1f}% of training data)
Anomaly:         position 5 propensity = {pos5_raw:.4f}
                 ({anomaly_factor:.0f}× lower than neighbors 4+6 avg {neighbor_avg:.4f})
Likely cause:    sponsored-slot or UI fold artifact in Expedia source data
Mitigation:      isotonic monotone-decreasing smoothing (sklearn.isotonic)
                 → clamp at {CLAMP_SMOOTH} (max weight {1/CLAMP_SMOOTH:.0f}×) → normalize (mean=1)
Position 5 fix:  raw {pos5_raw:.4f} → smoothed {pos5_smooth:.4f} (corrected to neighbor level)
Weight change:   max weight {w_max_naive:.1f}× (naive) → {w_max_smooth:.3f}× (smooth, normalized)
Effect on NDCG:  overall ΔNDCG = {ndcg_delta:+.4f}  (vs naive-IPS baseline 0.385)
Fail-bucket Δ:   {fail_delta:+.1f}%  (vs unweighted baseline 41.8%)
Assessment:      {lift_str}
─────────────────────────────────────────────────────────────────────────""")
