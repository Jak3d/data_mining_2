import itertools
import time
import pandas as pd
import numpy as np
from catboost import CatBoost, Pool

from prepare_data import (
    load_data, prepare, split_train_val,
    build_target_encodings,
    compute_clip_bounds, CLIP_COLS,
)

TRAIN_PATH = "training_set_VU_DM.csv"
TEST_PATH  = "test_set_VU_DM.csv"

def _ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

SMOOTHING_VALUES = [1, 5, 10, 25, 50, 100]
CLICK_WEIGHTS    = [0.01, 0.05, 0.1, 0.2]

DROP_COLS = ["srch_id", "date_time", "relevance", "score",
             "click_bool", "booking_bool", "gross_bookings_usd"]
GROUP_COL = "srch_id"

PROXY_PARAMS = {
    "iterations":      200,
    "depth":           6,
    "learning_rate":   0.1,
    "loss_function":   "YetiRankPairwise",
    "bootstrap_type":  "Bernoulli",
    "subsample":       0.8,
    "custom_metric":   [],
    "task_type":       "GPU",
    "verbose":         50,
}

print("Loading data...")
train_raw = load_data(TRAIN_PATH)
test_raw  = load_data(TEST_PATH)

clip_bounds = compute_clip_bounds(train_raw, CLIP_COLS)
price_cap   = train_raw["price_usd"].quantile(0.995)

print("Preparing base datasets...")
train_base = prepare(train_raw, is_train=True,  clip_bounds=clip_bounds, price_cap=price_cap)
test_base  = prepare(test_raw,  is_train=False, clip_bounds=clip_bounds, price_cap=price_cap)

train_fold_base, val_fold_base = split_train_val(train_base)

print(f"Grid: {len(SMOOTHING_VALUES)} smoothing × {len(CLICK_WEIGHTS)} click weights "
      f"= {len(SMOOTHING_VALUES) * len(CLICK_WEIGHTS)} trials\n")

results = []

for smoothing, click_w in itertools.product(SMOOTHING_VALUES, CLICK_WEIGHTS):
    train_fold = train_fold_base.copy()
    val_fold   = val_fold_base.copy()

    train_fold["relevance"] = train_fold["booking_bool"] * 5 + train_fold["click_bool"]
    val_fold["relevance"]   = val_fold["booking_bool"]   * 5 + val_fold["click_bool"]
    train_fold["score"]     = train_fold["booking_bool"] + click_w * train_fold["click_bool"]
    val_fold["score"]       = val_fold["booking_bool"]   + click_w * val_fold["click_bool"]

    encoding_specs = [
        (["prop_id"],                        "click_bool",   "prop_ctr"),
        (["prop_id"],                        "booking_bool", "prop_cvr"),
        (["srch_destination_id"],            "booking_bool", "dest_cvr"),
        (["prop_id", "srch_booking_window"], "booking_bool", "prop_cvr_bw"),
    ]
    for keys, target, name in encoding_specs:
        enc         = build_target_encodings(train_fold, keys, target, smoothing=smoothing)
        global_mean = train_fold[target].mean()
        train_fold[name] = train_fold.set_index(keys).index.map(enc.get).fillna(global_mean)
        val_fold[name]   = val_fold.set_index(keys).index.map(enc.get).fillna(global_mean)

    FEATURE_COLS = [c for c in train_fold.columns if c not in DROP_COLS]

    train_pool = Pool(train_fold[FEATURE_COLS], label=train_fold["relevance"], group_id=train_fold[GROUP_COL])
    val_pool   = Pool(val_fold[FEATURE_COLS],   label=val_fold["relevance"],   group_id=val_fold[GROUP_COL])

    t0 = time.time()
    print(f"  [trial {len(results)+1}/24] smoothing={smoothing}  click_w={click_w}  training...", flush=True)

    model = CatBoost(PROXY_PARAMS)
    model.fit(train_pool)

    print(f"    training done ({time.time()-t0:.0f}s)  computing NDCG...", flush=True)
    preds = model.predict(val_pool)
    val_fold["pred"] = preds
    ndcg = val_fold.groupby(GROUP_COL).apply(
        lambda g: _ndcg_at_k(g["relevance"].values, g["pred"].values, k=5)
    ).mean()
    results.append({"smoothing": smoothing, "click_weight": click_w, "ndcg": ndcg})
    print(f"    → NDCG={ndcg:.6f}  (total {time.time()-t0:.0f}s)", flush=True)

results_df = pd.DataFrame(results).sort_values("ndcg", ascending=False)
print("\n--- TOP 5 CONFIGURATIONS ---")
print(results_df.head())

best = results_df.iloc[0]
print(f"\nBest: smoothing={best.smoothing}, click_weight={best.click_weight}, NDCG={best.ndcg:.6f}")
