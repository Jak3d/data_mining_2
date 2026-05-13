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

# --- load val + ensemble scores ---
val    = pd.read_parquet("prepared_val.parquet")
lgbm   = pd.read_csv("lightgbm_val_scores.csv").rename(columns={"pred_score": "lgbm"})
nn     = pd.read_csv("neural_val_scores.csv").rename(columns={"pred_score": "nn"})
scores = lgbm.merge(nn, on=["srch_id", "prop_id"])
scores["ensemble"] = scores[["lgbm", "nn"]].mean(axis=1)
val = val.merge(scores[["srch_id", "prop_id", "ensemble"]], on=["srch_id", "prop_id"])

# --- per-query NDCG ---
per_query_ndcg = val.groupby(GROUP_COL).apply(
    lambda g: ndcg_at_k(g["relevance"].values, g["ensemble"].values)
).rename("ndcg")

# --- query-level features ---
def query_features(g):
    price = g["price_usd"]
    price_min = price.min()
    return pd.Series({
        "n_items":            len(g),
        "n_bookings":         g["booking_bool"].sum(),
        "n_clicks":           g["click_bool"].sum(),
        "n_positives":        (g["relevance"] > 0).sum(),
        "max_relevance":      g["relevance"].max(),
        "price_min":          price_min,
        "price_max":          price.max(),
        "price_mean":         price.mean(),
        "price_std":          price.std(),
        "price_ratio":        price.max() / price_min if price_min > 0 else np.nan,
        "star_min":           g["prop_starrating"].min(),
        "star_max":           g["prop_starrating"].max(),
        "star_std":           g["prop_starrating"].std(),
        "has_visitor_hist":   g["visitor_hist_starrating"].notna().any().astype(int),
        "length_of_stay":     g["srch_length_of_stay"].iloc[0],
        "booking_window":     g["srch_booking_window"].iloc[0],
        "adults_count":       g["srch_adults_count"].iloc[0],
        "children_count":     g["srch_children_count"].iloc[0],
        "room_count":         g["srch_room_count"].iloc[0],
        "saturday_night":     g["srch_saturday_night_bool"].iloc[0],
        "is_random":          g["random_bool"].iloc[0],
        "affinity_score":     g["srch_query_affinity_score"].dropna().iloc[0]
                              if g["srch_query_affinity_score"].notna().any() else np.nan,
        "prop_country_div":   g["prop_country_id"].nunique(),
        "mean_prop_ctr":      g["prop_ctr"].mean() if "prop_ctr" in g.columns else np.nan,
        "std_loc_score2":     g["prop_location_score2"].std(),
    })

print("Computing query-level features...")
qf = val.groupby(GROUP_COL).apply(query_features)
qf = qf.join(per_query_ndcg)

qf["bucket"] = pd.cut(qf["ndcg"],
                       bins=[-0.001, 0.0, 0.9999, 1.0001],
                       labels=["fail", "partial", "perfect"])

# --- counts and mean NDCG per bucket ---
print("\n=== Bucket summary ===")
summary = qf.groupby("bucket")["ndcg"].agg(["count", "mean"])
summary.columns = ["count", "mean_ndcg"]
print(summary.to_string())

# --- mean of every feature per bucket ---
numeric_cols = [c for c in qf.columns if c not in ["ndcg", "bucket"]]
bucket_means = qf.groupby("bucket")[numeric_cols].mean()
print("\n=== Feature means per bucket ===")
print(bucket_means.T.to_string(float_format="{:.3f}".format))

# --- standardized difference: fail vs partial ---
fail    = qf[qf["bucket"] == "fail"][numeric_cols]
partial = qf[qf["bucket"] == "partial"][numeric_cols]

mean_diff  = fail.mean() - partial.mean()
pooled_std = np.sqrt(
    (fail.std() ** 2 * (len(fail) - 1) + partial.std() ** 2 * (len(partial) - 1))
    / (len(fail) + len(partial) - 2)
).replace(0, np.nan)

std_diff = (mean_diff / pooled_std).abs().sort_values(ascending=False)

print("\n=== Top 15 features by |standardized difference| (fail vs partial) ===")
top15 = std_diff.head(15)
for feat, val_s in top15.items():
    direction = "↑ in fail" if mean_diff[feat] > 0 else "↓ in fail"
    print(f"  {feat:30s}  d={val_s:.4f}  {direction}")

# --- save ---
qf.to_csv("failure_analysis.csv", index=True)
print("\nSaved failure_analysis.csv")
