import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

TRAIN_PATH = "training_set_VU_DM.csv"
TEST_PATH  = "test_set_VU_DM.csv"


#LOAD

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, na_values="NULL", parse_dates=["date_time"])


#MISSING-VALUE IMPUTATION [Wang, 2nd place ICDM 2013]
#
# comp*_rate / comp*_inv   -> 0  (no systematic Expedia vs competitor diff)
# prop_location_score2     -> worst (min) value  (missingness = bad listing)
# srch_query_affinity_score-> worst (min) value
# visitor_hist_*           -> left as NaN; handled via match/mismatch (#5)

COMP_RATE_COLS = [f"comp{i}_rate"              for i in range(1, 9)]
COMP_INV_COLS  = [f"comp{i}_inv"               for i in range(1, 9)]
COMP_DIFF_COLS = [f"comp{i}_rate_percent_diff"  for i in range(1, 9)]

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Competitor fields -> 0
    for col in COMP_RATE_COLS + COMP_INV_COLS + COMP_DIFF_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Worst-case for sparse score / affinity fields
    for col in ["prop_location_score2", "srch_query_affinity_score"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].min())

    return df


#MATCH / MISMATCH INDICATORS [Wang, 2nd place ICDM 2013] 
#
# visitor_hist_* are ~95% missing.  Instead of imputing, signal alignment.

def add_visitor_history_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_history = df["visitor_hist_starrating"].notna()

    df["hist_starrating_match"]    = np.where(
        has_history,
        (df["visitor_hist_starrating"] == df["prop_starrating"]).astype(int),
        0,
    )
    df["hist_starrating_mismatch"] = np.where(
        has_history,
        (df["visitor_hist_starrating"] != df["prop_starrating"]).astype(int),
        0,
    )
    df["hist_price_match"]         = np.where(
        has_history & df["visitor_hist_adr_usd"].notna(),
        (df["visitor_hist_adr_usd"] >= df["price_usd"]).astype(int),
        0,
    )
    df["hist_price_mismatch"]      = np.where(
        has_history & df["visitor_hist_adr_usd"].notna(),
        (df["visitor_hist_adr_usd"] < df["price_usd"]).astype(int),
        0,
    )
    return df


#GAP / ECONOMIC FEATURES [Liu et al., 5th place ICDM 2013]

def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    df["ump"]       = np.exp(df["prop_log_historical_price"]) - df["price_usd"]
    df["per_fee"]   = df["price_usd"] * df["srch_room_count"] / (
                          df["srch_adults_count"] + df["srch_children_count"] + eps)
    df["total_fee"] = df["price_usd"] * df["srch_room_count"]
    df["score1d2"]  = df["prop_location_score2"] / (df["prop_location_score1"] + eps)
    df["score2ma"]  = df["prop_location_score2"] * df["srch_query_affinity_score"].fillna(0)

    # Gap vs visitor history (valid only when history exists; else 0)
    has_hist = df["visitor_hist_adr_usd"].notna()
    df["price_diff"]      = np.where(has_hist, df["visitor_hist_adr_usd"] - df["price_usd"], 0)
    df["starrating_diff"] = np.where(
        df["visitor_hist_starrating"].notna(),
        df["visitor_hist_starrating"] - df["prop_starrating"],
        0,
    )
    return df


#RANK FEATURES and LISTWISE NORMALIZATION [Wang, 2nd place ICDM 2013]

LISTWISE_FEATURES = [
    "price_usd",
    "prop_starrating",
    "prop_location_score1",
    "prop_location_score2",
    "prop_review_score",
]

def add_listwise_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby("srch_id")

    for feat in LISTWISE_FEATURES:
        if feat not in df.columns:
            continue
        mean = grp[feat].transform("mean")
        std  = grp[feat].transform("std").replace(0, eps := 1e-6)

        df[f"{feat}_rank"]    = grp[feat].rank(method="average")
        df[f"{feat}_demean"]  = df[feat] - mean
        df[f"{feat}_zscore"]  = (df[feat] - mean) / std

    return df


#CTR / BOOKING RATE TARGET ENCODING [Liu et al., 5th place ICDM 2013]
#
# Applied with leave-one-out on the training fold only.
# Call encode_targets(train) to get back the encoded train frame.
# At test time pass the lookup dicts produced by build_target_encodings().

def build_target_encodings(
    df: pd.DataFrame,
    keys: list[str],
    target: str,
    smoothing: float = 10.0,
) -> pd.Series:
    global_mean = df[target].mean()
    agg = df.groupby(keys)[target].agg(["sum", "count"])
    encoded = (agg["sum"] + smoothing * global_mean) / (agg["count"] + smoothing)
    return encoded


def add_target_encodings(
    train: pd.DataFrame,
    test: pd.DataFrame,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train.copy(), test.copy()

    encoding_specs = [
        (["prop_id"],                        "click_bool",   "prop_ctr"),
        (["prop_id"],                        "booking_bool", "prop_cvr"),
        (["srch_destination_id"],            "booking_bool", "dest_cvr"),
        (["prop_id", "srch_booking_window"], "booking_bool", "prop_cvr_bw"),
    ]

    for keys, target, name in encoding_specs:
        global_mean = train[target].mean()
        agg_sum     = train.groupby(keys)[target].transform("sum")
        agg_count   = train.groupby(keys)[target].transform("count")
        # LOO: subtract current row so the model can't memorise its own label
        train[name] = (agg_sum - train[target] + smoothing * global_mean) / (agg_count - 1 + smoothing)
        # Global smoothed encoding for held-out sets (no leakage risk there)
        enc_lookup  = build_target_encodings(train, keys, target, smoothing)
        test[name]  = test.set_index(keys).index.map(enc_lookup.get).fillna(global_mean)

    return train, test


#DESTINATION-SEASON INTERACTION FEATURES
#
# Captures "this destination is in-season during this month" signal.
# All aggregates computed from train_fold ONLY, then mapped onto val/test.
# Requires dest_cvr from add_target_encodings — call this after that step.

def add_destination_season_features(
    train: pd.DataFrame,
    held_out: pd.DataFrame,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, held_out = train.copy(), held_out.copy()
    keys = ["srch_destination_id", "month"]
    global_mean = train["booking_bool"].mean()
    eps = 1e-6

    # dest_month_cvr: LOO on train, smoothed lookup on held_out
    agg_sum   = train.groupby(keys)["booking_bool"].transform("sum")
    agg_count = train.groupby(keys)["booking_bool"].transform("count")
    train["dest_month_cvr"] = (
        (agg_sum - train["booking_bool"] + smoothing * global_mean)
        / (agg_count - 1 + smoothing)
    )
    dm_agg  = train.groupby(keys)["booking_bool"].agg(["sum", "count"])
    dm_enc  = (dm_agg["sum"] + smoothing * global_mean) / (dm_agg["count"] + smoothing)
    dm_dict = dm_enc.to_dict()   # {(dest_id, month): value}

    held_out["dest_month_cvr"] = [
        dm_dict.get((d, m), float("nan"))
        for d, m in zip(held_out["srch_destination_id"], held_out["month"])
    ]
    # Cold-start (destination, month) pairs fall back to destination-only cvr
    cold = held_out["dest_month_cvr"].isna()
    held_out.loc[cold, "dest_month_cvr"] = held_out.loc[cold, "dest_cvr"]

    # dest_month_lift: > 1 = in-season, < 1 = off-season, ≈ 1 = no seasonality
    for df in [train, held_out]:
        df["dest_month_lift"] = df["dest_month_cvr"] / (df["dest_cvr"] + eps)

    # dest_month_price_ratio: captures peak-season pricing effects
    dm_price   = train.groupby(keys)["price_usd"].median().to_dict()
    dest_price = train.groupby("srch_destination_id")["price_usd"].median().to_dict()
    for df in [train, held_out]:
        dm_p   = np.array([dm_price.get((d, m), float("nan"))
                           for d, m in zip(df["srch_destination_id"], df["month"])],
                          dtype=float)
        dest_p = df["srch_destination_id"].map(dest_price).values.astype(float)
        ratio  = dm_p / (dest_p + eps)
        df["dest_month_price_ratio"] = np.where(np.isnan(ratio), 1.0, ratio)

    # dest_month_n_searches_log + is_peak_month_for_dest
    dm_searches = train.groupby(keys)["srch_id"].nunique()
    dm_s_dict   = dm_searches.to_dict()
    dm_s_df     = dm_searches.rename("n_searches").reset_index()
    dest_p75    = (dm_s_df.groupby("srch_destination_id")["n_searches"]
                          .quantile(0.75).to_dict())

    for df in [train, held_out]:
        n_s = np.array([dm_s_dict.get((d, m), 0)
                        for d, m in zip(df["srch_destination_id"], df["month"])],
                       dtype=float)
        df["dest_month_n_searches_log"] = np.log1p(n_s)
        threshold = df["srch_destination_id"].map(dest_p75).fillna(0).values
        df["is_peak_month_for_dest"] = (n_s > threshold).astype(int)

    return train, held_out


#WEIGHTED TARGET CONSTRUCTION [Liu et al. & Wang, ICDM 2013]
#
# Relevance: booking=5, click=1, none=0  (for LambdaMART / NDCG optimisation)
# Score:     booking + 0.05 * click      (for regression / pointwise models)

def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["relevance"] = df["booking_bool"] * 5 + df["click_bool"] * 1
    df["score"]     = df["booking_bool"] + 0.05 * df["click_bool"]
    return df


#QUERY-LEVEL DIFFICULTY FEATURES
#
# Same value for every row in a query. prop_ctr-dependent feature
# (query_mean_prop_ctr) is added in __main__ after target encodings.

def add_query_difficulty_features(df: pd.DataFrame) -> pd.DataFrame:
    df  = df.copy()
    grp = df.groupby("srch_id")
    df["query_size"]            = grp["srch_id"].transform("count")
    price_min                   = grp["price_usd"].transform("min")
    df["query_price_ratio"]     = grp["price_usd"].transform("max") / (price_min + 1e-6)
    df["query_price_std"]       = grp["price_usd"].transform("std").fillna(0)
    df["query_std_loc_score2"]  = grp["prop_location_score2"].transform("std").fillna(0)
    df["query_star_std"]        = grp["prop_starrating"].transform("std").fillna(0)
    return df


#TIEBREAKER FEATURES
#
# Used when hotels in a query are similar on prop_location_score2 (the primary signal).
# Auxiliary ratio/rank features that discriminate within homogeneous result sets.

def add_tiebreaker_features(df: pd.DataFrame) -> pd.DataFrame:
    df  = df.copy()
    eps = 1e-6

    df["price_per_star"]               = df["price_usd"] / (df["prop_starrating"] + 1)
    df["price_per_star_rank"]          = df.groupby("srch_id")["price_per_star"].rank(method="average")
    df["review_per_star"]              = df["prop_review_score"] / (df["prop_starrating"] + 1)
    df["review_per_star_rank"]         = df.groupby("srch_id")["review_per_star"].rank(method="average")
    df["prop_brand_rank_within_query"] = df.groupby("srch_id")["prop_brand_bool"].rank(method="average")
    df["loc1_loc2_ratio"]              = df["prop_location_score1"] / (df["prop_location_score2"] + eps)
    df["loc1_loc2_ratio_rank"]         = df.groupby("srch_id")["loc1_loc2_ratio"].rank(method="average")

    if "query_std_loc_score2" not in df.columns:
        df["query_std_loc_score2"] = (
            df.groupby("srch_id")["prop_location_score2"].transform("std").fillna(0)
        )
    threshold = df["query_std_loc_score2"].quantile(0.25)
    df["query_is_homogeneous_loc"] = (df["query_std_loc_score2"] < threshold).astype(int)

    return df


#DATE / CALENDAR FEATURES

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"]       = df["date_time"].dt.month
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["hour"]        = df["date_time"].dt.hour
    return df


#TRAIN / VALIDATION SPLIT  (group by srch_id so no search leaks)

def split_train_val(
    train: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(train, groups=train["srch_id"]))
    return train.iloc[train_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)


#POSITION-BIAS HANDLING [Wang, 2nd place ICDM 2013]
#
# 'position' is the strongest predictor of clicks but is unavailable at test
# time.  We keep it for reference but train only on random_bool=1 rows, or
# drop it before modelling.  Set UNBIASED_ONLY=True to filter.

UNBIASED_ONLY = False  # flip to True to train on random impressions only

def handle_position_bias(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()
    if is_train and UNBIASED_ONLY:
        df = df[df["random_bool"] == 1].reset_index(drop=True)
    # Always drop position - not available at inference
    df = df.drop(columns=["position"], errors="ignore")
    return df


#COMPETITOR AGGREGATES
#
# Collapse 24 sparse comp* columns into 3 signals [report §3]

def add_competitor_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["comp_count"]         = df[COMP_RATE_COLS].notna().sum(axis=1)
    df["comp_cheaper_count"] = (df[COMP_RATE_COLS] == -1).sum(axis=1)
    df["comp_diff_mean"]     = df[COMP_DIFF_COLS].mean(axis=1)
    return df


#MISSINGNESS FLAGS
#
# Absence is informative for distance, visitor history, and competitor blocks [report §3]

def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dist_missing"]         = df["orig_destination_distance"].isna().astype(int)
    df["visitor_hist_missing"] = df["visitor_hist_starrating"].isna().astype(int)
    df["comp_missing"]         = df[COMP_RATE_COLS].isna().all(axis=1).astype(int)
    return df


#PRICE LOG TRANSFORM
#
# Cap at 99.5th percentile then log(1+price) to reduce right skew [report §3]

def transform_price(df: pd.DataFrame, cap: float = None) -> pd.DataFrame:
    df = df.copy()
    if cap is not None:
        df["price_usd"] = df["price_usd"].clip(upper=cap)
    df["log_price"] = np.log1p(df["price_usd"])
    return df


#OUTLIER HANDLING
#
# Trees are robust to outliers, so we clip rather than remove.
# Thresholds are IQR-based and computed on training data only.

CLIP_COLS = ["price_usd", "orig_destination_distance", "gross_bookings_usd"]

def compute_clip_bounds(train: pd.DataFrame, cols: list[str], k: float = 3.0) -> dict:
    bounds = {}
    for col in cols:
        if col not in train.columns:
            continue
        q1, q3 = train[col].quantile(0.25), train[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - k * iqr, q3 + k * iqr)
    return bounds

def clip_outliers(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df

def report_outliers(df: pd.DataFrame, bounds: dict) -> None:
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        n = ((df[col] < lo) | (df[col] > hi)).sum()
        print(f"  {col}: {n:,} outliers clipped  (bounds [{lo:.2f}, {hi:.2f}])")


# MASTER PIPELINE

def prepare(df: pd.DataFrame, is_train: bool = True, clip_bounds: dict = None, price_cap: float = None) -> pd.DataFrame:
    df = add_date_features(df)
    df = add_missingness_flags(df)      # before imputation so NaNs are still present
    df = add_competitor_aggregates(df)  # before imputation so comp_count reflects actual coverage
    df = impute_missing(df)
    df = add_visitor_history_indicators(df)
    df = add_gap_features(df)
    df = transform_price(df, cap=price_cap)
    df = add_listwise_features(df)
    df = add_query_difficulty_features(df)
    df = add_tiebreaker_features(df)
    if clip_bounds:
        df = clip_outliers(df, clip_bounds)
    if is_train:
        df = add_target_columns(df)
        df = handle_position_bias(df, is_train=True)
    else:
        df = handle_position_bias(df, is_train=False)
    return df

if __name__ == "__main__":
    print("Loading data ")
    train_raw = load_data(TRAIN_PATH)
    test_raw  = load_data(TEST_PATH)

    print("Computing outlier bounds from training data ")
    clip_bounds = compute_clip_bounds(train_raw, CLIP_COLS)
    report_outliers(train_raw, clip_bounds)

    price_cap = train_raw["price_usd"].quantile(0.995)
    print(f"  price cap (99.5th pct): {price_cap:.2f}")

    print("Preparing training set ")
    train = prepare(train_raw, is_train=True, clip_bounds=clip_bounds, price_cap=price_cap)

    print("Preparing test set")
    test = prepare(test_raw, is_train=False, clip_bounds=clip_bounds, price_cap=price_cap)

    print("Splitting train / validation ")
    train_fold, val_fold = split_train_val(train)

    print("Adding target encodings")
    train_fold, val_fold = add_target_encodings(train_fold, val_fold)
    _,          test     = add_target_encodings(train_fold, test)

    print("Adding query_mean_prop_ctr (requires prop_ctr from target encodings)")
    for df in [train_fold, val_fold, test]:
        df["query_mean_prop_ctr"] = df.groupby("srch_id")["prop_ctr"].transform("mean")

    print(f"  train : {len(train_fold):,} rows")
    print(f"  val   : {len(val_fold):,} rows")
    print(f"  test  : {len(test):,} rows")
    print(f"  features: {list(train_fold.columns)}")

    print("Dataset creation")
    train_fold.to_parquet("prepared_train.parquet", index=False)
    val_fold.to_parquet("prepared_val.parquet",   index=False)
    test.to_parquet("prepared_test.parquet",       index=False)
    print("  saved prepared_train.parquet, prepared_val.parquet, prepared_test.parquet")

    print("Adding destination-season features")
    train_dm, val_dm = add_destination_season_features(train_fold, val_fold)
    _,        test_dm = add_destination_season_features(train_fold, test)

    dm_pairs = train_fold.groupby(["srch_destination_id", "month"]).ngroups
    lift_vals = train_dm["dest_month_lift"]
    strong_seasonal = (
        train_dm.groupby("srch_destination_id")["dest_month_lift"]
                .agg(lambda x: (x.max() > 1.5) or (x.min() < 0.7))
                .sum()
    )
    print(f"  Unique (dest, month) pairs in train: {dm_pairs:,}")
    print(f"  dest_month_lift  p10={lift_vals.quantile(0.10):.3f}"
          f"  p50={lift_vals.quantile(0.50):.3f}"
          f"  p90={lift_vals.quantile(0.90):.3f}"
          f"  min={lift_vals.min():.3f}  max={lift_vals.max():.3f}")
    print(f"  Destinations with strong seasonality (lift>1.5 or min<0.7): {strong_seasonal:,}")

    train_dm.to_parquet("prepared_train_dm.parquet", index=False)
    val_dm.to_parquet("prepared_val_dm.parquet",     index=False)
    test_dm.to_parquet("prepared_test_dm.parquet",   index=False)
    print("  saved prepared_train_dm.parquet, prepared_val_dm.parquet, prepared_test_dm.parquet")


