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


#WEIGHTED TARGET CONSTRUCTION [Liu et al. & Wang, ICDM 2013]
#
# Relevance: booking=5, click=1, none=0  (for LambdaMART / NDCG optimisation)
# Score:     booking + 0.05 * click      (for regression / pointwise models)

def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["relevance"] = df["booking_bool"] * 5 + df["click_bool"] * 1
    df["score"]     = df["booking_bool"] + 0.05 * df["click_bool"]
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

    print(f"  train : {len(train_fold):,} rows")
    print(f"  val   : {len(val_fold):,} rows")
    print(f"  test  : {len(test):,} rows")
    print(f"  features: {list(train_fold.columns)}")

    print("Dataset creation")
    train_fold.to_parquet("prepared_train.parquet", index=False)
    val_fold.to_parquet("prepared_val.parquet",   index=False)
    test.to_parquet("prepared_test.parquet",       index=False)
    print("  saved prepared_train.parquet, prepared_val.parquet, prepared_test.parquet")


