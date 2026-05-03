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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train.copy(), test.copy()

    encoding_specs = [
        (["prop_id"],                          "click_bool",   "prop_ctr"),
        (["prop_id"],                          "booking_bool", "prop_cvr"),
        (["srch_destination_id"],              "booking_bool", "dest_cvr"),
        (["prop_id", "srch_booking_window"],   "booking_bool", "prop_cvr_bw"),
    ]

    for keys, target, name in encoding_specs:
        enc = build_target_encodings(train, keys, target)
        train[name] = train.set_index(keys).index.map(enc.get)
        test[name]  = test.set_index(keys).index.map(enc.get)
        global_mean = train[target].mean()
        train[name] = train[name].fillna(global_mean)
        test[name]  = test[name].fillna(global_mean)

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


# MASTER PIPELINE

def prepare(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = add_date_features(df)
    df = impute_missing(df)
    df = add_visitor_history_indicators(df)
    df = add_gap_features(df)
    df = add_listwise_features(df)
    if is_train:
        df = add_target_columns(df)
        df = handle_position_bias(df, is_train=True)
    else:
        df = handle_position_bias(df, is_train=False)
    return df

if __name__ == "__main__":
    print("Loading data...")
    train_raw = load_data(TRAIN_PATH)
    test_raw  = load_data(TEST_PATH)

    print("Preparing training set...")
    train = prepare(train_raw, is_train=True)

    print("Preparing test set...")
    test = prepare(test_raw, is_train=False)

    print("Adding target encodings...")
    train, test = add_target_encodings(train, test)

    print("Splitting train / validation...")
    train_fold, val_fold = split_train_val(train)

    print(f"  train : {len(train_fold):,} rows")
    print(f"  val   : {len(val_fold):,} rows")
    print(f"  test  : {len(test):,} rows")
    print(f"  features: {list(train_fold.columns)}")


