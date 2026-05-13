"""
prepare_external.py
───────────────────
Augments the prepared parquets with country-level external features and
saves prepared_train_ext.parquet / prepared_val_ext.parquet /
prepared_test_ext.parquet.

Country mapping inferred from data patterns (see analysis below):
  219 → United States    (99% confidence — 10× largest, 95% domestic rate)
  100 → United Kingdom   (75% — 2nd-largest English market, $129 median)
  220 → Singapore        (60% — 67.5% domestic, $386 mean, 4.28★)
  216 → Germany          (60% — visitor/prop ratio 3.56: world's biggest outbound)
   55 → Australia        (55% — 51.3% domestic, high prices, English-speaking)
  129 → Japan            (50% — 3.76★ mean, 1.35 v/p ratio)

External sources (2012 values, reflecting dataset time window late-2012/mid-2013):
  GDP per capita:       World Bank WDI indicator NY.GDP.PCAP.CD (2012, current USD)
  Tourist arrivals:     UNWTO Tourism Highlights 2013 Edition (millions, 2012)
  English / developed:  hardcoded from standard classifications
"""

import pandas as pd
import numpy as np

RAW_TRAIN = "training_set_VU_DM.csv"
SMOOTHING = 10.0   # for LOO country booking-rate encoding

# ── EXTERNAL LOOKUP (World Bank 2012 + UNWTO 2012) ───────────────────────────
# Keys are prop_country_id integers; confidence noted in comments.
COUNTRY_META = {
    219: dict(gdp_2012=51749, tourism_2012=66.7, is_english=1, is_developed=1),  # US   99%
    100: dict(gdp_2012=41681, tourism_2012=29.3, is_english=1, is_developed=1),  # UK   75%
    220: dict(gdp_2012=54578, tourism_2012=14.5, is_english=1, is_developed=1),  # SGP  60%
    216: dict(gdp_2012=43741, tourism_2012=30.4, is_english=0, is_developed=1),  # DEU  60%
     55: dict(gdp_2012=67458, tourism_2012= 6.0, is_english=1, is_developed=1),  # AUS  55%
    129: dict(gdp_2012=46667, tourism_2012= 8.4, is_english=0, is_developed=1),  # JPN  50%
}

MAPPED_IDS = set(COUNTRY_META)

# Fallback medians for unmapped countries (computed from mapped set)
_gdp_vals     = [v["gdp_2012"]     for v in COUNTRY_META.values()]
_tour_vals    = [v["tourism_2012"] for v in COUNTRY_META.values()]
GDP_MEDIAN    = float(np.median(_gdp_vals))
TOURISM_MED   = float(np.median(_tour_vals))

# ── STEP 1: COUNTRY-LEVEL STATS FROM RAW TRAINING DATA ───────────────────────
print("Loading raw CSV for country-level aggregates ...")
raw = pd.read_csv(
    RAW_TRAIN, na_values="NULL",
    usecols=["srch_id", "prop_country_id", "price_usd",
             "prop_starrating", "prop_review_score", "booking_bool"],
)

print(f"  Raw rows: {len(raw):,}   unique country IDs: {raw['prop_country_id'].nunique()}")

grp = raw.groupby("prop_country_id")

country_stats = pd.DataFrame({
    "n_rows":       grp["srch_id"].count(),
    "median_price": grp["price_usd"].median(),
    "mean_stars":   grp["prop_starrating"].mean(),
    "mean_review":  grp["prop_review_score"].mean(),
    "book_sum":     grp["booking_bool"].sum(),
    "book_count":   grp["booking_bool"].count(),
})
country_stats["n_rows_log"] = np.log1p(country_stats["n_rows"])

# LOO booking rate (avoids leakage for the large training set)
global_book_rate = raw["booking_bool"].mean()
country_stats["booking_rate_enc"] = (
    (country_stats["book_sum"] + SMOOTHING * global_book_rate)
    / (country_stats["book_count"] + SMOOTHING)
)

# ── STEP 2: BUILD FULL FEATURE FRAME ─────────────────────────────────────────
ext = country_stats[["n_rows_log", "median_price", "mean_stars",
                      "mean_review", "booking_rate_enc"]].copy()
ext.columns = [f"country_{c}" for c in ext.columns]

ext["country_gdp_2012"]     = ext.index.map(
    lambda cid: COUNTRY_META[cid]["gdp_2012"]     if cid in COUNTRY_META else GDP_MEDIAN)
ext["country_tourism_2012"] = ext.index.map(
    lambda cid: COUNTRY_META[cid]["tourism_2012"] if cid in COUNTRY_META else TOURISM_MED)
ext["country_is_english"]   = ext.index.map(
    lambda cid: COUNTRY_META[cid]["is_english"]   if cid in COUNTRY_META else 0)
ext["country_is_developed"] = ext.index.map(
    lambda cid: COUNTRY_META[cid]["is_developed"] if cid in COUNTRY_META else 0)
ext["country_is_us"]        = (ext.index == 219).astype(int)
ext["country_mapped"]       = ext.index.isin(MAPPED_IDS).astype(int)

# Log-transform skewed economic variables
ext["country_gdp_2012_log"]     = np.log1p(ext["country_gdp_2012"])
ext["country_tourism_2012_log"] = np.log1p(ext["country_tourism_2012"])

print(f"\nCountry feature table ({len(ext)} countries):")
print(ext.loc[sorted(MAPPED_IDS)].to_string(float_format="{:.3f}".format))
print(f"\nFallback GDP median:     {GDP_MEDIAN:.0f}")
print(f"Fallback tourism median: {TOURISM_MED:.1f}M arrivals")

# ── STEP 3: MERGE INTO PREPARED PARQUETS ─────────────────────────────────────
import os, sys

for split in ("train", "val", "test"):
    src = f"prepared_{split}.parquet"
    dst = f"prepared_{split}_ext.parquet"
    if not os.path.exists(src):
        print(f"  [skip] {src} not found")
        continue

    df = pd.read_parquet(src)
    n_before = len(df)

    # reset_index on ext so prop_country_id becomes a column for merge
    df = df.merge(ext.reset_index().rename(columns={"prop_country_id": "prop_country_id"}),
                  on="prop_country_id", how="left")

    assert len(df) == n_before, "Row count changed — merge error"
    n_unmapped = (df["country_mapped"] == 0).sum()
    print(f"  {split}: {n_before:,} rows  "
          f"mapped={n_before - n_unmapped:,}  "
          f"unmapped={n_unmapped:,} ({100*n_unmapped/n_before:.1f}%)")

    df.to_parquet(dst, index=False)
    print(f"  → saved {dst}")

print("\nNew feature columns added:")
new_cols = [c for c in ext.columns]
print("  " + "  ".join(new_cols))
