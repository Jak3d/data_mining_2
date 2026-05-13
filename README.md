# VU Data Mining 2026 : Execution Guide

## Pipeline overview

```
prepare_data.py
     │
     ├── train_lgbm_ensemble.py  → lightgbm_{val,test}_scores.csv
     ├── train_lgbm_aug.py      → lightgbm_aug_{val,test}_scores.csv
     ├── train_lgbm_dm.py       → lightgbm_dm_{val,test}_scores.csv
     ├── train_neural_ensemble.py → neural_{val,test}_scores.csv
     ├── train_xgb.py           → xgboost_{val,test}_scores.csv
     ├── train_catboost.py      → catboost_{val,test}_scores.csv
     └── tmp_dart.py            → dart_{val,test}_scores.csv
          │
          └── stack.py          → submission.csv
```

## Step 1 : Data preparation

```
python prepare_data.py
```

Reads `training_set_VU_DM.csv` and `test_set_VU_DM.csv`. Applies the full feature engineering pipeline, then writes six parquet files.

### Feature engineering pipeline

| Step | What it does |
|---|---|
| Date features | Extracts `month`, `day_of_week`, `hour` from `date_time` |
| Missingness flags | Binary indicators for `orig_destination_distance`, `visitor_hist_*`, and competitor block missingness |
| Competitor aggregates | Collapses 24 sparse `comp*` columns into `comp_count`, `comp_cheaper_count`, `comp_diff_mean` |
| Imputation | Competitor rate/inv/diff → 0; `prop_location_score2` and `srch_query_affinity_score` → column min |
| Visitor history indicators | Match/mismatch flags for star rating and price vs visitor history (handles ~95% missingness) |
| Gap features | `ump` (unexplained price vs historical), `per_fee`, `total_fee`, `score1d2`, `score2ma`, `price_diff`, `starrating_diff` |
| Price transform | Clip at 99.5th percentile, then `log1p` |
| Listwise normalisation | Per-query rank, demean, and z-score for price, stars, location scores, review score |
| Query difficulty | `query_size`, `query_price_ratio`, `query_price_std`, `query_std_loc_score2`, `query_star_std` |
| Tiebreaker features | `price_per_star`, `review_per_star`, `loc1_loc2_ratio` and their within-query ranks; `query_is_homogeneous_loc` |
| Outlier clipping | IQR-based bounds (k=3) computed on train only, applied to `price_usd`, `orig_destination_distance`, `gross_bookings_usd` |
| Target columns | `relevance = booking*5 + click*1`, `score = booking + 0.05*click` |
| Train/val split | 80/20 group split by `srch_id` (no search leaks across folds) |
| Target encodings (LOO) | `prop_ctr`, `prop_cvr`, `dest_cvr`, `prop_cvr_bw` — leave-one-out on train, smoothed lookup on val/test |
| Query mean CTR | `query_mean_prop_ctr` added after target encodings |
| Destination-season features | `dest_month_cvr` (LOO), `dest_month_lift`, `dest_month_price_ratio`, `dest_month_n_searches_log`, `is_peak_month_for_dest` — written to `_dm` parquets only |

### Output parquets

| File | Used by |
|---|---|
| `prepared_train.parquet` | all models except `train_lgbm_dm.py` |
| `prepared_val.parquet` | all models except `train_lgbm_dm.py` |
| `prepared_test.parquet` | all models except `train_lgbm_dm.py` |
| `prepared_train_dm.parquet` | `train_lgbm_dm.py` only |
| `prepared_val_dm.parquet` | `train_lgbm_dm.py` only |
| `prepared_test_dm.parquet` | `train_lgbm_dm.py` only |

## Step 2 : Train models (independent, run in any order)

### LightGBM ensemble (lambdarank, 3 labels × 3 seeds = 9 models)
```
python train_lgbm_ensemble.py
```
Labels: `relevance`, `booking_bool`, `click_bool`. Seeds: `42`, `123`, `777`. Rank-normalises and averages across all 9 models.

### LightGBM with query-subsampling augmentation (lambdarank, 1 seed, K=4 copies)
```
python train_lgbm_aug.py
```

### LightGBM with destination-season features (lambdarank, 1 seed)
```
python train_lgbm_dm.py
```
Requires `prepared_*_dm.parquet` from `prepare_data.py`.

### Neural network ensemble (residual MLP, SWA, 3 labels × 3 seeds = 9 models)
```
python train_neural_ensemble.py
```
Requires a CUDA GPU. Trains `relevance`, `booking_bool`, `click_bool` labels across seeds `42`, `123`, `777`. Architecture: 256-dim residual MLP with LayerNorm, weighted MSE loss, SWA from epoch 20, early stopping patience=10. Caches per-(label, seed) rank-normalized scores as parquets so reruns are instant.

### XGBoost (rank:ndcg, 3 labels × 3 seeds = 9 models)
```
python train_xgb.py
```
Auto-detects GPU. Caches per-(label, seed) predictions as parquets so reruns are instant.

Labels: `relevance`, `booking_bool`, `click_bool`  
Seeds: `42`, `123`, `2024`

### CatBoost YetiRank (GPU, 3 seeds)
```
python train_catboost.py
```
Requires GPU : aborts if none found. Trains 1505 iterations per seed. Caches per-seed parquets.

Seeds: `42`, `123`, `2024`

### LightGBM DART (3 labels × 3 seeds = 9 models)
```
python tmp_dart.py
```
No early stopping (incompatible with DART dropout). Fixed 2000 rounds. Caches per-(label, seed) parquets.

## Step 3 : Stack and generate submission

```
python stack.py
```

- Loads whichever `*_val_scores.csv` / `*_test_scores.csv` files exist (missing models are skipped)
- Rank-normalises per query within each model
- Reports individual model NDCG@5 and equal-weight average
- Fits optimal blend weights via differential evolution (softmax-parametrised, maximises NDCG@5 on val)
- Runs an overfit check: fits weights on val first-half, evaluates on val second-half
- Writes `submission.csv`

## Output files

| File | Written by |
|---|---|
| `lightgbm_val_scores.csv` / `_test_scores.csv` | `train_lgbm_ensemble.py` |
| `lightgbm_aug_val_scores.csv` / `_test_scores.csv` | `train_lgbm_aug.py` |
| `lightgbm_dm_val_scores.csv` / `_test_scores.csv` | `train_lgbm_dm.py` |
| `neural_val_scores.csv` / `_test_scores.csv` | `train_neural_ensemble.py` |
| `xgboost_val_scores.csv` / `_test_scores.csv` | `train_xgb.py` |
| `catboost_val_scores.csv` / `_test_scores.csv` | `train_catboost.py` |
| `dart_val_scores.csv` / `_test_scores.csv` | `tmp_dart.py` |
| `submission.csv` | `stack.py` |

## Known val NDCG@5 baselines

| Model | Val NDCG@5 |
|---|---|
| Single LightGBM | 0.385 |
| Single neural | 0.387 |
| Single XGBoost (9-model) | 0.394 |
| 5-model equal-weight ensemble | 0.3948 |
| 5-model stacked ensemble | 0.3954 |
| Val–leaderboard gap | ~0.0002 |
