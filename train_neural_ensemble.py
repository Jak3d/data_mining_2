import os
import time
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

train = pd.read_parquet("prepared_train.parquet")
val   = pd.read_parquet("prepared_val.parquet")
test  = pd.read_parquet("prepared_test.parquet")

DROP_COLS    = ["srch_id", "date_time", "relevance", "score",
                "click_bool", "booking_bool", "gross_bookings_usd", "random_bool"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
GROUP_COL    = "srch_id"

SEEDS         = [42, 123, 777]
LABEL_CONFIGS = ["relevance", "booking_bool", "click_bool"]


def add_missingness_indicators(df, feature_cols):
    new_cols = []
    for col in feature_cols:
        if df[col].isna().any():
            indicator = f"{col}_missing"
            df[indicator] = df[col].isna().astype(np.float32)
            new_cols.append(indicator)
    return df, new_cols


train, miss_cols = add_missingness_indicators(train.copy(), FEATURE_COLS)
val,   _         = add_missingness_indicators(val.copy(),   FEATURE_COLS)
test,  _         = add_missingness_indicators(test.copy(),  FEATURE_COLS)

ALL_FEATURE_COLS = FEATURE_COLS + miss_cols

print(f"Train: {train[GROUP_COL].nunique():,} queries  {len(train):,} rows")
print(f"Val:   {val[GROUP_COL].nunique():,} queries  {len(val):,} rows")
print(f"Features: {len(ALL_FEATURE_COLS)} ({len(FEATURE_COLS)} base + {len(miss_cols)} missingness)")
print(f"Labels: {LABEL_CONFIGS}")
print(f"Seeds:  {SEEDS}")
print(f"Models: {len(LABEL_CONFIGS)} × {len(SEEDS)} = {len(LABEL_CONFIGS)*len(SEEDS)} total\n")

train_medians = train[FEATURE_COLS].median()


def prepare_X(df, medians):
    X = df[ALL_FEATURE_COLS].copy()
    X[FEATURE_COLS] = X[FEATURE_COLS].fillna(medians)
    return X.values.astype(np.float32)


X_train_raw = prepare_X(train, train_medians)
X_val_raw   = prepare_X(val,   train_medians)
X_test_raw  = prepare_X(test,  train_medians)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)


class RankDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ScoringNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.res_blocks(self.input_proj(x))).squeeze(1)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = (2 ** target - 1).clamp(min=1.0)
    return (weights * (pred - target) ** 2).mean()


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(mdl, X, df):
    mdl.eval()
    with torch.no_grad():
        preds = mdl(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    return df.assign(p=preds).groupby(GROUP_COL).apply(
        lambda g: ndcg_at_k(g["relevance"].values, g["p"].values)
    ).mean()


def rank_pct(df, col):
    return df.groupby(GROUP_COL)[col].rank(pct=True).values


label_val_acc  = {label: np.zeros(len(val))  for label in LABEL_CONFIGS}
label_test_acc = {label: np.zeros(len(test)) for label in LABEL_CONFIGS}

total_start = time.time()

for label_col in LABEL_CONFIGS:
    print(f"\n{'='*60}")
    print(f"Label: {label_col}")
    print(f"{'='*60}")

    y_train_label = train[label_col].values.astype(np.float32)

    for run_idx, seed in enumerate(SEEDS):
        print(f"\n  --- seed {seed} ({run_idx+1}/{len(SEEDS)}) ---")

        val_cache  = f"neural_{label_col}_seed{seed}_val.parquet"
        test_cache = f"neural_{label_col}_seed{seed}_test.parquet"

        if os.path.exists(val_cache) and os.path.exists(test_cache):
            print(f"  Loading cache: {val_cache}")
            vc = pd.read_parquet(val_cache).merge(val[["srch_id", "prop_id"]], on=["srch_id", "prop_id"])
            tc = pd.read_parquet(test_cache).merge(test[["srch_id", "prop_id"]], on=["srch_id", "prop_id"])
            label_val_acc[label_col]  += vc["rank_score"].values
            label_test_acc[label_col] += tc["rank_score"].values
            ndcg = val.assign(p=vc["rank_score"].values).groupby(GROUP_COL).apply(
                lambda g: ndcg_at_k(g["relevance"].values, g["p"].values)
            ).mean()
            print(f"  ndcg@5={ndcg:.6f}")
            continue

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(
            RankDataset(X_train, y_train_label), batch_size=4096,
            shuffle=True, num_workers=0, generator=g,
        )

        model     = ScoringNet(len(ALL_FEATURE_COLS)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        swa_model = AveragedModel(model)
        SWA_START = 20

        best_ndcg     = 0.0
        patience      = 10
        no_improve    = 0
        WARMUP_EPOCHS = 10
        n_batches     = len(train_loader)
        LOG_EVERY     = max(1, n_batches // 5)
        ckpt_path     = f"best_neural_{label_col}_seed{seed}.pt"

        run_start = time.time()

        for epoch in range(100):
            model.train()
            total_loss  = 0.0
            epoch_start = time.time()

            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                loss = weighted_mse(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if (batch_idx + 1) % LOG_EVERY == 0:
                    pct     = 100 * (batch_idx + 1) / n_batches
                    elapsed = time.time() - epoch_start
                    print(f"    [{epoch+1:3d}] {pct:5.1f}%  loss={total_loss/(batch_idx+1):.4f}  {elapsed:.0f}s")

            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            if epoch + 1 >= SWA_START:
                swa_model.update_parameters(model)

            val_ndcg   = evaluate(model, X_val, val.copy())
            epoch_time = time.time() - epoch_start
            improved   = val_ndcg > best_ndcg

            if improved:
                marker = " *"
            elif epoch + 1 <= WARMUP_EPOCHS:
                marker = f"  (warmup {epoch+1}/{WARMUP_EPOCHS})"
            else:
                marker = f"  (no improve {no_improve+1}/{patience})"

            print(f"  Epoch {epoch+1:3d}  loss={total_loss/n_batches:.4f}"
                  f"  val_NDCG@5={val_ndcg:.6f}  lr={lr:.2e}"
                  f"  {epoch_time:.0f}s{marker}")

            if improved:
                best_ndcg = val_ndcg
                torch.save(model.state_dict(), ckpt_path)
                no_improve = 0
            elif epoch + 1 > WARMUP_EPOCHS:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n  Early stopping at epoch {epoch+1}")
                    break

        swa_ndcg = evaluate(swa_model, X_val, val.copy())
        print(f"\n  best single: {best_ndcg:.6f}  SWA: {swa_ndcg:.6f}"
              f"  ({(time.time()-run_start)/60:.1f}m)")

        if swa_ndcg >= best_ndcg:
            final_model = swa_model
            print("  using SWA")
        else:
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            final_model = model
            print("  using best checkpoint")

        final_model.eval()
        with torch.no_grad():
            vp = final_model(torch.tensor(X_val,  dtype=torch.float32).to(DEVICE)).cpu().numpy()
            tp = final_model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()

        vr = rank_pct(val.assign(p=vp),  "p")
        tr = rank_pct(test.assign(p=tp), "p")

        label_val_acc[label_col]  += vr
        label_test_acc[label_col] += tr

        val[["srch_id", "prop_id"]].assign(rank_score=vr).to_parquet(val_cache,  index=False)
        test[["srch_id", "prop_id"]].assign(rank_score=tr).to_parquet(test_cache, index=False)
        print(f"  Saved {val_cache} / {test_cache}")

    label_val_acc[label_col]  /= len(SEEDS)
    label_test_acc[label_col] /= len(SEEDS)


def score_ensemble(labels):
    combined = sum(label_val_acc[l] for l in labels) / len(labels)
    return val.assign(p=combined).groupby(GROUP_COL).apply(
        lambda g: ndcg_at_k(g["relevance"].values, g["p"].values)
    ).mean()


total_time = time.time() - total_start

print(f"\n--- Per-label NDCG@5 ---")
for label_col in LABEL_CONFIGS:
    print(f"  {label_col:20s}  {score_ensemble([label_col]):.6f}")

print(f"\n--- Ablation: drop one label type ---")
for label_col in LABEL_CONFIGS:
    remaining = [l for l in LABEL_CONFIGS if l != label_col]
    print(f"  drop {label_col:16s}  {score_ensemble(remaining):.6f}")

full_ndcg = score_ensemble(LABEL_CONFIGS)
print(f"\n  all three:            {full_ndcg:.6f}  (total {total_time/60:.1f}m)")

val_acc  = sum(label_val_acc[l]  for l in LABEL_CONFIGS) / len(LABEL_CONFIGS)
test_acc = sum(label_test_acc[l] for l in LABEL_CONFIGS) / len(LABEL_CONFIGS)

val["pred_score"]  = val_acc
test["pred_score"] = test_acc

val[["srch_id",  "prop_id", "pred_score"]].to_csv("neural_val_scores.csv",  index=False)
test[["srch_id", "prop_id", "pred_score"]].to_csv("neural_test_scores.csv", index=False)
print("Exported neural_val_scores.csv and neural_test_scores.csv")
