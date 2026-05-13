import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.preprocessing import QuantileTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

train = pd.read_parquet("prepared_train.parquet")
val   = pd.read_parquet("prepared_val.parquet")
test  = pd.read_parquet("prepared_test.parquet")

DROP_COLS    = ["srch_id", "date_time", "relevance", "score",
                "click_bool", "booking_bool", "gross_bookings_usd"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
GROUP_COL    = "srch_id"

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

print(f"Train groups: {train[GROUP_COL].nunique():,}  rows: {len(train):,}")
print(f"Val   groups: {val[GROUP_COL].nunique():,}  rows: {len(val):,}")
print(f"Test  groups: {test[GROUP_COL].nunique():,}  rows: {len(test):,}")
print(f"Features: {len(FEATURE_COLS)} base + {len(miss_cols)} missingness indicators = {len(ALL_FEATURE_COLS)} total")
print()

train_medians = train[FEATURE_COLS].median()

def prepare_X(df, medians):
    X = df[ALL_FEATURE_COLS].copy()
    X[FEATURE_COLS] = X[FEATURE_COLS].fillna(medians)
    return X.values.astype(np.float32)

X_train_raw = prepare_X(train, train_medians)
X_val_raw   = prepare_X(val,   train_medians)
X_test_raw  = prepare_X(test,  train_medians)

n_base = len(FEATURE_COLS)
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                         subsample=200_000, random_state=42)
print("Fitting QuantileTransformer...")
X_train = np.hstack([qt.fit_transform(X_train_raw[:, :n_base]), X_train_raw[:, n_base:]])
X_val   = np.hstack([qt.transform(X_val_raw[:, :n_base]),       X_val_raw[:, n_base:]])
X_test  = np.hstack([qt.transform(X_test_raw[:, :n_base]),      X_test_raw[:, n_base:]])
print("Done.\n")

y_train = train["relevance"].values.astype(np.float32)

class RankDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(RankDataset(X_train, y_train), batch_size=4096, shuffle=True,  num_workers=0)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class RankNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
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

model     = RankNet(len(ALL_FEATURE_COLS)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
swa_model = AveragedModel(model)
SWA_START = 20

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
    df = df.copy()
    df["pred"] = preds
    return df.groupby(GROUP_COL).apply(
        lambda g: ndcg_at_k(g["relevance"].values, g["pred"].values)
    ).mean()

best_ndcg     = 0.0
patience      = 10
no_improve    = 0
WARMUP_EPOCHS = 10
n_batches     = len(train_loader)
LOG_EVERY     = max(1, n_batches // 5)

print(f"Training: {n_batches} batches/epoch  (batch_size=4096)  SWA from epoch {SWA_START}")
print(f"Warmup: {WARMUP_EPOCHS} epochs  Patience: {patience}  Max epochs: 100")
print("-" * 70)

train_start = time.time()

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
            print(f"  [{epoch+1:3d}] {pct:5.1f}%  batch {batch_idx+1}/{n_batches}"
                  f"  loss={total_loss/(batch_idx+1):.4f}  {elapsed:.0f}s")

    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]

    if epoch + 1 >= SWA_START:
        swa_model.update_parameters(model)

    val_ndcg   = evaluate(model, X_val, val.copy())
    epoch_time = time.time() - epoch_start
    total_time = time.time() - train_start
    improved   = val_ndcg > best_ndcg

    if improved:
        marker = " *"
    elif epoch + 1 <= WARMUP_EPOCHS:
        marker = f"  (warmup {epoch+1}/{WARMUP_EPOCHS})"
    else:
        marker = f"  (no improvement {no_improve+1}/{patience})"

    print(f"Epoch {epoch+1:3d}  loss={total_loss/n_batches:.4f}"
          f"  val_NDCG@5={val_ndcg:.6f}  lr={lr:.2e}"
          f"  {epoch_time:.0f}s/epoch  total={total_time/60:.1f}m{marker}")

    if improved:
        best_ndcg = val_ndcg
        torch.save(model.state_dict(), "best_neural.pt")
        no_improve = 0
    elif epoch + 1 > WARMUP_EPOCHS:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}  (no improvement for {patience} epochs)")
            break

total_time = time.time() - train_start
print(f"\nBest single-model val NDCG@5: {best_ndcg:.6f}  (training took {total_time/60:.1f}m)")

print("Updating SWA BatchNorm stats...")
update_bn(train_loader, swa_model, device=DEVICE)
swa_ndcg = evaluate(swa_model, X_val, val.copy())
print(f"SWA model val NDCG@5:         {swa_ndcg:.6f}")

if swa_ndcg > best_ndcg:
    print("Using SWA model for submission.")
    final_model = swa_model
else:
    print("Using best single-epoch checkpoint for submission.")
    model.load_state_dict(torch.load("best_neural.pt", map_location=DEVICE))
    final_model = model

final_model.eval()
with torch.no_grad():
    val_preds  = final_model(torch.tensor(X_val,  dtype=torch.float32).to(DEVICE)).cpu().numpy()
    test_preds = final_model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()

val["pred_score"]  = val_preds
test["pred_score"] = test_preds

final_ndcg = val.groupby(GROUP_COL).apply(
    lambda g: ndcg_at_k(g["relevance"].values, g["pred_score"].values)
).mean()
print(f"Final val NDCG@5: {final_ndcg:.6f}")

val[["srch_id",  "prop_id", "pred_score"]].to_csv("neural_val_scores.csv",  index=False)
test[["srch_id", "prop_id", "pred_score"]].to_csv("neural_test_scores.csv", index=False)

submission = (
    test[["srch_id", "prop_id", "pred_score"]]
    .sort_values(["srch_id", "pred_score"], ascending=[True, False])
    .drop(columns="pred_score")
    .reset_index(drop=True)
)
submission.to_csv("submission.csv", index=False)
print(f"submission.csv  ({len(submission):,} rows, {submission['srch_id'].nunique():,} queries)")