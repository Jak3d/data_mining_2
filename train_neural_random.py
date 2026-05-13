import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

train_full = pd.read_parquet("prepared_train.parquet")
val        = pd.read_parquet("prepared_val.parquet")
test       = pd.read_parquet("prepared_test.parquet")

train = train_full[train_full["random_bool"] == 1].copy().reset_index(drop=True)
print(f"Random-only train: {len(train):,} rows  {train['srch_id'].nunique():,} queries"
      f"  ({100*len(train)/len(train_full):.1f}% of full train)")

DROP_COLS    = ["srch_id", "date_time", "relevance", "score",
                "click_bool", "booking_bool", "gross_bookings_usd", "random_bool"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
GROUP_COL    = "srch_id"


def add_missingness_indicators(df, feature_cols):
    new_cols = []
    for col in feature_cols:
        if df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype(np.float32)
            new_cols.append(f"{col}_missing")
    return df, new_cols


train, miss_cols = add_missingness_indicators(train.copy(), FEATURE_COLS)
val,   _         = add_missingness_indicators(val.copy(),   FEATURE_COLS)
test,  _         = add_missingness_indicators(test.copy(),  FEATURE_COLS)
ALL_FEATURE_COLS = FEATURE_COLS + miss_cols

train_medians = train[FEATURE_COLS].median()


def prepare_X(df, medians):
    X = df[ALL_FEATURE_COLS].copy()
    X[FEATURE_COLS] = X[FEATURE_COLS].fillna(medians)
    return X.values.astype(np.float32)


scaler  = StandardScaler()
X_train = scaler.fit_transform(prepare_X(train, train_medians))
X_val   = scaler.transform(prepare_X(val,   train_medians))
X_test  = scaler.transform(prepare_X(test,  train_medians))
y_train = train["relevance"].values.astype(np.float32)


class RankDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.block(x))


class ScoringNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(n, 256), nn.LayerNorm(256), nn.GELU())
        self.res_blocks = nn.Sequential(ResidualBlock(256), ResidualBlock(256))
        self.head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x): return self.head(self.res_blocks(self.input_proj(x))).squeeze(1)


def weighted_mse(pred, target):
    return ((2 ** target - 1).clamp(min=1.0) * (pred - target) ** 2).mean()


def ndcg_at_k(relevance, scores, k=5):
    order = np.argsort(scores)[::-1][:k]
    gains = 2 ** relevance[order] - 1
    dcg   = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(2 ** relevance - 1)[::-1][:k]
    idcg  = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(mdl, X, df):
    mdl.eval()
    with torch.no_grad():
        p = mdl(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    return df.assign(p=p).groupby(GROUP_COL).apply(
        lambda g: ndcg_at_k(g["relevance"].values, g["p"].values)
    ).mean()


torch.manual_seed(42)
train_loader = DataLoader(RankDataset(X_train, y_train), batch_size=2048, shuffle=True, num_workers=0)
model        = ScoringNet(len(ALL_FEATURE_COLS)).to(DEVICE)
optimizer    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler    = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

best_ndcg, patience, no_improve, WARMUP = 0.0, 10, 0, 10
n_batches = len(train_loader)
start     = time.time()

for epoch in range(100):
    model.train()
    total_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = weighted_mse(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    val_ndcg = evaluate(model, X_val, val.copy())
    improved  = val_ndcg > best_ndcg

    marker = " *" if improved else (f"  (warmup {epoch+1}/{WARMUP})" if epoch+1 <= WARMUP
                                    else f"  (no improve {no_improve+1}/{patience})")
    print(f"Epoch {epoch+1:3d}  loss={total_loss/n_batches:.4f}"
          f"  val_NDCG@5={val_ndcg:.6f}  {time.time()-start:.0f}s{marker}")

    if improved:
        best_ndcg = val_ndcg
        torch.save(model.state_dict(), "best_neural_random.pt")
        no_improve = 0
    elif epoch + 1 > WARMUP:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(torch.load("best_neural_random.pt", map_location=DEVICE))
model.eval()
with torch.no_grad():
    vp = model(torch.tensor(X_val,  dtype=torch.float32).to(DEVICE)).cpu().numpy()
    tp = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()

def rank_pct(df, col):
    return df.groupby(GROUP_COL)[col].rank(pct=True).values

vr = rank_pct(val.assign(p=vp),  "p")
tr = rank_pct(test.assign(p=tp), "p")

val[["srch_id",  "prop_id"]].assign(pred_score=vr).to_csv("neural_random_val_scores.csv",  index=False)
test[["srch_id", "prop_id"]].assign(pred_score=tr).to_csv("neural_random_test_scores.csv", index=False)
print(f"\nBest val NDCG@5: {best_ndcg:.6f}")
print("Exported neural_random_val_scores.csv and neural_random_test_scores.csv")
