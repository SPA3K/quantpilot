#!/usr/bin/env python3
"""
L2 GRU/LSTM — TRUE Streaming Training
Read one parquet file at a time, build Alpha360 windows, train, discard.
Memory peak: ~50MB per file (never loads full dataset).

Strategy:
  Pass 0: Compute normalization stats (running mean/std across all files)
  Pass 1+: For each epoch, iterate files one by one:
           read → build windows → normalize → train mini-batches → discard
"""
import os, sys, gc, warnings
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import torch
import torch.nn as nn

BASE = Path("/home/spark/workspace/quantpilot")
sys.path.insert(0, str(BASE / "src"))

# ─── Settings ───
SEQ_LEN = 60        # Alpha360 lookback
D_FEAT = 6          # close, open, high, low, vwap, volume
FORWARD = 20        # forward return days
BATCH_SIZE = 2048
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU = torch.cuda.is_available()

DATA_DIR = BASE / "data" / "ml" / "train_2008_2022"
MODEL_DIR = BASE / "models" / "trained"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(DATA_DIR.glob("*.parquet"))
print(f"Found {len(files)} parquet files")

# ─── Build Alpha360 windows from one file ───
def build_windows_from_file(filepath):
    """Read one parquet, build Alpha360 windows.
    Returns (X[N, 360], y[N]) or None.
    X is NOT normalized yet (raw ratios)."""
    df = pd.read_parquet(filepath)
    if len(df) < SEQ_LEN + FORWARD + 10:
        return None
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    amount = df["amount"].values.astype(np.float64) if "amount" in df.columns else close * volume

    X_list = []
    y_list = []
    for j in range(SEQ_LEN, len(close) - FORWARD):
        cc = close[j]
        cv = volume[j]
        if cc <= 0 or cv <= 0:
            continue

        # Alpha360: 6 features × 60 steps, normalized by current close/volume
        c_slice = close[j-SEQ_LEN:j] / cc
        o_slice = open_[j-SEQ_LEN:j] / cc
        h_slice = high[j-SEQ_LEN:j] / cc
        l_slice = low[j-SEQ_LEN:j] / cc
        vwap_slice = (amount[j-SEQ_LEN:j] / (volume[j-SEQ_LEN:j] + 1e-12)) / cc
        vol_slice = volume[j-SEQ_LEN:j] / (cv + 1e-12)

        flat = np.concatenate([
            c_slice[::-1], o_slice[::-1], h_slice[::-1],
            l_slice[::-1], vwap_slice[::-1], vol_slice[::-1],
        ]).astype(np.float32)

        fwd = close[j + FORWARD] / cc - 1
        if np.isnan(fwd) or np.isinf(fwd):
            continue

        X_list.append(flat)
        y_list.append(fwd)

    if len(X_list) < 10:
        return None

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ─── Pass 0: Compute normalization stats (streaming) ───
print("\nPass 0: Computing normalization stats (streaming)...")
n_total = 0
running_mean = np.zeros(360, dtype=np.float64)
running_M2 = np.zeros(360, dtype=np.float64)

for i, f in enumerate(files):
    result = build_windows_from_file(f)
    if result is None:
        continue
    X, _ = result
    for row in X:
        if np.isnan(row).any() or np.isinf(row).any():
            continue
        n_total += 1
        delta = row - running_mean
        running_mean += delta / n_total
        delta2 = row - running_mean
        running_M2 += delta * delta2
    if (i + 1) % 1000 == 0:
        print(f"  Scanned {i+1}/{len(files)}, {n_total:,} rows...")

running_std = np.sqrt(running_M2 / max(n_total - 1, 1))
running_std[running_std < 1e-8] = 1.0
mean_np = running_mean.astype(np.float32)
std_np = running_std.astype(np.float32)
print(f"  Total windows: {n_total:,}")
print(f"  Mean range: [{mean_np.min():.4f}, {mean_np.max():.4f}]")
print(f"  Std range:  [{std_np.min():.4f}, {std_np.max():.4f}]")

# ─── Models (Alpha360 format: flat 360 → reshape to [N, 6, 60] → permute to [N, 60, 6]) ───
class GRUModel(nn.Module):
    def __init__(self, d_feat=D_FEAT, hidden=64, layers=2, dropout=0.0):
        super().__init__()
        self.d_feat = d_feat
        self.rnn = nn.GRU(d_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1).permute(0, 2, 1)  # [N, 60, 6]
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, d_feat=D_FEAT, hidden=64, layers=2, dropout=0.0):
        super().__init__()
        self.d_feat = d_feat
        self.rnn = nn.LSTM(d_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1).permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ─── Training: stream files, accumulate batches, train ───
def train_epoch(model, optimizer, criterion, epoch, train_files, val_files):
    model.train()
    total_loss = 0.0
    n_batches = 0
    batch_buf_X = []
    batch_buf_y = []

    for i, f in enumerate(train_files):
        result = build_windows_from_file(f)
        if result is None:
            continue
        X, y = result
        # Normalize
        X = (X - mean_np) / std_np
        # Filter NaN/Inf rows
        valid = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y) | np.isinf(y))
        X, y = X[valid], y[valid]
        if len(X) == 0:
            continue
        batch_buf_X.append(X)
        batch_buf_y.append(y)

        # Flush when buffer is big enough
        total_samples = sum(len(b) for b in batch_buf_X)
        if total_samples >= BATCH_SIZE * 4:
            X_all = np.concatenate(batch_buf_X)
            y_all = np.concatenate(batch_buf_y)
            for j in range(0, len(X_all), BATCH_SIZE):
                xb = torch.from_numpy(X_all[j:j+BATCH_SIZE]).to(DEVICE)
                yb = torch.from_numpy(y_all[j:j+BATCH_SIZE]).to(DEVICE)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            batch_buf_X = []
            batch_buf_y = []
            del X_all, y_all

        if (i + 1) % 500 == 0:
            avg = total_loss / max(n_batches, 1)
            print(f"    Epoch {epoch+1} train: {i+1}/{len(train_files)} files, loss={avg:.4f}")
            gc.collect()

    # Flush remaining
    if batch_buf_X:
        X_all = np.concatenate(batch_buf_X)
        y_all = np.concatenate(batch_buf_y)
        for j in range(0, len(X_all), BATCH_SIZE):
            xb = torch.from_numpy(X_all[j:j+BATCH_SIZE]).to(DEVICE)
            yb = torch.from_numpy(y_all[j:j+BATCH_SIZE]).to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    # Validate (streaming)
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for f in val_files:
            result = build_windows_from_file(f)
            if result is None:
                continue
            X, y = result
            X = (X - mean_np) / std_np
            valid = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y) | np.isinf(y))
            X, y = X[valid], y[valid]
            if len(X) == 0:
                continue
            for j in range(0, len(X), BATCH_SIZE):
                xb = torch.from_numpy(X[j:j+BATCH_SIZE]).to(DEVICE)
                pred = model(xb).cpu().numpy()
                val_preds.append(pred)
                val_trues.append(y[j:j+BATCH_SIZE])

    val_ic = 0.0
    if val_preds:
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_ic = spearmanr(val_preds, val_trues)[0]

    return total_loss / max(n_batches, 1), val_ic


# ─── Main ───
print(f"\nDevice: {DEVICE}")
if USE_GPU:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Train/val split by file index (80/20)
np.random.seed(42)
indices = np.random.permutation(len(files))
split = int(len(indices) * 0.8)
train_files = [files[i] for i in indices[:split]]
val_files = [files[i] for i in indices[split:]]
print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

results = {}
for model_name, ModelClass in [("gru", GRUModel), ("lstm", LSTMModel)]:
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} ({EPOCHS} epochs, streaming)...")
    model = ModelClass().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_ic = -1.0
    best_path = MODEL_DIR / f"{model_name}_streaming.pt"

    for epoch in range(EPOCHS):
        avg_loss, val_ic = train_epoch(model, optimizer, criterion, epoch, train_files, val_files)
        scheduler.step()
        status = "✓ BEST" if val_ic > best_ic else ""
        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} val_ic={val_ic:.4f} {status}")
        if val_ic > best_ic:
            best_ic = val_ic
            torch.save(model.state_dict(), best_path)
        gc.collect()
        if USE_GPU:
            torch.cuda.empty_cache()

    results[model_name] = {"ic": best_ic, "path": str(best_path)}
    del model, optimizer; gc.collect()
    if USE_GPU:
        torch.cuda.empty_cache()

# ─── Summary ───
print("\n" + "="*60)
print("L2 Deep Learning (Streaming) — Complete")
print(f"  Total files:    {len(files)}")
print(f"  Total windows:  {n_total:,}")
print(f"  Sequence len:   {SEQ_LEN}")
print(f"  GRU best IC:    {results['gru']['ic']:.4f} → {results['gru']['path']}")
print(f"  LSTM best IC:   {results['lstm']['ic']:.4f} → {results['lstm']['path']}")
print(f"  L1 baseline:    0.1292")
