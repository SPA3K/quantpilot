#!/usr/bin/env python3
"""
单股 L2 GRU 训练脚本（通用版）
数据源: 腾讯行情API（港股+A股通用）
用途: 与 L1 LightGBM 并行训练，作为深度学习对照模型
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, pickle, urllib.request, datetime
from pathlib import Path

MODEL_DIR = Path.home() / "workspace" / "quantpilot" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def get_kline(code, days=800):
    url = f"https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?param={code},day,,,{days},qfq"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
    node = data["data"][code]
    kl = node.get("qfqday") or node.get("day")
    return [{"date": r[0], "open": float(r[1]), "close": float(r[2]),
             "high": float(r[3]), "low": float(r[4]),
             "volume": float(r[5]) if len(r) > 5 else 0} for r in kl]

def build_sequences(kline, seq_len=30):
    """构建时序样本: 过去seq_len天的OHLCV → 未来1天方向"""
    import numpy as np
    closes = [k["close"] for k in kline]
    opens = [k["open"] for k in kline]
    highs = [k["high"] for k in kline]
    lows = [k["low"] for k in kline]
    vols = [k["volume"] for k in kline]
    # 归一化
    c = np.array(closes)
    norm = lambda x: np.array(x) / c[-1]
    X_list, y_list = [], []
    for i in range(seq_len, len(closes) - 1):
        seq = np.stack([
            norm(opens[i-seq_len:i]), norm(closes[i-seq_len:i]),
            norm(highs[i-seq_len:i]), norm(lows[i-seq_len:i]),
            norm(vols[i-seq_len:i]) if vols[i-seq_len:i] else np.zeros(seq_len)
        ], axis=1)  # (seq_len, 5)
        X_list.append(seq)
        y_list.append(1 if closes[i+1] > closes[i] else 0)
    return np.array(X_list), np.array(y_list)

def main():
    import torch
    import torch.nn as nn
    import numpy as np

    if len(sys.argv) < 2:
        print("用法: python train_l2_single.py <code>（如 sh600519 / hk00700）")
        sys.exit(1)
    code = sys.argv[1]
    print(f"拉取 {code} K线...")
    kline = get_kline(code, 800)
    print(f"获取 {len(kline)} 根")

    X, y = build_sequences(kline, seq_len=30)
    print(f"样本: {len(X)}")

    # 时序切分
    n = len(X); t1 = int(n * 0.7); t2 = int(n * 0.85)
    X_train, y_train = torch.FloatTensor(X[:t1]), torch.FloatTensor(y[:t1])
    X_val, y_val = torch.FloatTensor(X[t1:t2]), torch.FloatTensor(y[t1:t2])
    X_test, y_test = torch.FloatTensor(X[t2:]), torch.FloatTensor(y[t2:])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    class GRUModel(nn.Module):
        def __init__(self, input_dim=5, hidden_dim=32, num_layers=2):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
            self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x):
            out, _ = self.gru(x)
            return torch.sigmoid(self.fc(out[:, -1])).squeeze()

    model = GRUModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_val_acc = 0
    for epoch in range(20):
        model.train()
        idx = torch.randperm(len(X_train))
        total_loss = 0
        for i in range(0, len(idx), 64):
            batch = idx[i:i+64]
            pred = model(X_train[batch].to(device))
            loss = criterion(pred, y_train[batch].to(device))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        # 验证
        model.eval()
        with torch.no_grad():
            vpred = model(X_val.to(device))
            vacc = ((vpred > 0.5).float() == y_val.to(device)).float().mean().item()
        if vacc > best_val_acc:
            best_val_acc = vacc
            torch.save(model.state_dict(), MODEL_DIR / f"gru_single_{code}.pt")
        if (epoch+1) % 5 == 0:
            print(f"  epoch {epoch+1}/20 loss={total_loss/(len(X_train)//64):.4f} val_acc={vacc:.2%}")

    # 测试
    model.load_state_dict(torch.load(MODEL_DIR / f"gru_single_{code}.pt"))
    model.eval()
    with torch.no_grad():
        tpred = model(X_test.to(device))
        tacc = ((tpred > 0.5).float() == y_test.to(device)).float().mean().item()
        conf = (tpred > 0.55) | (tpred < 0.45)
        if conf.sum() > 0:
            cacc = ((tpred[conf] > 0.5).float() == y_test.to(device)[conf]).float().mean().item()
            print(f"测试集: 整体={tacc:.2%}  高信心({conf.sum()}样本)={cacc:.2%}")
        else:
            print(f"测试集: 整体={tacc:.2%}  无高信心信号")
    print(f"最佳验证acc: {best_val_acc:.2%}")

if __name__ == "__main__":
    main()
