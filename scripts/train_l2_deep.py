"""
QuantPilot L2 GRU/LSTM 微调脚本
数据: 2008-2022年A股日线 → Alpha360格式 → 微调Qlib预训练权重
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path.home() / 'workspace' / 'quantpilot' / 'data' / 'ml' / 'train_2008_2022'
MODEL_DIR = Path.home() / 'workspace' / 'quantpilot' / 'models' / 'trained'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 60
D_FEAT = 6
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-4


# ═══════════════ 数据准备 ═══════════════

def prepare_alpha360_panel():
    """加载数据并构建Alpha360格式的面板"""
    logger.info(f"加载数据: {DATA_DIR}")
    files = sorted(DATA_DIR.glob('*.parquet'))
    files = files[:1000]  # 限制文件数量避免OOM
    
    all_samples = []  # [(date, code, features_360, label)]
    
    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f)
            if len(df) < 80:  # 至少80天(60天序列+20天前瞻)
                continue
            
            df = df.sort_values('date').reset_index(drop=True)
            code = df['code'].iloc[0] if 'code' in df.columns else f.stem.split('_')[-1]
            
            close = df['close'].values
            open_ = df['open'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            amount = df['amount'].values if 'amount' in df.columns else close * volume
            
            # 标签: 未来20日收益的截面排名(后续处理)
            fwd_ret = np.full(len(close), np.nan)
            for j in range(len(close) - 20):
                fwd_ret[j] = close[j+20] / close[j] - 1
            
            # 滑动窗口构建Alpha360
            for j in range(SEQ_LEN, len(close) - 20):
                cc = close[j]; cv = volume[j]
                if cc <= 0 or cv <= 0:
                    continue
                
                # 过去60天的数据
                c_slice = close[j-SEQ_LEN:j] / cc
                o_slice = open_[j-SEQ_LEN:j] / cc
                h_slice = high[j-SEQ_LEN:j] / cc
                l_slice = low[j-SEQ_LEN:j] / cc
                vwap_slice = (amount[j-SEQ_LEN:j] / (volume[j-SEQ_LEN:j] + 1e-12)) / cc
                vol_slice = volume[j-SEQ_LEN:j] / (cv + 1e-12)
                
                # Alpha360格式: 360维扁平
                flat = np.concatenate([
                    c_slice[::-1], o_slice[::-1], h_slice[::-1],
                    l_slice[::-1], vwap_slice[::-1], vol_slice[::-1],
                ])
                
                all_samples.append({
                    'date': df['date'].iloc[j],
                    'code': code,
                    'features': flat.astype(np.float32),
                    'label': float(fwd_ret[j]) if not np.isnan(fwd_ret[j]) else None,
                })
        except Exception as e:
            pass
        
        if (i+1) % 500 == 0:
            logger.info(f"  已处理 {i+1}/{len(files)} 样本数:{len(all_samples)}")
    
    logger.info(f"  总样本: {len(all_samples)}")
    return all_samples


def build_dataset(samples):
    """构建PyTorch Dataset"""
    # 过滤无效样本
    valid = [s for s in samples if s['label'] is not None and not np.isnan(s['label'])]
    logger.info(f"  有效样本: {len(valid)}")
    
    # 按日期排序
    valid.sort(key=lambda x: x['date'])
    
    # 截面排名标签
    dates = [s['date'] for s in valid]
    labels = np.array([s['label'] for s in valid])
    
    # 按月分组排名
    df = pd.DataFrame({'date': dates, 'label': labels})
    df['ym'] = pd.to_datetime(df['date']).dt.to_period('M')
    df['rank_label'] = df.groupby('ym')['label'].transform(lambda x: x.rank(pct=True))
    
    features = np.array([s['features'] for s in valid])
    labels = df['rank_label'].values.astype(np.float32)
    
    return features, labels, dates


# ═══════════════ 模型定义 ═══════════════

class GRUModel(nn.Module):
    def __init__(self, d_feat=D_FEAT, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat
    
    def forward(self, x):
        # x: [N, 360] → reshape to [N, 6, 60] → permute to [N, 60, 6]
        x = x.reshape(len(x), self.d_feat, -1).permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, d_feat=D_FEAT, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat
    
    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1).permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze(-1)


# ═══════════════ 训练 ═══════════════

def train_model(model, X_train, y_train, X_valid, y_valid, model_name):
    """训练/微调模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 转为Dataset
    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    valid_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_valid), torch.FloatTensor(y_valid))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
    
    best_valid_loss = float('inf')
    best_state = None
    patience = 5
    no_improve = 0
    
    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0; n_batch = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item(); n_batch += 1
        
        # 验证
        model.eval()
        valid_loss = 0; n_valid = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                valid_loss += loss.item(); n_valid += 1
        
        train_loss /= n_batch
        valid_loss /= n_valid
        
        logger.info(f"  Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.6f} | Valid: {valid_loss:.6f}")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  早停 at epoch {epoch+1}")
                break
    
    # 加载最佳权重
    model.load_state_dict(best_state)
    
    # 保存
    model_path = MODEL_DIR / f'{model_name}_finetuned.pt'
    torch.save(best_state, model_path)
    logger.info(f"  模型已保存: {model_path}")
    
    return model


# ═══════════════ 主流程 ═══════════════

def main():
    # Step 1: 准备数据
    samples = prepare_alpha360_panel()
    features, labels, dates = build_dataset(samples)
    
    # Step 2: 时间分割
    dates_arr = pd.to_datetime(dates)
    train_mask = dates_arr < '2019-01-01'
    valid_mask = (dates_arr >= '2019-01-01') & (dates_arr < '2020-01-01')
    test_mask = dates_arr >= '2020-01-01'
    
    X_train, y_train = features[train_mask], labels[train_mask]
    X_valid, y_valid = features[valid_mask], labels[valid_mask]
    X_test, y_test = features[test_mask], labels[test_mask]
    
    logger.info(f"训练: {len(X_train)} | 验证: {len(X_valid)} | 测试: {len(X_test)}")
    
    # Step 3: 加载预训练权重并微调
    # GRU
    logger.info("\n=== 微调 GRU ===")
    gru = GRUModel()
    pretrained_path = Path.home() / 'workspace' / 'quantpilot' / 'references' / 'qlib' / 'examples' / 'benchmarks' / 'GRU' / 'model_gru_csi300.pkl'
    if pretrained_path.exists():
        import pickle
        with open(pretrained_path, 'rb') as f:
            pretrained = pickle.load(f)
        # 预训练权重可能是Qlib格式,尝试加载
        try:
            if isinstance(pretrained, dict) and 'state_dict' in pretrained:
                gru.load_state_dict(pretrained['state_dict'], strict=False)
            else:
                gru.load_state_dict(pretrained, strict=False)
            logger.info("  预训练权重已加载")
        except:
            logger.info("  预训练权重格式不匹配,从头训练")
    
    gru = train_model(gru, X_train, y_train, X_valid, y_valid, 'gru')
    
    # LSTM
    logger.info("\n=== 微调 LSTM ===")
    lstm = LSTMModel()
    pretrained_path = Path.home() / 'workspace' / 'quantpilot' / 'references' / 'qlib' / 'examples' / 'benchmarks' / 'LSTM' / 'model_lstm_csi300.pkl'
    if pretrained_path.exists():
        with open(pretrained_path, 'rb') as f:
            pretrained = pickle.load(f)
        try:
            if isinstance(pretrained, dict) and 'state_dict' in pretrained:
                lstm.load_state_dict(pretrained['state_dict'], strict=False)
            else:
                lstm.load_state_dict(pretrained, strict=False)
            logger.info("  预训练权重已加载")
        except:
            logger.info("  预训练权重格式不匹配,从头训练")
    
    lstm = train_model(lstm, X_train, y_train, X_valid, y_valid, 'lstm')
    
    # Step 4: 测试集评估
    from scipy.stats import spearmanr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for name, model in [('GRU', gru), ('LSTM', lstm)]:
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
        ic = spearmanr(pred, y_test)[0]
        logger.info(f"\n{name} 测试集 Rank-IC: {ic:.4f}")
    
    print("\nL2 微调完成!")


if __name__ == '__main__':
    main()
