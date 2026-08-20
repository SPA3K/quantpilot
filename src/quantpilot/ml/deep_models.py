"""
QuantPilot L2 - 预训练深度学习模型
使用Qlib官方预训练的GRU/LSTM checkpoint
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# 预训练模型路径
GRU_PATH = os.path.expanduser('~/workspace/quantpilot/references/qlib/examples/benchmarks/GRU/model_gru_csi300.pkl')
LSTM_PATH = os.path.expanduser('~/workspace/quantpilot/references/qlib/examples/benchmarks/LSTM/model_lstm_csi300.pkl')

# 自训练streaming模型路径
STREAMING_GRU_PATH = os.path.expanduser('~/workspace/quantpilot/models/trained/gru_streaming.pt')
STREAMING_LSTM_PATH = os.path.expanduser('~/workspace/quantpilot/models/trained/lstm_streaming.pt')


class GRUModel(nn.Module):
    """GRU模型 (匹配Qlib预训练结构)"""
    
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, d_feat)
        out, _ = self.rnn(x)
        out = self.fc_out(out[:, -1, :])  # 取最后一个时间步
        return out.squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM模型 (匹配Qlib预训练结构)"""
    
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, d_feat)
        out, _ = self.rnn(x)
        out = self.fc_out(out[:, -1, :])
        return out.squeeze(-1)


class PretrainedDeepModel:
    """
    预训练深度学习模型包装器
    使用Qlib官方checkpoint，适配我们的因子面板
    """
    
    def __init__(self, model_type='gru', device=None, seq_len=60):
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.model = None
        self._loaded = False
        
        # 特征映射：使用6个基础特征 (匹配Qlib Alpha360)
        self.feature_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap_dev']
    
    def load_model(self):
        """加载预训练模型"""
        if self._loaded:
            return
        
        if self.model_type == 'gru':
            self.model = GRUModel(d_feat=6, hidden_size=64, num_layers=2)
            weight_path = GRU_PATH
        elif self.model_type == 'lstm':
            self.model = LSTMModel(d_feat=6, hidden_size=64, num_layers=2)
            weight_path = LSTM_PATH
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # 加载预训练权重
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self._loaded = True
        logger.info(f"Loaded pre-trained {self.model_type} model on {self.device}")
    
    def _prepare_sequences(self, panel: pd.DataFrame, date: str) -> tuple:
        """
        准备序列数据 - 严格匹配Qlib Alpha360格式
        
        Qlib Alpha360布局（每只股票360个值，2D扁平）:
        [CLOSE59..CLOSE0, OPEN59..OPEN0, HIGH59..HIGH0, LOW59..LOW0, VWAP59..VWAP0, VOLUME59..VOLUME0]
        
        归一化: 
          CLOSE_i  = Ref($close, i) / $close
          OPEN_i   = Ref($open, i) / $close
          HIGH_i   = Ref($high, i) / $close
          LOW_i    = Ref($low, i) / $close
          VWAP_i   = Ref($vwap, i) / $close   ($vwap = amount/volume)
          VOLUME_i = Ref($volume, i) / $volume
        
        GRU输入: x = [N, 360] → reshape(N, 6, 60) → permute(0,2,1) → [N, 60, 6]
        
        返回: (codes, features) where features shape = (n_stocks, 360)  ← 2D扁平！
        """
        date = pd.Timestamp(date)
        
        # 获取历史数据（seq_len天）
        dates = panel[panel['date'] <= date]['date'].unique()
        dates = sorted(dates)[-self.seq_len:]
        
        if len(dates) < self.seq_len:
            logger.warning(f"Not enough history: {len(dates)} < {self.seq_len}")
            return None, None
        
        # 获取该日期的所有股票（如果指定日期不存在，用最近的交易日）
        target_data = panel[panel['date'] == date]
        if target_data.empty:
            last_date = panel['date'].max()
            target_data = panel[panel['date'] == last_date]
            if target_data.empty:
                return None, None
        codes = target_data['code'].unique()
        
        # 构建序列 — Qlib Alpha360格式
        sequences = []
        valid_codes = []
        
        for code in codes:
            stock_data = panel[panel['code'] == code].sort_values('date')
            stock_data = stock_data[stock_data['date'].isin(dates)]
            
            if len(stock_data) < self.seq_len:
                continue
            
            # 提取原始OHLCV
            close = stock_data['close'].values[-self.seq_len:]
            open_ = stock_data['open'].values[-self.seq_len:]
            high = stock_data['high'].values[-self.seq_len:]
            low = stock_data['low'].values[-self.seq_len:]
            volume = stock_data['volume'].values[-self.seq_len:]
            
            # VWAP = amount / volume（如果有amount列）
            if 'amount' in stock_data.columns:
                amount = stock_data['amount'].values[-self.seq_len:]
                vwap = amount / (volume + 1e-12)
            else:
                vwap = (high + low + close) / 3  # 近似
            
            # 处理NaN（copy确保可写）
            close = np.nan_to_num(close, nan=0.0)
            open_ = np.nan_to_num(open_, nan=0.0)
            high = np.nan_to_num(high, nan=0.0)
            low = np.nan_to_num(low, nan=0.0)
            volume = np.nan_to_num(volume, nan=0.0)
            vwap = np.nan_to_num(vwap, nan=0.0)
            
            current_close = close[-1]
            current_volume = volume[-1]
            
            if current_close <= 0 or current_volume <= 0:
                continue
            
            # Alpha360归一化: 除以当前close/volume
            # Qlib特征顺序: CLOSE×60, OPEN×60, HIGH×60, LOW×60, VWAP×60, VOLUME×60
            # 注意：是Ref($close, i)/$close，即 历史值/当前值，顺序从59到0
            feat_close = close / current_close      # [60]
            feat_open = open_ / current_close        # [60]
            feat_high = high / current_close          # [60]
            feat_low = low / current_close            # [60]
            feat_vwap = vwap / current_close          # [60]
            feat_volume = volume / current_volume     # [60]
            
            # 拼接为360维向量（Qlib顺序：从59到0，即时间倒序）
            # Qlib: Ref($close, 59)/$close, Ref($close, 58)/$close, ..., $close/$close
            # 我们的数据已经是正序（index 0 = 最早），需要翻转
            flat = np.concatenate([
                feat_close[::-1],   # CLOSE59..CLOSE0
                feat_open[::-1],    # OPEN59..OPEN0
                feat_high[::-1],    # HIGH59..HIGH0
                feat_low[::-1],     # LOW59..LOW0
                feat_vwap[::-1],    # VWAP59..VWAP0
                feat_volume[::-1],  # VOLUME59..VOLUME0
            ])  # shape: (360,)
            
            sequences.append(flat)
            valid_codes.append(code)
        
        if not sequences:
            return None, None
        
        # 返回2D张量 [N, 360]，和Qlib一致
        return valid_codes, torch.FloatTensor(np.array(sequences)).to(self.device)
    
    def predict(self, panel: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        预测：返回DataFrame(date, code, score)
        """
        from quantpilot.ml.model_zoo import BaseModel
        
        self.load_model()
        
        codes, features = self._prepare_sequences(panel, date)
        
        if codes is None:
            return pd.DataFrame(columns=['date', 'code', 'score'])
        
        # 推理 — _prepare_sequences返回[N, 360]，reshape为[N, 6, 60]再permute为[N, 60, 6]
        # 匹配Qlib的forward逻辑: x.reshape(N, 6, 60).permute(0, 2, 1)
        with torch.no_grad():
            features_3d = features.reshape(len(features), 6, 60).permute(0, 2, 1)  # [N, 60, 6]
            scores = self.model(features_3d).cpu().numpy()
        
        # 归一化到0-1
        score_min = scores.min()
        score_max = scores.max()
        if score_max > score_min:
            scores = (scores - score_min) / (score_max - score_min)
        else:
            scores = np.full_like(scores, 0.5)
        
        return pd.DataFrame({
            'date': date,
            'code': codes,
            'score': scores
        })


class StreamingGRU(nn.Module):
    """自训练GRU模型 (匹配streaming训练脚本结构)"""
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.d_feat = d_feat
        self.rnn = nn.GRU(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        # x: (batch, 360) → reshape to (batch, 6, 60) → permute to (batch, 60, 6)
        x = x.reshape(len(x), self.d_feat, -1).permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class StreamingLSTM(nn.Module):
    """自训练LSTM模型 (匹配streaming训练脚本结构)"""
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.d_feat = d_feat
        self.rnn = nn.LSTM(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1).permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── 注册到model_zoo ──────────────────────────────────────────────

from quantpilot.ml.model_zoo import BaseModel, register_model


class GRUPretrainedModel(BaseModel):
    """Qlib预训练GRU模型"""
    name = "gru_pretrained"
    category = "deep"
    description = "Qlib官方预训练GRU (CSI300, 2008-2020)"
    requires_training = False
    
    def __init__(self):
        self._model = PretrainedDeepModel(model_type='gru')
    
    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        return self._model.predict(factor_panel, date)


class LSTMPretrainedModel(BaseModel):
    """Qlib预训练LSTM模型"""
    name = "lstm_pretrained"
    category = "deep"
    description = "Qlib官方预训练LSTM (CSI300, 2008-2020)"
    requires_training = False
    
    def __init__(self):
        self._model = PretrainedDeepModel(model_type='lstm')
    
    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        return self._model.predict(factor_panel, date)


# 注册模型
register_model(GRUPretrainedModel())
register_model(LSTMPretrainedModel())


class StreamingDeepModel:
    """
    自训练streaming模型包装器
    使用我们训练的checkpoint（IC=0.2227），比Qlib预训练模型更好
    """
    def __init__(self, model_type='gru', device=None, seq_len=60):
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.model = None
        self._loaded = False

    def load_model(self):
        if self._loaded:
            return
        if self.model_type == 'gru':
            self.model = StreamingGRU(d_feat=6, hidden_size=64, num_layers=2)
            weight_path = STREAMING_GRU_PATH
        elif self.model_type == 'lstm':
            self.model = StreamingLSTM(d_feat=6, hidden_size=64, num_layers=2)
            weight_path = STREAMING_LSTM_PATH
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info(f"Loaded streaming {self.model_type} model on {self.device}")

    def predict(self, panel: pd.DataFrame, date: str) -> pd.DataFrame:
        """复用PretrainedDeepModel的_prepare_sequences逻辑"""
        self.load_model()
        proxy = PretrainedDeepModel(model_type=self.model_type, device=self.device, seq_len=self.seq_len)
        codes, features = proxy._prepare_sequences(panel, date)
        if codes is None:
            return pd.DataFrame(columns=['date', 'code', 'score'])

        with torch.no_grad():
            raw_scores = self.model(features).cpu().numpy()

        # z-score标准化：让分数在同一量级（mean=0, std=1）
        std = raw_scores.std()
        if std > 1e-8:
            normalized = (raw_scores - raw_scores.mean()) / std
        else:
            normalized = np.zeros_like(raw_scores)

        return pd.DataFrame({'date': date, 'code': codes, 'score': normalized})


class StreamingGRUModel(BaseModel):
    """自训练GRU模型 (IC=0.2227)"""
    name = "gru_streaming"
    category = "deep"
    description = "自训练GRU (6.8M窗口, 2008-2022, IC=0.2227)"
    requires_training = False

    def __init__(self):
        self._model = StreamingDeepModel(model_type='gru')

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        return self._model.predict(factor_panel, date)


class StreamingLSTMModel(BaseModel):
    """自训练LSTM模型 (IC=0.1895)"""
    name = "lstm_streaming"
    category = "deep"
    description = "自训练LSTM (6.8M窗口, 2008-2022, IC=0.1895)"
    requires_training = False

    def __init__(self):
        self._model = StreamingDeepModel(model_type='lstm')

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        return self._model.predict(factor_panel, date)


register_model(StreamingGRUModel())
register_model(StreamingLSTMModel())


# ── 测试 ──────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing pre-trained deep learning models...")
    
    # 测试模型加载
    gru = PretrainedDeepModel(model_type='gru')
    gru.load_model()
    print(f"✅ GRU loaded on {gru.device}")
    
    lstm = PretrainedDeepModel(model_type='lstm')
    lstm.load_model()
    print(f"✅ LSTM loaded on {lstm.device}")
    
    # 测试推理
    batch_size = 32
    seq_len = 20
    d_feat = 20
    
    x = torch.randn(batch_size, seq_len, d_feat).to(gru.device)
    
    with torch.no_grad():
        gru_out = gru.model(x)
        lstm_out = lstm.model(x)
    
    print(f"\n✅ GRU output shape: {gru_out.shape}")
    print(f"✅ LSTM output shape: {lstm_out.shape}")
    print(f"GRU sample scores: {gru_out[:5].cpu().numpy()}")
    print(f"LSTM sample scores: {lstm_out[:5].cpu().numpy()}")
