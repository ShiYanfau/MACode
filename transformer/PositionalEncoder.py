



import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # 创建固定的 PE 矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:  # 避免 d_model 为奇数时越界
                    pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))

        pe = pe.unsqueeze(0)  # 形状 (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)  # 注册为 buffer，不参与梯度更新

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # 缩放 + 加上位置编码
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)




