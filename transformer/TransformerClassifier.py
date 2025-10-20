

# TransformerClassifier.py
import torch
import torch.nn as nn
from .Encoder import Encoder

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, num_classes, pad_idx=0, dropout=0.1):
        super().__init__()
        # 可选：确保 d_model 可被 heads 整除
        assert d_model % heads == 0, "d_model 必须能被 heads 整除"
        self.pad_idx = pad_idx
        self.encoder = Encoder(vocab_size, d_model, N, heads, dropout)
        self.fc = nn.Linear(d_model, num_classes)


    def forward(self, input_ids, attention_mask=None):
        # 1) 基础 padding 掩码 (B, L)，True 表示有效 token
        if attention_mask is None:
            key_mask_2d = (input_ids != self.pad_idx)  # (B, L)
        else:
            key_mask_2d = attention_mask.bool()  # (B, L)

        # 2) 扩成 (B, L, L) 用于注意力分数的屏蔽（只屏蔽 key 维即可）
        B, L = input_ids.size()
        attn_mask_3d = key_mask_2d[:, None, :].expand(B, L, L)  # (B, L, L)

        # 3) 编码
        h = self.encoder(input_ids, attn_mask_3d)  # (B, L, d)

        # 4) masked mean pooling 仍用 2D 掩码
        denom = key_mask_2d.sum(1, keepdim=True).clamp(min=1)  # (B, 1)
        pooled = (h * key_mask_2d.unsqueeze(-1)).sum(1) / denom  # (B, d)

        # 5) 分类头
        return self.fc(pooled)  # (B, C)




