

import torch
import torch.nn as nn
from Attention import Attention
from MLP import Mlp
from DropPath import DropPath


class Block(nn.Module):
    def __init__(self,
                 dim,  # 嵌入维度，token维度
                 num_heads,  # 头的数量
                 mlp_ratio=4.,  # MLP隐藏层维度和输入输出维度的比例
                 qkv_bias=False,  # 偏置，生成qkv时是否使用偏置
                 qk_scale=None,  # 缩放因子，如果是None则使用默认1/sqrt(head_dim)的缩放
                 drop_ratio=0.,  # 输出的dropout概率
                 attn_drop=0.,  # 注意力权重的dropout概率,防止过拟合，softmax(qk/sqrt(head_num))后使用
                 drop_path=0.,  # 随机深度的概率，随机跳过某一层
                 act_layer=nn.GELU,  # MLP的激活函数
                 norm_layer=nn.LayerNorm  # 归一化层
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)  # 第一个归一化层
        self.attn = Attention(  # 注意力层
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop_ratio
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # 随机深度
        self.norm2 = norm_layer(dim)  # 第二个归一化层
        mlp_hidden_dim = int(dim * mlp_ratio)  # MLP隐藏层维度
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)  # MLP层

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))  # 残差连接和注意力层
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 残差连接和MLP层
        return x
