

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self,
                 dim = 768,  # 嵌入维度，token维度
                 num_heads=8,  # 头的数量
                 qkv_bias=False, # 偏置，生成qkv时是否使用偏置
                 qk_scale=None, # 缩放因子，如果是None则使用默认1/sqrt(head_dim)的缩放
                 attn_drop=0.,  # 注意力权重的dropout概率,防止过拟合，softmax(qk/sqrt(head_num))后使用
                 proj_drop=0.): # 输出投影的dropout概率

        super().__init__()

        self.num_heads = num_heads # 头的数量
        head_dim = dim // num_heads # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5 # qk缩放因子,d_k
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 生成qkv的线性层 ,Wq, Wk, Wv,写成三个是为了并行计算，参数更少
        self.attn_drop = nn.Dropout(attn_drop) # 注意力权重的dropout,在softmax后使用
        self.proj_drop = nn.Dropout(proj_drop) # 输出投影的dropout

        # 将每个head的输出拼接后通过contact连接，得到的数据线性层映射回原始维度
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B,N,C = x.shape # B: batch size, N: num_patch+1, token数量, C: 嵌入维度，就是dim  注：token数量 = patch数量 + 1 (class token)

        # B,N,C -> B,N,3*C -> B,N,3,num_heads,head_dim -> 3,B,num_heads,N,head_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 方便之后做运算
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别取出q,k,v,每个的维度都是 B,num_heads,N,head_dim

        #k.transpose(-2, -1)是将k的最后两个维度交换位置，B, num_heads, N, C//self.num_heads变成 B,num_heads,C//self.num_heads,N
        # q是 B,num_heads,N,C//self.num_heads, 和k的相乘是后两个维度的计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算注意力得分，维度是 B,num_heads,N,N
        attn = attn.softmax(dim=-1)  # 对最后一个维度做softmax，得到注意力权重，使得每行和为1
        x = (attn @ v)  # 将注意力权重和v相乘，得到加权后的值，维度是 B,num_heads,N,head_dim
        x = x.transpose(1, 2).reshape(B, N, C)  # 维度变换， B,num_heads,N,head_dim -> B,N,num_heads,head_dim -> B,N,C 回到了总的嵌入维度
        x = self.proj(x)  # 线性变换，映射回原始维度
        x = self.proj_drop(x)  # 输出投影的dropout

        return x




















