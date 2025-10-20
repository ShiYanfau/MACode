from collections import OrderedDict

import torch
import torch.nn as nn
from Block import Block
from PatchEmbed import PatchEmbed
from functools import partial



class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, in_c=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None, representation_size=None,distilled=False,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None, emded_layer= PatchEmbed,act_layer=None):
        super(VisionTransformer,self).__init__()


        self.num_classes = num_classes  # 分类数量
        self.num_features = self.embed_dim = embed_dim  # 嵌入维度
        self.num_tokens = 2 if distilled else 1  # token数量，distilled为True时有两个token，class token和distill token，这个按下不表，有关模型蒸馏
        norm_layer = norm_layer or partial(nn.LayerNorm,eps=1e-6)  # 归一化层，默认是LayerNorm
        act_layer = act_layer or nn.GELU()  # 激活函数，默认是GELU
        self.patch_embed = emded_layer(  # Patch嵌入层
            img_size=image_size,  # 图像大小
            patch_size=patch_size,  # patch大小
            in_chans=in_c,  # 输入通道数
            embed_dim=embed_dim  # 嵌入维度
        )
        num_patches = self.patch_embed.num_patches  # 计算patch数量
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # class token，分类token （1，1，embed_dim）
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None  # distill token，蒸馏token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # 位置嵌入，（1，num_patches+num_tokens，embed_dim） （1，197，768）
        self.pos_drop = nn.Dropout(p=drop_ratio)  #
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # 随机深度的概率，从0到drop_path_ratio，长度为depth，每一层 encoder block 设置不同的随机深度概率 (Drop Path Rate)

        self.block = nn.Sequential(  # Transformer编码器块,打包了12个Block
            *[Block(
                dim=embed_dim,  # 嵌入维度
                num_heads=num_heads,  # 头的数量
                mlp_ratio=mlp_ratio,  # MLP隐藏层维度和输入输出维度的比例
                qkv_bias=qkv_bias,  # 生成qkv时是否使用偏置
                qk_scale=qk_scale,  # qk缩放因子
                drop_ratio=drop_ratio,  # 输出的dropout概率
                attn_drop=attn_drop_ratio,  # 注意力权重的dropout
                drop_path=dpr[i],  # 随机深度的概率
                norm_layer=norm_layer,  # 归一化层
                act_layer=act_layer  # 激活函数
            ) for i in range(depth)]
        )
        self.norm = norm_layer(embed_dim)  # 最后的归一化层


        #这一段代码用来决定是否在分类头（MLP Head）前面加一个额外的全连接层 + Tanh 激活，作为一个中间“表征层（representation layer）”。

        if representation_size and not distilled:  # 如果有表示层且不是蒸馏模型
            self.has_logits = True
            self.num_features = representation_size  # 表示层的维度
            self.pre_logits = nn.Sequential( OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),  # 线性层，embed_dim -> representation_size
                ('act', nn.Tanh())  # 激活函数，Tanh
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()  # 恒等映射,不做任何处理

        # 分类头
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()  # 分类头，num_classes > 0 则使用线性层，否则恒等映射
        self.head_dist = None
        if distilled:  # 如果是蒸馏模型
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()  # 蒸馏分类头

        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=.02)  # 位置嵌入初始化
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=.02)  # 蒸馏token初始化
        nn.init.trunc_normal_(self.cls_token, std=.02)  # class token初始化
        self.apply(_init_vit_weights)  # 应用权重初始化函数

    def forward_features(self, x):
        #从B C H W -> B num_patcher embed_dim
        x = self.patch_embed(x)  # 将图像分割成patch并嵌入，得到 (B, num_patches, embed_dim)
        #1,1,768->B,1,768
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # 扩展class token到 (B, 1, embed_dim)

        #如果有distill token，拼接class token和distill token，否则只拼接class token和输入patch的x
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # 拼接class token，得到 (B,197,768)
        else:
            dist_token = self.dist_token.expand(x.shape[0], -1, -1)  # 扩展distill token到 (B, 1, embed_dim)
            x = torch.cat((cls_token, dist_token, x), dim=1)  # 拼接class token和distill token，得到 (B, 198, 768)

        x = x + self.pos_embed  # 加上位置嵌入
        x = self.pos_drop(x)  # 位置嵌入的dropout
        x = self.block(x)  # 通过Transformer编码器块，得到 (B, num_patches+num_tokens, embed_dim)
        x = self.norm(x)  # 最后的归一化层

        if self.dist_token is None:  # 如果没有distill token

            return self.pre_logits(x[:, 0])  # 返回class token的表示，经过pre_logits层
        else:  # 如果有distill token
            return x[:, 0], x[:, 1]  # 返回class token和distill token的表示

    def forward(self, x):
        x = self.forward_features(x)  # 提取特征
        if self.head_dist is not None:  # 如果有蒸馏分类头
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # 分别通过分类头和蒸馏分类头
            if self.training and not torch.jit.is_scripting():  # 如果在训练模式
                return x, x_dist  # 返回两个分类结果
            else:  # 如果在评估模式
                return (x + x_dist) / 2  # 返回两个分类结果的平均值
        else:  # 如果没有蒸馏分类头
            x = self.head(x)  # 通过分类头
        return x  # 返回分类结果



def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes:int=1000, pretrained:bool=False):
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=None
    )
    return model








