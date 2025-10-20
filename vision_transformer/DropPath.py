


import torch
import torch.nn as nn



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths（随机深度）在残差块的主路径中应用时。
    这个实现类似于 DropConnect，用于 EfficientNet 等网络，但名字不同。
    DropConnect 是另一种形式的 dropout。
    详细讨论参见：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    我们使用 'drop path' 而不是 'DropConnect' 来避免混淆，并将参数名用 'survival rate' 来代替。

    参数:
    - x: 输入张量。
    - drop_prob: 丢弃路径的概率。
    - training: 是否处于训练模式。

    返回:
    - 如果不在训练模式或丢弃概率为 0，返回输入张量 x；
    - 否则，返回经过丢弃操作后的张量。
    """
    if drop_prob == 0. or not training:  # 如果丢弃率为 0 或不处于训练模式，直接返回原始输入
        return x
    keep_prob = 1 - drop_prob  # 保持路径的概率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 生成与 x 的维度匹配的形状，只保持 batch 维度
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 生成一个与 x 大小相同的随机张量
    random_tensor.floor_()  # 将随机张量二值化（小于 keep_prob 的值为 0，其他为 1）
    output = x.div(keep_prob) * random_tensor  # 将输入 x 缩放并与随机张量相乘，实现部分路径的丢弃
    return output  # 返回经过 drop path 操作后的张量



class DropPath(nn.Module):
    """
    Drop paths（随机深度）在残差块的主路径中应用时。
    这是一个 PyTorch 模块，用于在训练期间随机丢弃某些路径，以增强模型的泛化能力。
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()  # 调用父类 nn.Module 的构造函数
        self.drop_prob = drop_prob  # 初始化丢弃概率

    def forward(self, x):
        """
        前向传播函数，调用 drop_path 函数。

        参数:
        - x: 输入张量。

        返回:
        - 经过 drop path 操作后的张量。
        """
        return drop_path(x, self.drop_prob, self.training)  # 调用上面定义的 drop_path 函数
