
import torch
import torch.nn as nn



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features #如果没传输出维度，那么就是输入维度
        hidden_features = hidden_features or in_features #如果没传隐藏层维度，那么就是输入维度

        self.fc1 = nn.Linear(in_features, hidden_features) #第一个
        self.act = act_layer #激活函数
        self.fc2 = nn.Linear(hidden_features, out_features) #第二个
        self.drop = nn.Dropout(drop) #dropout

    def forward(self, x):
        x = self.fc1(x) #第一个线性层
        x = self.act(x) #激活函数
        x = self.drop(x) #dropout
        x = self.fc2(x) #第二个线性层
        x = self.drop(x) #dropout
        return x