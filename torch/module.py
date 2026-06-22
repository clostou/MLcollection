# # # # # # # # # # # # # # # # # # # # # # # #
#
#    自定义torch网络
#
# # # # # # # # # # # # # # # # # # # # # # # #

import math

import torch
from torch import nn
import torch.nn.functional as F

from typing import Any

__all__ = ['Swish', 'MLP', 'MLPSoftmax', 'MLPGauss', 'MLPReduce2D',
           'SelfAttention', 'Transformer']


class Swish(nn.Module):
    """
    Swish激活函数

    Swish(x) = x * Sigmoid(x)
    """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


class Residual(nn.Module):
    """
    残差连接块

    实际效果待验证
    Input Tensor: (..., Num_channel)
    Output Tensor: (..., Num_channel)
    """

    def __init__(self, num_channels: int):
        super(Residual, self).__init__()
        self.fc = nn.Linear(num_channels, num_channels)
        self.ln = nn.LayerNorm(num_channels, elementwise_affine=True)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc(self.ln(x))) + x  # 相比transformer的前馈网络，去掉了第二个线性变换，并使用Pre-Norm
        return x


class MLP(nn.Module):
    """
    全连接网络

    包含一个输入层、一个输出层、和多个等宽的隐藏层，其中输出层不包含激活函数
    支持残差连接，参见 `Residual`（效果存疑，默认不使用）
    Input Tensor: (..., Input_channel)
    Output Tensor: (..., Output_channel)
    """

    def __init__(self, input_channels: int, output_channels: int, num_channels: int,
                 hidden_layer_n: int = 3, skip_connect: bool = False, act_func: Any = nn.ReLU()):
        super(MLP, self).__init__()
        self.fc_in = nn.Linear(input_channels, num_channels)
        hidden = []
        if skip_connect:
            for _ in range(hidden_layer_n):
                hidden.append(Residual(num_channels))
        else:
            for _ in range(hidden_layer_n):
                hidden.extend([nn.Linear(num_channels, num_channels), act_func])
        self.hidden = nn.Sequential(*hidden)
        self.fc_out = nn.Linear(num_channels, output_channels)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc_in(x))
        x = self.hidden(x)
        return self.fc_out(x)


class MLPSoftmax(nn.Module):
    """
    全连接网络

    输出为一维概率分布，用于离散空间建模
    参见 `MLP`
    Input Tensor: (..., Input_channel)
    Output Tensor: (..., Output_channel)
    """

    def __init__(self, input_channels: int, output_channels: int, num_channels: int,
                 hidden_layer_n: int = 3, skip_connect: bool = False):
        super(MLPSoftmax, self).__init__()
        self.mlp = MLP(input_channels, output_channels, num_channels, hidden_layer_n, skip_connect)
        self.softmax = nn.Softmax(dim=-1)  # 使用Softmax层将输出转换为概率

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return self.softmax(x)


class MLPGauss(nn.Module):
    """
    全连接网络

    输出为多元高斯概率分布，用于连续空间建模
    参见 `MLP`
    Input Tensor: (..., Input_channel)
    Output Tensor: (..., Output_channel, [mu, sigma])
    """

    def __init__(self, input_channels: int, output_channels: int, num_channels: int,
                 hidden_layer_n: int = 3, skip_connect: bool = False):
        super(MLPGauss, self).__init__()
        self.mlp = MLP(input_channels, 2 * output_channels, num_channels, hidden_layer_n, skip_connect)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        x = torch.reshape(x, (*x.shape[: -1], -1, 2))
        return torch.stack([x[..., 0], self.softplus(x[..., 1])], dim=-1)


class MLPReduce2D(nn.Module):
    """
    用于特征提取的树状全连接网络

    输入二维张量 (batch, m, n)，输出一维张量 (batch, k)
    通常 m>>n ，中间层维度n'满足 n'>n 且 n'>k
    相比将二维输入展平后直接接入全连接层，参数量从 m*n*k 降至 m*n'+n'*k
    中间层使用了silu激活函数
    Input Tensor: (..., Input_sample, Input_channel)
    Output Tensor: (..., Output_channel)
    """

    def __init__(self, input_samples: int, input_channels: int, output_channels: int, hidden_channels: int = None):
        super(MLPReduce2D, self).__init__()
        if hidden_channels is None:
            hidden_channels = 2 * output_channels
        hidden_dim = math.ceil(hidden_channels / input_channels)  # 每个次级分支网络的输出通道数
        self.mlp_in = nn.ModuleList([nn.Linear(input_samples, hidden_dim) for _ in range(input_channels)])  # 次级分支网络
        self.mlp_out = nn.Linear(hidden_dim * input_channels, output_channels)  # 分支网络

    def forward(self, x: torch.Tensor):
        x = torch.stack([self.mlp_in[i](x[..., i]) for i in range(x.shape[-1])], dim=-1)
        return self.mlp_out(F.silu(x.flatten(start_dim=-2)))


class SelfAttention(nn.Module):
    """
    自注意力模块

    使用单头自注意力，不包含Wo线性变换
    Input Tensor: (Batch, Sample, Input_dim)
    Output Tensor: (Batch, Sample, Output_dim)
    """

    def __init__(self, input_channels: int, output_channels: int, query_channels: int = None):
        super(SelfAttention, self).__init__()
        if query_channels is None:
            query_channels = output_channels
        self.w_q = nn.Linear(input_channels, query_channels)
        self.w_k = nn.Linear(input_channels, query_channels)
        self.w_v = nn.Linear(input_channels, output_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_factor = 1.0 / query_channels ** 0.5

    def forward(self, x: torch.Tensor):
        query = self.w_q(x)  # 查询
        key = self.w_k(x)  # 键
        attn = self.softmax(torch.einsum('ijl,ikl->ijk', query, key) * self.attn_factor)  # 注意力打分
        value = self.w_v(x)  # 值
        return torch.matmul(attn, value)


class SelfAttentionGated(nn.Module):
    """
    门控自注意力模块
    详见：《Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free》
    论文连接：https://openreview.net/pdf?id=1b7whO4SfY

    使用单头自注意力，不包含Wo线性变换
    Input Tensor: (Batch, Sample, Input_dim)
    Output Tensor: (Batch, Sample, Output_dim)
    """

    def __init__(self, input_channels: int, output_channels: int, query_channels: int = None):
        super(SelfAttentionGated, self).__init__()
        if query_channels is None:
            query_channels = output_channels
        self.w_q = nn.Linear(input_channels, query_channels)
        self.w_k = nn.Linear(input_channels, query_channels)
        self.w_v = nn.Linear(input_channels, output_channels)
        self.w_g = nn.Linear(input_channels, output_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_factor = 1.0 / query_channels ** 0.5
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        query = self.w_q(x)  # 查询
        key = self.w_k(x)  # 键
        attn = self.softmax(torch.einsum('ijl,ikl->ijk', query, key) * self.attn_factor)  # 注意力打分
        value = self.w_v(x)  # 值
        head = torch.matmul(attn, value)  # 原始注意力计算结果
        gate = self.sigmoid(self.w_g(x))  # 门控分数
        return torch.mul(gate, head)


class Transformer(nn.Module):
    """
    简化的Transformer架构
    
    支持多头自注意力和通道数调整，去掉了层归一化、跳层连接和部分线性变换
    Input Tensor: (Batch, Sample, Input_dim)
    Output Tensor: (Batch, Sample, Output_dim)
    """

    def __init__(self, input_channels: int, output_channels: int,
                 head_num: int = 1, head_channels: int = None, gated: bool = False):
        super(Transformer, self).__init__()
        if head_channels is None:
            assert output_channels % head_num == 0, \
                "When 'head_channel' is not specified, 'output_channels' should be an integer multiple of " + \
                "'head_num', but (%d) and (%d) were given" % \
                (output_channels, head_num)
            head_channels = output_channels // head_num
        if gated:
            attention_module = SelfAttentionGated
        else:
            attention_module = SelfAttention
        self.attn = nn.ModuleList([attention_module(input_channels, head_channels) for _ in range(head_num)])
        self.mlp = nn.Linear(head_num * head_channels, output_channels)  # Wo矩阵

    def forward(self, x: torch.Tensor):
        x = torch.concatenate([self.attn[i](x) for i in range(len(self.attn))], dim=-1)
        return self.mlp(x)


