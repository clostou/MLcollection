# # # # # # # # # # # # # # # # # # # # # # # #
#
#    torch相关工具函数
#
# # # # # # # # # # # # # # # # # # # # # # # #

import os
import random
from typing import Any, Tuple
from collections import deque

import numpy as np
import torch
from torch import nn
try:
    import gym
except ImportError:
    _env_type = Any
    _has_gym = False
else:
    _env_type = gym.Env
    _has_gym = True
try:
    from thop import profile
except ImportError:
    _has_thop = False
else:
    _has_thop = True


__all__ = ['net_arch', 'all_seed', 'ReplayBuffer']


def net_arch(net: nn.Module, input_size: Tuple):
    """统计网络参数"""
    if not _has_thop:
        return
    X = torch.rand(size=input_size)
    flops, params = profile(net, inputs=(X,), verbose=False)
    print("[%s Profile]" % net.__class__.__name__, end=' ')
    # print("(0) Network input shape:\t", X.shape)
    # for i, layer in enumerate(net):
    #     X = layer(X)
    #     print("(%i)" % (i + 1), layer.__class__.__name__, "output shape:\t", X.shape)
    unit = lambda x: f'{x:.0f}' if x < 1e2 else f'{x / 1e3:.2f}K' if x < 1e5 \
        else f'{x / 1e6:.2f}M' if x < 1e8 else f'{x / 1e9:.2f}G'
    print("MACs: %s" % unit(flops), "Total params: %s" % unit(params), sep='; ')


def all_seed(env: _env_type = None, seed: int = 42):
    """万能的seed函数"""
    if seed == 0:
        return
    os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts
    np.random.seed(seed)
    random.seed(seed)
    if env is not None and _has_gym:
        env.reset(seed=seed)  # env config
    torch.manual_seed(seed)  # config for CPU
    torch.cuda.manual_seed(seed)  # config for GPU
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class ReplayBuffer:
    """
    实现经验回放
    """

    def __init__(self, capacity: int = None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        """存储transition到经验回放中"""
        self.buffer.append(transition)

    def sample(self, batch_size: int, ordered: bool = False):
        """从历史样本中采样"""
        if batch_size > len(self.buffer):  # 如果批量大小大于经验回放的容量，则取经验回放的容量
            batch_size = len(self.buffer)
        if ordered:  # 顺序采样
            start_ind = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(start_ind, start_ind + batch_size)]
            return zip(*batch)
        else:  # 随机采样
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)

    def sample_all(self):
        """采样所有样本"""
        batch = list(self.buffer)
        return zip(*batch)

    def clear(self):
        """清空经验回放"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


