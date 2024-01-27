# # # # # # # # # # # # # # # # # # # # # # # #
#
#    最优化算法模块
#
# # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from time import sleep

from post import PlotAniOptim


__all__ = ['GDM', 'Momentum', 'RMSProp', 'AdaDelta', 'Adam', 'Newton']


class Optimization:
    """
    示例代码：优化算法展示与对比
    """

    def __init__(self, optimizer, init_x, precision=1e-3):
        self.n = len(optimizer)
        self.optimizer = optimizer
        self.precision = precision
        self.x_list = []
        self.plotter = PlotAniOptim("优化算法可视化", (self.n, int(np.ceil(self.n / 2)), 2),
                                    [opt.__class__.__name__ for opt in optimizer])
        self.plotter.plotFunc(self._func, (-8, 2), (-3, 3))
        for i in range(self.n):
            x = np.array(init_x, dtype=np.float64)
            self.x_list.append(x)
            self.plotter.addPoint(i, *x)

    def _func(self, x, y):
        return 0.1 * x ** 2 + 2 * y ** 2

    def _grad(self, x, y):
        return np.array([0.2 * x, 4 * y])

    def _iter(self, i):
        x = self.x_list[i]
        dx = self.optimizer[i](self._grad(*x))
        x += dx
        self.plotter.addPoint(i, *x)
        return np.linalg.norm(dx) <= self.precision

    def run(self, steps=None, interval=0.01):
        if steps:
            for _ in range(steps):
                for i in range(self.n):
                    self._iter(i)
                sleep(interval)
        else:
            convergence = np.zeros(self.n).astype(bool)
            while not convergence.all():
                for i in range(self.n):
                    if convergence[i]: continue
                    convergence[i] = self._iter(i)
                sleep(interval)


# # # # # # # # # # # #  一阶优化算法  # # # # # # # # # # # #

class GDM:
    """
    梯度下降法 (Gradient Descent Method)
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, grad):
        return - self.alpha * grad


class Momentum:
    """
    动量项梯度下降法 (Momentum)
    """

    def __init__(self, alpha, mu=0.9):
        self.alpha = alpha
        self.mu = mu
        self._last_momentum = 0.

    def __call__(self, grad):
        momentum = - self.alpha * grad + self.mu * self._last_momentum
        self._last_momentum = momentum
        return momentum


class RMSProp:
    """
    均方根传递算法 (Root Mean Square Propagation)
    """

    def __init__(self, alpha=0.001, delta=0.9, eps=1e-8):
        self.alpha = alpha
        self.delta = delta
        self.eps = eps
        self._rms_g = None

    def __call__(self, grad):
        if isinstance(self._rms_g, type(None)):
            self._rms_g = np.zeros(grad.shape)
        self._rms_g = self.delta * self._rms_g + (1 - self.delta) * grad ** 2
        return - self.alpha * grad / np.sqrt(self._rms_g + self.eps)


class AdaDelta:
    """
    自适应学习率算法 (Adadelta, an extension of Adagrad)
    """

    def __init__(self, delta=0.95, eps=1e-6):
        self.delta = delta
        self.eps = eps
        self._rms_g = None
        self._rms_dx = None
        self._last_dx = None

    def __call__(self, grad):
        if isinstance(self._rms_g, type(None)):
            self._rms_g = np.zeros(grad.shape)
            self._rms_dx = np.zeros(grad.shape)
        else:
            self._rms_dx = self.delta * self._rms_dx + (1 - self.delta) * self._last_dx ** 2
        self._rms_g = self.delta * self._rms_g + (1 - self.delta) * grad ** 2
        dx = - np.sqrt(self._rms_dx + self.eps) * grad / np.sqrt(self._rms_g + self.eps)
        self._last_dx = dx
        return dx


class Adam:
    """
    自适应矩估计算法 (Adaptive Moment Estimation)
    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = alpha    # 全局学习率: 0.0001 ~ 0.1
        self.beta1 = beta1    # 动量项衰减系数（惯性）: 0.8 ~ 0.99
        self.beta2 = beta2    # 梯度历史衰减系数（阻尼）: 0.95 ~ 0.9999
        self.eps = eps
        self._m = None
        self._v = None
        self._k = 0

    def __call__(self, grad):
        if isinstance(self._m, type(None)):
            self._m = np.zeros(grad.shape)
            self._v = np.zeros(grad.shape)
        self._m = self.beta1 * self._m + (1 - self.beta1) * grad    # 移动指数加权平均
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad ** 2
        self._k += 1
        alpha = self.alpha * np.sqrt(1 - self.beta2 ** self._k) / (1 - self.beta1 ** self._k)
        return - alpha * self._m / (np.sqrt(self._v) + self.eps)


# # # # # # # # # # # #  二阶优化算法（包括拟牛顿法）  # # # # # # # # # # # #

class Newton:
    """
    牛顿法 (Newton Method)
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, grad, hess):
        return - self.alpha * np.dot(np.linalg.inv(hess), grad)


if __name__ == '__main__':
    optimizers = [Momentum(0.4, 0.2), RMSProp(0.4, 0.9), AdaDelta(0.9, 1e-2), Adam(1, 0.2, 0.9)]
    opt = Optimization(optimizers, (-5, -2))
    opt.run()


