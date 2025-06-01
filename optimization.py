# # # # # # # # # # # # # # # # # # # # # # # #
#
#    最优化算法模块
#
# # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from time import sleep

from .post import PlotAniOptim


__all__ = ['Optimization', 'GDM', 'Momentum', 'RMSProp', 'AdaDelta', 'Adam', 'Newton']


class Optimization:
    """
    基本优化类，使用一种或多种优化器计算函数极值
    """

    def __init__(self, optimizer, dy, d2y=None):
        if hasattr(optimizer, '__len__'):
            self.optimizer = list(optimizer)
        else:
            self.optimizer = [optimizer]
        self.n = len(self.optimizer)
        self.optimizer_tag = [opt.__class__.__name__ for opt in self.optimizer]
        self.grad_func = dy
        self.hass_func = d2y
        self.x_list = []
        self.print = False
        self.plot = False

    def start_print(self):
        self.print = True
        self._iter_i = 0
        print("\nOptimization Details:\n")

    def update_print(self):
        if not self.print:
            return
        x_sheet = np.array(self.x_list)
        x_sheet = x_sheet.squeeze(axis=x_sheet.ndim - 1)
        if self._iter_i % 10 == 0:
            print('iter  ' + ''.join(map(lambda s: '{:<12s}'.format(s), self.optimizer_tag)))
            print('-' * (self.n * 12 + 6))
        for i in range(x_sheet.shape[1]):
            if i == 0:
                print('{:<6d}'.format(self._iter_i), end='')
            else:
                print('      ', end='')
            print(''.join(map(lambda x: '{:<12.2e}'.format(x), x_sheet[: , i])))
        print()
        self._iter_i += 1

    def start_plot_2d(self, target_func, dim_1=0, dim_2=1, box_size=(10, 10)):
        self.plot = True
        self.target_func = target_func
        self._plot_dim_x = dim_1
        self._plot_dim_y = dim_2
        self._plot_margin_x = box_size[0] / 2
        self._plot_margin_y = box_size[1] / 2
        self._plotter = PlotAniOptim("优化算法可视化", (self.n, int(np.ceil(self.n / 2)), 2),
                                     self.optimizer_tag)

    def update_plot(self, i):
        if not self.plot:
            return
        def func_2d(x):
            _x = np.tile(self.x_list[i], x.shape[1])
            _x[self._plot_dim_x] = x[0]
            _x[self._plot_dim_y] = x[1]
            return self.target_func(_x)
        x_center = self.x_list[i][self._plot_dim_x, 0]
        y_center = self.x_list[i][self._plot_dim_y, 0]
        x_range = (x_center - self._plot_margin_x, x_center + self._plot_margin_x)
        y_range = (y_center - self._plot_margin_y, y_center + self._plot_margin_y)
        self._plotter.plotFunc(i, func_2d, x_range, y_range)
        self._plotter.addPoint(i, x_center, y_center)

    def _iter(self, i, precision):
        x = self.x_list[i]
        if self.optimizer[i].order == 1:
            dx = self.optimizer[i](self.grad_func(x))
        elif self.optimizer[i].order == 2:
            dx = self.optimizer[i](self.grad_func(x), self.hass_func(x))
        else:
            return True
        x += dx
        self.update_plot(i)
        return np.linalg.norm(dx) <= precision


    def run(self, init_x, steps=None, precision=1e-3, max_iter=1000, interval=0):
        for i in range(self.n):
            x = np.array(init_x, dtype=np.float32).reshape((-1, 1))
            self.x_list.append(x)
            self.update_plot(i)
        self.update_print()
        if steps:
            for _ in range(steps):
                for i in range(self.n):
                    self._iter(i, precision)
                self.update_print()
                sleep(interval)
            print("Iterations completed.\n")
        else:
            iter = 0
            convergence = np.zeros(self.n).astype(bool)
            while iter < max_iter and not convergence.all():
                for i in range(self.n):
                    if convergence[i]:
                        continue
                    convergence[i] = self._iter(i, precision)
                self.update_print()
                sleep(interval)
                iter += 1
            if iter < max_iter:
                print("Iterations converged.\n")
            else:
                print("Max iterations were reached.\n")
        if self.n == 1:
            return self.x_list[0]
        else:
            return self.x_list


# # # # # # # # # # # #  一阶优化算法  # # # # # # # # # # # #

class GDM:
    """
    梯度下降法 (Gradient Descent Method)
    """

    def __init__(self, alpha):
        self.order = 1
        self.alpha = alpha

    def __call__(self, grad):
        return - self.alpha * grad


class Momentum:
    """
    动量项梯度下降法 (Momentum)
    """

    def __init__(self, alpha, mu=0.9):
        self.order = 1
        self.alpha = alpha
        self.mu = mu
        self._last_momentum = 0.

    def __call__(self, grad):
        self.order = 1
        momentum = - self.alpha * grad + self.mu * self._last_momentum
        self._last_momentum = momentum
        return momentum


class RMSProp:
    """
    均方根传递算法 (Root Mean Square Propagation)
    """

    def __init__(self, alpha=0.001, delta=0.9, eps=1e-8):
        self.order = 1
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
        self.order = 1
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
        self.order = 1
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
        self.order = 2
        self.alpha = alpha

    def __call__(self, grad, hess):
        return - self.alpha * np.dot(np.linalg.inv(hess), grad)


def test_optimization():
    def func(x):
        return 0.1 * x[0] ** 2 + 2 * x[1] ** 2

    def grad(x):
        return np.vstack((0.2 * x[0], 4 * x[1]))

    optimizers = [Momentum(0.4, 0.5), RMSProp(0.4, 0.9), AdaDelta(0.9, 1e-1), Adam(1, 0.2, 0.9)]
    opt = Optimization(optimizers, grad)
    opt.start_print()
    opt.start_plot_2d(func, box_size=(6, 3))
    opt.run([-5, -2], precision=1e-2)


if __name__ == '__main__':
    test_optimization()


