# # # # # # # # # # # # # # # # # # # # # # # #
#
#    回归模块
#
# # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from optimization import Optimization, Newton
import matplotlib.pyplot as plt

from pre import read
import post


__all__ = ['Logit', 'RLS']


class Logit:
    """
    分类 - 对数几率回归

    学习核心：
        数值优化算法 | 牛顿法、(随机)梯度下降法 等
    """

    def __init__(self, data, label):
        self.m = data.shape[1]
        self.n = data.shape[0] + 1
        self.data = np.vstack((data, np.ones((1, self.m))))
        self.label = np.array(label, ndmin=2).T
        self.beta = None

    def _cache(self, beta):
        if (self.beta != beta).any():
            self.beta = beta.copy()
            self._p1 = 1 / (1 + np.exp(-np.dot(self.data.T, beta)))

    def _dldb(self, beta):    # 对数似然函数一阶导数
        self._cache(beta)
        return np.dot(self.data, self._p1 - self.label)

    def _d2ldb2(self, beta):    # 对数似然函数二阶导数
        self._cache(beta)
        _p1p0 = self._p1 * (1 - self._p1)
        return np.dot(np.tile(_p1p0, self.n).T * self.data, self.data.T)

    def train(self):
        optimizer = Optimization(Newton(1), self._dldb, self._d2ldb2)    # 牛顿法
        self.beta = optimizer.run(np.ones(self.n))

    def classify(self, x):
        return 1 / (1 + np.exp(-(np.dot(self.beta[: -1, 0], x) + self.beta[-1, 0])))


class LMS:
    """
    线性回归/自适应滤波器 - 最小均方算法

    学习核心：
        （随机）梯度下降法
    """

    def __init__(self):
        pass


class RLS:
    """
    线性回归/自适应滤波器 - 递归最小方差算法

    学习核心：
        无
    """

    def __init__(self, data, label):
        self.m = data.shape[1]
        self.n = data.shape[0] + 1
        self.data = np.vstack((data, np.ones((1, self.m))))
        self.label = label
        self.w = np.zeros((self.n, 1))
        self._P = np.identity(self.n) * 1e6    # 误差协方差矩阵/增益矩阵
        self._lambda = 1.   # 遗忘因子/时间衰减系数，一般取0.98~1.0

    def train(self, cycle_count=3):
        i, j = 0, 0
        while i < cycle_count:
            while j < self.m:
                x_j = self.data[:, j].reshape((-1, 1))
                alpha = self.label[j] - np.dot(self.w.T, x_j)
                Px = np.dot(self._P, x_j)
                g = Px * (self._lambda + np.dot(x_j.T, Px))**-1
                self._P = (self._P - g.dot(x_j.T).dot(self._P)) * self._lambda**-1    # 更新P
                self.w += alpha * g
                j += 1
            i += 1

    def classify(self, x):
        return np.dot(self.w[: -1, 0], x) + self.w[-1, 0]


class CurveFitting:
    """
    非线性回归 - 局部加权线性回归算法

    Class of curve fitting, using Locally Weighted Linear Regression algorithm.

    Function: Single point prediction;
              Global fitting;
              Backward point prediction;
              Max y-value Searching.
    """

    def __init__(self, y: list, *x: list, k=1.0):
        self.m = len(y)
        self.n = len(x) + 1
        if self.n == 0:
            self.ret_code = -1
            return
        for xi in x:
            if len(xi) != self.m:
                self.ret_code = -2
                return
        self.xMat = np.concatenate((np.ones((self.m, 1)), np.mat(x).T), axis=1)
        self.yMat = np.mat(y).T
        self.k = k
        self.ret_code = 0

    def _LWLRegress(self, target_x, k):
        """
        LWLR prediction core, always called by other member functions.
        """
        weights = np.mat(np.eye(self.m))
        for i in range(self.m):
            diffX = self.xMat[i] - target_x
            weights[i, i] = np.exp(-0.5 * diffX * diffX.T / k ** 2)
        xTwx = self.xMat.T * weights * self.xMat
        if np.linalg.det(xTwx) == 0:
            print('WARNING: Matrix shrink xTwx is singular when calculating:\n\t%s'
                  % target_x)
            xTwx += np.mat(np.eye(self.n) * 1e-6)

        ws = xTwx.I * self.xMat.T * weights * self.yMat
        return target_x * ws

    def Estimate(self, targetPoint, k=None):
        """Return predicted y-value of given target point,
        which is index of one X in samples or vector of new X."""
        if isinstance(targetPoint, int):
            target_x = self.xMat[targetPoint]
        elif isinstance(targetPoint, list):
            length = len(targetPoint)
            if length == self.n - 1:
                target_x = [1] + targetPoint
            elif length > self.n - 1:
                target_x = [1] + targetPoint[: self.n - 1]
            else:
                target_x = [1] + targetPoint + [0] * (self.n - length - 1)
            target_x = np.mat(target_x)
        else:
            return
        if not k:
            k = self.k
        return self._LWLRegress(target_x, k)[0, 0]

    def Regress(self, show=True, **kwargs):
        """Plot or return predicted y-values of the whole curve."""
        dim = kwargs.get('dim', 1)
        k = kwargs.get('k')
        if k:
            self.k = k
        else:
            k = self.k
        num = kwargs.get('num', 100)
        title = kwargs.get('title')
        xlabel = kwargs.get('xlabel')
        ylabel = kwargs.get('ylabel')
        axis_reverse = kwargs.get('axis_reverse', False)
        scatter_kwargs = kwargs.get('dot_ctrl', {})
        plot_kwargs = kwargs.get('line_ctrl', {'c': 'red'})

        yHat = []
        scale = np.linspace(self.xMat[0, dim], self.xMat[-1, dim], num)
        x = np.zeros(self.n)
        x[0] = 1.
        i = 0
        while i < num:
            _x = x[:]
            _x[dim] = scale[i]
            yHat.append(self._LWLRegress(np.mat(_x), k)[0, 0])
            i += 1

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.suptitle('Locally Weighted Linear Regress Estimation(k=%.1e)'
                         % k)
            if title:
                ax.set_title(str(title))
            if xlabel:
                plt.xlabel(str(xlabel))
            if ylabel:
                plt.ylabel(str(ylabel))
            if axis_reverse:
                ax.scatter(np.array(self.yMat.array()), np.array(self.xMat[:, dim]),
                           **scatter_kwargs)
                ax.plot(yHat, scale, **plot_kwargs)
            else:
                ax.scatter(np.array(self.xMat[:, dim]), np.array(self.yMat),
                           **scatter_kwargs)
                ax.plot(scale, yHat, **plot_kwargs)
            ax.grid(True)
            plt.show()
        else:
            return yHat

    def EstimateY(self, y, x_district, **kwargs):
        """Get x-value of any dimension(default 1) for given y-value in the initial district.
        Using secant method."""
        def func(value):
            _x = x[:]
            _x[dim] = value
            return self._LWLRegress(np.mat(_x), self.k)[0, 0] - y
        x = np.zeros(self.n)
        x[0] = 1.

        dim = kwargs.get('dim', 1)
        prec = kwargs.get('precision', 1e-3)
        divlim = kwargs.get('divergence_limit', 100)
        x0 = x_district[0]
        x1 = x_district[1]
        y_bound = abs(func(x0))
        while abs(x1 - x0) >= prec:
            y0 = func(x0)
            y1 = func(x1)
            dy = y1 - y0
            if abs(dy) >= divlim * y_bound:
                print('Iteration reaches the limit of divergence.')
                break
            x2 = x1 - y1 * (x1 - x0) / dy
            x0 = x1
            x1 = x2
        return round((x0 + x1) / 2, int(- np.log10(prec)))

    def EstimateMax(self, initial_x, **kwargs):
        """Get max y-value of any dimension(default 1), starting from given initial x-value.
        Using gradient descent algorithm."""
        def func(value):
            _x = x[:]
            _x[dim] = value
            return self._LWLRegress(np.mat(_x), self.k)[0, 0]
        x = np.zeros(self.n)
        x[0] = 1.

        dim = kwargs.get('dim', 1)
        step = kwargs.get('step', 0.05)
        prec = kwargs.get('precision', 1e-3)
        dx = kwargs.get('diff_step', 0.01)
        while True:
            grad = (func(initial_x + dx) - func(initial_x)) / dx
            x_new = initial_x + step * grad
            if x_new - initial_x < prec:
                break
            initial_x = x_new
        ndights = int(- np.log10(prec))
        return round(initial_x, ndights), round(func(initial_x), ndights)


def test_logit():
    """
    Test of Logit Regression
    """
    dataname = 'watermelon_3.0alpha'
    data, label, tag = read(dataname)
    l = Logit(data, label)
    l.train()
    post.item_print('beta vector', l.beta, newline=True)
    post.plot(data, label, line=l.beta, title=dataname, tag=tag)
    predict = l.classify(data)
    post.item_print('positive probability', predict, newline=True)


def test_rls():
    """
    Test of Recursive Least Squares
    """
    dataname = 'watermelon_3.0alpha'
    data, label, tag = read(dataname)
    # 重映射样本标签
    label = list(map(lambda x: 2 * x - 1, label))
    # 以下4行：随机打乱数据集
    randInd = np.arange(len(label))
    np.random.shuffle(randInd)
    data = data[:, randInd]
    label = np.array(label)[randInd].tolist()
    l = RLS(data, label)
    l.train()
    post.item_print('w matrix', l.w, newline=True)
    post.plot(data, label, line=l.w, title=dataname, tag=tag)


def test_lwlr():
    # 实验力学-实验九
    X = [10.00, 11.00, 12.00, 13.00, 14.00, 15.00, 16.00, 17.00, 17.25, 17.50, 17.75, 18.00, 18.25, 18.50,
         18.75, 19.00, 20.00, 21.00, 22.00, 23.00, 24.00, 25.00, 26.00, 27.00, 28.00, 29.00, 30.00]
    Y = [15.996, 17.724, 19.835, 22.660, 26.292, 31.264, 37.925, 45.974, 47.619, 48.726, 50.044, 50.410, 50.422, 50.123,
         49.114, 47.372, 38.424, 30.173, 24.310, 20.226, 17.239, 14.709, 12.938, 11.479, 10.345, 9.3460, 8.6284]
    # 构造曲线拟合对象，准备数据
    curve = CurveFitting(Y, X)
    # 以给定k值局部线性加权回归拟合曲线，并显示图像
    curve.Regress(k=0.2, title='单自由度系统幅频响应曲线 B-f',
                  xlabel='f / Hz', ylabel='B / μm')
    # 以给定初始x值寻找拟合曲线的最大值，使用梯度上升法
    x_m, y_m = curve.EstimateMax(18)
    # 从给定初始区间开始寻找y值对应的x值，使用弦截法
    y = y_m / 2 ** 0.5
    x1 = curve.EstimateY(y, [12, 18])
    x2 = curve.EstimateY(y, [18, 25])
    # 打印结果
    print('f0 = %s\nf1 = %s\nf2 = %s' % (x_m, x1, x2))
    zeta = 0.5 * (x2 - x1) / x_m
    print('zeta = %.3f' % zeta)


if __name__ == '__main__':
    test_logit()
    #test_rls()
    #test_lwlr()


