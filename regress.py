
import numpy as np
from optimization import *


class logit():
    '''
    分类 - 对数几率回归

    学习核心：
        数值优化算法 | 牛顿法、(随机)梯度下降法 等
    '''

    def __init__(self, data, label):
        self.m = data.shape[1]
        self.n = data.shape[0] + 1
        self.data = np.vstack((data, np.ones((1, self.m))))
        self.label = np.array(label, ndmin=2).T
        self.beta = None

    def _cache(self, beta):
        if (self.beta != beta).any():
            self.beta = beta
            self._p1 = 1 / (1 + np.exp(-np.dot(self.data.T, beta)))

    def _dldb(self, beta):    # 对数似然函数一阶导数
        self._cache(beta)
        return np.dot(self.data, self._p1 - self.label)

    def _d2ldb2(self, beta):    # 对数似然函数二阶导数
        self._cache(beta)
        _p1p0 = self._p1 * (1 - self._p1)
        return np.dot(np.tile(_p1p0, self.n).T * self.data, self.data.T)

    def train(self):
        optimizer = Newton(self._dldb, self._d2ldb2)    # 牛顿法
        self.beta = optimizer.run(np.ones((self.n, 1)))

    def classify(self, x):
        return 1 / (1 + np.exp(-(np.dot(self.beta[: -1, 0], x) + self.beta[-1, 0])))


class lms():
    '''
    线性回归/自适应滤波器 - 最小均方算法

    学习核心：
        （随机）梯度下降法
    '''

    def __init__(self):
        pass


class rls():
    '''
    线性回归/自适应滤波器 - 递归最小方差算法

    学习核心：
        无
    '''

    def __init__(self, data, label):
        self.m = data.shape[1]
        self.n = data.shape[0] + 1
        self.data = np.vstack((data, np.ones((1, self.m))))
        self.label = label
        self.w = np.zeros((self.n, 1))
        self._P = np.identity(self.n) * 1e6    # 误差协方差矩阵/增益矩阵
        self._lambda = 1.   # 遗忘因子/时间衰减系数，一般取0.98~1.0

    def train(self, cycle_count=3):
        i = 0; j = 0
        while i < cycle_count:
            while j < self.m:
                x_j = self.data[:, j].reshape((-1, 1))
                alpha = self.label[j] - np.dot(self.w.T, x_j)
                Px = np.dot(self._P, x_j)
                g = Px * (self._lambda + np.dot(x_j.T, Px))**-1
                self._P = (self._P - g.dot(x_j.T).dot(self._P)) * self._lambda**-1    # 更新P
                self.w += alpha * g
                j += 1
            print(self.w)
            i += 1

    def classfy(self, x):
        return np.dot(self.w[: -1, 0], x) + self.w[-1, 0]


