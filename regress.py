# # # # # # # # # # # # # # # # # # # # # # # #
#
#    回归模块
#
# # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from optimization import Optimization, Newton
import pre
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


def test_logit():
    """
    Test of Logit Regression
    """
    dataname = 'watermelon_3.0alpha'
    data, label, tag = pre.read(dataname)
    print('database: %s' % dataname, 'count: %i' % len(label), 'tags: %s' % tag, sep='\n')
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
    data, label, tag = pre.read(dataname)
    print('database: %s' % dataname, 'count: %i' % len(label), 'tags: %s' % tag, sep='\n')
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


if __name__ == '__main__':
    test_logit()
    #test_rls()


