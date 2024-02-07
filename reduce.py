# # # # # # # # # # # # # # # # # # # # # # # #
#
#    降维模块
#
# # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from scipy.linalg import eig

from pre import read
import post


class LDA:
    """
    监督降维 - 线性判别分析 (Linear Discriminant Analysis)

    学习核心：
        广义特征值分解
    """

    def __init__(self, data, label):
        self.data = []
        label_arr = np.array(label)
        for label in set(label):
            self.data.append(data[: , label_arr == label])
        self.n, _ = data.shape
        self.mu = np.mean(data, axis=1, keepdims=True)
        self.W = None

    def train(self, ndim=2):
        Sb = np.zeros((self.n, self.n))    # 类间散度矩阵
        Sw = Sb.copy()    # 类内散度矩阵
        for s in self.data:
            m = s.shape[1]
            mu = np.mean(s, axis=1, keepdims=True)
            vb = mu - self.mu
            Sb += m * np.dot(vb, vb.T)
            vw = s - np.tile(mu, (1, m))
            Sw += np.dot(vw, vw.T)
        d, p = eig(Sb, Sw)    # 广义特征值分解
        ind = np.argsort(d)[: -len(self.data): -1]
        ind = ind[np.abs(d[ind]) >= 1e-8]    # d'个最大非零广义特征值（d'<=N-1，其中N为类别数）
        if ndim < len(ind):
            ind = ind[: ndim]
        self.W = p[: , ind]

    def project(self, x):
        return np.dot(self.W.T, x)


class PCA:
    """
    无监督降维 - 主成分分析 (Principal Component Analysis)

    学习核心：
        特征值分解
    """

    def __init__(self, data):
        self.data = data
        self.n, self.m = data.shape
        self.W = None

    def train(self, ndim=2):
        X = self.data - np.mean(self.data, axis=1, keepdims=True)
        conv = np.dot(X, X.T) / self.m
        d, p = eig(conv)
        ind = np.argsort(d)[: : -1]
        if ndim < len(ind):
            ind = ind[: ndim]
        self.W = p[: , ind]

    def project(self, x):
        return np.dot(self.W.T, x)


def test_lda():
    """
    Test of Linear Discriminant Analysis
    """
    data, label, tag = read("abalone")
    l = LDA(data, label)
    l.train(ndim=3)
    post.item_print("W matrix", l.W, True)
    data_reduced = l.project(data)
    post.plot(data_reduced[(0, 1), : ], label, title="Test of LDA", tag=('x', 'y'))
    post.plot(data_reduced[(0, 2), : ], label, title="Test of LDA", tag=('x', 'z'))
    post.plot(data_reduced[(1, 2), : ], label, title="Test of LDA", tag=('y', 'z'))


def test_pca():
    """
    Test of Pincipal Component Analysis
    """
    data, label, tag = read("abalone")
    l = PCA(data)
    l.train(ndim=3)
    post.item_print("W matrix", l.W, True)
    data_reduced = l.project(data)
    post.plot(data_reduced[(0, 1), :], label, title="Test of PCA", tag=('x', 'y'))
    post.plot(data_reduced[(0, 2), :], label, title="Test of PCA", tag=('x', 'z'))
    post.plot(data_reduced[(1, 2), :], label, title="Test of PCA", tag=('y', 'z'))


if __name__ == '__main__':
    test_lda()
    #test_pca()


