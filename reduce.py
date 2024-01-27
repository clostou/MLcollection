
import numpy as np
from numpy import linalg as LA
from scipy.linalg import eig


class lda():
    '''
    监督降维 - 线性判别分析

    学习核心：
        广义特征值分解
    '''

    def __init__(self, data, label):
        self.data = []
        label_arr = np.array(label)
        for label in set(label):
            self.data.append(data[: , label_arr == label])
        self.n, _ = data.shape
        self.mu = np.mean(data, axis=1, keepdims=True)

    def train(self):
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
        self.W = p[: , ind]

    def classify(self, x):
        return np.dot(self.W.T, x)


