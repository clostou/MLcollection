# # # # # # # # # # # # # # # # # # # # # # # #
#
#    算法：支持向量机
#
# # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from time import time

from pre import read
import post


class SMO:
    """
    序列最小优化算法 (Sequential Minimal Optimization, SMO)

    用于对软间隔非线性支持向量机 (Support Vector Machine, SVM) 进行训练
    """
    
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.dataMat = np.mat(dataMatIn, dtype=np.float64)
        self.labelMat = np.mat(classLabels, dtype=np.float64).transpose()
        self.n, self.m = np.shape(self.dataMat)
        self.C = C
        self.toler_ktt = toler
        self.toler_sv = 1e-5
        self.kTup = kTup
        self.alphas = None
        self.b = None
        self.eCache = None
        self.K = None
        self.counter = [0, 0, 0, 0, 0.]
    
    def train(self, maxIter):
        self.counter[4] = - time()
        self.initialize()
        iterNum = 0
        while iterNum < maxIter:
            ifAlphaPairsChanged = self.outerL()
            if not ifAlphaPairsChanged:
                break
            iterNum += 1
        self.counter[4] += time()
    
    def train_visual(self, maxIter, **kwargs):
        # 同train函数，但会动态绘制迭代过程
        self.counter[4] = - time()
        draw = post.PlotAniSVM(self.dataMat, self.labelMat, **kwargs)
        draw.scatter()
        self.initialize()
        iterNum = 0
        while iterNum < maxIter:
            ifAlphaPairsChanged = self.outerL()
            if ifAlphaPairsChanged:
                sv = np.nonzero(self.checkSV())[0]
                if self.kTup[0] == 'lin':
                    w = self.dataMat * np.multiply(self.alphas, self.labelMat)
                    draw.plotLine(w, self.b, sv)
                elif self.kTup[0] == 'rbf':
                    draw.plotContour(lambda x: self.classify(x, continuous=True), sv)
                else:
                    print("Unknown flag of kernel in parameter 'kTup'")
                    break
            else:
                break
            iterNum += 1
        self.counter[4] += time()
    
    def initialize(self):
        # 变量初始化
        self.K = self.kernelTrans()
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0.
        self.eCache = - np.copy(self.labelMat)
        # 初值不是最优解但满足KTT条件，需要手动对所有样本进行一轮循环
        alphaIList = np.arange(self.m)
        outer_ind = 0
        while outer_ind < len(alphaIList):
            i = alphaIList[outer_ind]
            self.innerL(i)
            outer_ind += 1
    
    def kernelTrans(self):
        if self.kTup[0] == 'lin':
            # 默认内积
            kMat = self.dataMat.T * self.dataMat
        elif self.kTup[0] == 'rbf':
            # 高斯径向基函数
            kMat = np.mat(np.zeros((self.m, self.m)))
            for i in range(self.m):
                row = np.zeros(self.m)
                for j in range(self.m):
                    deltaRow = self.dataMat[: , j] - self.dataMat[: , i]
                    row[j] = np.exp(deltaRow.T * deltaRow / (- self.kTup[1]**2))
                kMat[i, : ] = row
        else:
            kMat = None
            print("Unknown flag of kernel in parameter 'kTup'")
        return kMat
    
    def updateEk(self, k):
        # 计算预测误差
        fXk = self.K[: , k].T * np.multiply(self.alphas, self.labelMat) + self.b
        Ei = fXk - self.labelMat[k]
        self.eCache[k] = Ei
        return Ei
    
    def checkSV(self):
        # 检查是否为非松弛的支持向量
        isSV = (self.alphas >= self.toler_sv) & (self.alphas <= self.C - self.toler_sv)
        return isSV
    
    def checkKTT(self, k):
        # 检查是否满足互补松弛条件（最优解的必要条件）
        Ey = np.multiply(self.eCache[k], self.labelMat[k])
        alpha = self.alphas[k]
        dissatisfyKTT = ((alpha != self.C) & (Ey < - self.toler_ktt)) | \
                        ((alpha != 0) & (Ey > self.toler_ktt))
        return np.nonzero(dissatisfyKTT)[0]
    
    def outerL(self):
        isSV = self.checkSV()
        alphaIList = np.nonzero(isSV)[0]
        ifAlphaPairsChanged = self._outerL(alphaIList)
        if not ifAlphaPairsChanged:
            alphaIList = np.nonzero(~ isSV)[0]
            ifAlphaPairsChanged = self._outerL(alphaIList)
        return ifAlphaPairsChanged
    
    def _outerL(self, alphaIList):
        # 外循环，按顺序选择第一个alpha
        index = self.checkKTT(alphaIList)
        if len(index) == 0:
            return False
        alphaIList = alphaIList[index]
        alphaPairsChanged = 0
        outer_ind = 0
        while outer_ind < len(alphaIList):
            i = alphaIList[outer_ind]
            alphaPairsChanged += self.innerL(i)
            outer_ind += 1
        return alphaPairsChanged > 0
    
    def innerL(self, i):
        isSV = self.checkSV()
        alphaJList = np.nonzero(isSV)[0]
        ifAlphaPairsChanged = self._innerL(i, alphaJList)
        if not ifAlphaPairsChanged:
            alphaJList = np.nonzero(~ isSV)[0]
            ifAlphaPairsChanged = self._innerL(i, alphaJList)
        return ifAlphaPairsChanged
    
    def _innerL(self, i, alphaJList):
        # 内循环，使用最大化步长的启发式选定第二个alpha
        if len(alphaJList) == 0 or (len(alphaJList) == 1 and alphaJList[0]) == i:
            return False
        Ei = self.eCache[i]
        Ek = self.eCache[alphaJList]
        inner_ind = np.argmax(np.abs(Ek - Ei))
        j = alphaJList[inner_ind]
        return self.updateAlpha(i, j)
    
    def clipAlpha(self, aj, H, L):
        if aj < L: aj = L
        elif aj > H: aj = H
        return aj
    
    def updateAlpha(self, i, j):
        # 尝试优化给定的两个alpha，若成功则返回True
        yij = self.labelMat[i, 0] * self.labelMat[j, 0]
        if yij < 0:
            H = min(self.C, self.C + self.alphas[j, 0] - self.alphas[i, 0])
            L = max(0, self.alphas[j, 0] - self.alphas[i, 0])
        else:
            H = min(self.C, self.alphas[j, 0] + self.alphas[i, 0])
            L = max(0, - self.C + self.alphas[j, 0] + self.alphas[i, 0])
        if H == L:
            self.counter[1] += 1
            return False
        eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
        if eta == 0:
            self.counter[2] += 1
            return False
        aj = self.alphas[j, 0] + \
            self.labelMat[j, 0] * (self.eCache[i, 0] - self.eCache[j, 0]) / eta
        aj = self.clipAlpha(aj, H, L)
        if np.abs(aj - self.alphas[j, 0]) < self.toler_sv:
            self.counter[3] += 1
            return False
        ai = self.alphas[i, 0] - (aj - self.alphas[j, 0]) * yij
        # 更新乘子alpha
        dayi = (ai - self.alphas[i, 0]) * self.labelMat[i, 0]
        dayj = (aj - self.alphas[j, 0]) * self.labelMat[j, 0]
        self.alphas[i, 0] = ai
        self.alphas[j, 0] = aj
        # 更新偏置b
        dbi = - self.eCache[i, 0] - dayi * self.K[i, i] - dayj * self.K[j, i]
        dbj = - self.eCache[j, 0] - dayi * self.K[i, j] - dayj * self.K[j, j]
        if 0 < ai < self.C: db = dbi
        elif 0 < aj < self.C: db = dbj
        else: db = (dbi + dbj) / 2
        self.b += db
        # 更新误差向量Ek
        dEk = dayi * (self.K[: , i] - self.K[: , j]) + db
        self.eCache += dEk
        self.counter[0] += 1
        return True
    
    def classify(self, x, continuous=False):
        # 对单个样本（1维数组）或多个样本（按列排列的k维数组）进行分类
        svInd = np.nonzero(self.alphas > self.toler_sv)[0]
        dataArr = self.dataMat[: , svInd].A
        alphas = self.alphas[svInd].A
        labels = self.labelMat[svInd].A
        ay = (alphas * labels).flatten()
        if self.kTup[0] == 'lin':
            w = np.dot(dataArr, ay)
            distance = np.dot(w, x) + self.b
        elif self.kTup[0] == 'rbf':
            # 下式数组维度变化为：(m, n, 1) - (1, n, k) = (m, n, k)
            delta = dataArr.T[: , : , np.newaxis] - x[np.newaxis, : ]
            product = np.exp( np.sum(np.power(delta, 2), axis=1) / (- self.kTup[1]**2))
            distance = np.dot(ay, product) + self.b
        else:
            print("Unknown flag of kernel in parameter 'kTup'")
            return
        if continuous:
            # 直接返回样本点的函数距离（连续值）
            return distance
        else:
            # 返回样本点的预测标签（离散值）
            if isinstance(distance, np.ndarray):
                predictLabel = np.ones(len(distance))
                predictLabel[distance < 0] = -1
            else:
                predictLabel = 1 if distance >= 0 else -1
            return predictLabel
    
    def accuracy(self, x=None, y=None):
        if isinstance(x, type(None)):
            m = self.m
            x = self.dataMat.A
            y = self.labelMat.A.flatten()
        else:
            _, m = x.shape
        return len(np.nonzero(self.classify(x) == y)[0]) / m


if __name__ == '__main__':
    data_file = ['testSet', 'testSetRBF', 'testSetRBF2']
    dataArr, labelArr, _ = read(data_file[2])
    smo = SMO(dataArr, labelArr, 50, 0.001, ('rbf', 0.5))
    smo.train_visual(500)
    post.item_print("SMO Counter", smo.counter)
    post.item_print("Accuracy: ", smo.accuracy())
    #print(np.nonzero(smo.alphas)[0])


