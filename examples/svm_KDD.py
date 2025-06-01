# # # # # # # # # # # # # # # # # # # # # # # #
#
#    案例：基于SVM的多分类（KDD数据集）
#
# # # # # # # # # # # # # # # # # # # # # # # #

import os
from collections import OrderedDict
import pre
import reduce
import SVM
import numpy as np
import post

from sklearn import svm
from time import time


def binary_label(label, string=False):
    new_label = - np.ones(len(label), dtype=int)
    new_label[np.array(label) == 'Normal'] = 1
    if string:
        new_label = new_label.astype(str)
    return new_label.tolist()


def score(label_pred, label_b):
    label_pred_b = - np.ones(len(label_pred), dtype=int)
    label_pred_b[np.array(label_pred) == 'Normal'] = 1
    return len(np.nonzero(label_pred_b == np.array(label_b))[0]) / len(label_pred)


def standardize(data, *other_data):
    var = np.sqrt(data.T.var(axis=0))
    mean = data.T.mean(axis=0)
    std_data = []
    for _data in (data, ) + other_data:
        std_data.append(((_data.T - mean) / var).T)
    return tuple(std_data)


class SMOs(SVM.SMO):

    def __init__(self, dataMatIn, classLabels, C, toler, kTup, cacheSize=10):
        super(SMOs, self).__init__(dataMatIn, classLabels, C, toler, kTup)
        self.dataMat = self.dataMat.astype(np.float32)
        self.labelMat = self.labelMat.astype(np.float32)
        self._cache_size = cacheSize

    class Kdict:

        def __init__(self, dataMat, kTup, max_len=None):
            self._data = dataMat
            self._m = dataMat.shape[1]
            if kTup[0] == 'rbf':
                self._lin = False
                self._den = - 1.0 / kTup[1] ** 2
            else:
                self._lin = True
            if max_len:
                self._max_len = int(max_len)
            else:
                self._max_len = int(8 * 1024**3 / (4 * self._m))
            self._dict = OrderedDict()
            self._n_write = 0
            self._n_read = 0

        def __getitem__(self, index):
            ind_1, ind_2 = index
            if isinstance(ind_1, np.int64):
                if isinstance(ind_2, np.int64):
                    return self._get(ind_1)[ind_2, 0]
                else:
                    return self._get(ind_1).T
            else:
                if isinstance(ind_2, np.int64):
                    return self._get(ind_2)
                else:
                    raise KeyError

        def _get(self, key):
            value = self._dict.get(key)
            if value is None:
                value = self._kernel(key)
                self._dict[key] = value
                if len(self._dict) > self._max_len:
                    self._dict.popitem(last=False)
                self._n_write += 1
            else:
                self._n_read += 1
            return value

        def _kernel(self, i):
            if self._lin:
                # 默认内积
                column = self._data.T * self._data[:, i]
            else:
                # 高斯径向基函数
                column = np.mat(np.zeros((self._m, 1)))
                j = 0
                while j < self._m:
                    delta = self._data[:, j] - self._data[:, i]
                    column[j] = np.exp(delta.T * delta * self._den)
                    j += 1
            return column

    def kernelTrans(self):
        length = int(self._cache_size * 1024 ** 2 / (4 * self.m))
        return self.Kdict(self.dataMat, self.kTup, length)


if __name__ == '__main__':
    data, label, tag = pre.read('KDDTrain')
    label_b = binary_label(label)
    data_test, label_test, _ = pre.read('KDDTest')
    label_test_b = binary_label(label_test)

    # 预处理：PCA白化
    # pca = reduce.PCA(data)
    # pca.train()
    # data_reduced = pca.project(data)
    # data_test_reduced = pca.project(data_test)
    # data = data_reduced
    # data_test = data_test_reduced

    # 预处理：标准化
    data_s = pre.standardize(np.concatenate((data, data_test), axis=1))
    data = data_s[: , : data.shape[1]]
    data_test = data_s[: , data.shape[1]: ]

    # 自己实现的SVM
    # data_flat = data.flatten()
    # print("Sparse Rate: %.2f %%" % (100 * len(np.nonzero(data_flat)[0]) / len(data_flat)))
    #
    # sigma = np.sqrt(data.shape[0])
    # print("RBF Sigma: %f" % sigma)
    # classifier = SMOs(data, label_b, C=1, toler=1e-5, kTup=('lin', sigma), cacheSize=8000)
    # print("Start training...")
    # classifier.train(500)
    # print("Done!")
    #
    # post.item_print("Kdict", "%i write, %i read" % (classifier.K._n_write, classifier.K._n_read))
    # post.item_print("SMO Counter", classifier.counter)
    # post.item_print("SV Count", "%i (slack: %i)" % (len(np.nonzero(classifier.checkSV())[0]),
    #                                                 len(np.nonzero(
    #                                                     classifier.alphas > classifier.C - classifier.toler_sv)[0])))
    # post.item_print("Accuracy on Train Set", "%.2f %%" % (100 * classifier.accuracy()))
    # post.item_print("Accuracy on Validation Set", "%.2f %%" % (100 * classifier.accuracy(data_test, label_test_b)))

    # 使用libsvm库
    # timer = - time()
    # classifier = svm.SVC(C=1, kernel='linear', gamma='scale', max_iter=-1, class_weight=None, shrinking=True)
    # classifier.fit(data.T, label_b)
    # timer += time()
    # print("Time spent: %.1f s" % timer)
    # print("SV Count: %i" % classifier.n_support_.sum())
    # print("Accuracy on Train Set: %.2f %%" % (100 * classifier.score(data.T, label_b)))
    # print("Accuracy on Test Set: %.2f %%" % (100 * classifier.score(data_test.T, label_test_b)))

    # 使用libsvm库，并引入类别权重和多分类
    label = np.array(label)
    cls_set = set(label)
    print(cls_set)
    weight = {}
    for cls in cls_set:
        weight[cls] = len(label) - len(np.nonzero(label == cls)[0])
    timer = - time()
    classifier = svm.SVC(C=1, kernel='linear', max_iter=-1, class_weight=weight, decision_function_shape='ovr')
    classifier.fit(data.T, label)
    timer += time()
    print("Time spent: %.1f s" % timer)
    print("SV Count: %i" % classifier.n_support_.sum())
    print("Accuracy on Train Set: %.2f %%" % (100 * score(classifier.predict(data.T), label_b)))
    print("Accuracy on Test Set: %.2f %%" % (100 * score(classifier.predict(data_test.T), label_test_b)))


