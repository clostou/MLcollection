# # # # # # # # # # # # # # # # # # # # # # # #
#
#    神经网络模块
#
# # # # # # # # # # # # # # # # # # # # # # # #


import numpy as np
import matplotlib.pyplot as plt
import pre
import post


class _Layer:

    def __init__(self, input_n, neuron_n):
        self.m = neuron_n
        self.n = input_n
        self.w = np.random.random((input_n, neuron_n))
        self.b = np.random.random(neuron_n)

    def act_func(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __call__(self, input):
        return self.act_func(np.dot(self.w.T, input).T - self.b).T


class ActFunc:

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def ReLU(z):
        z[z < 0] = 0
        return z


class _FullyConnectedLayer:

    def __init__(self, n_in, n_out, activation_fn=ActFunc.sigmoid, p_dropout=0.0):
        pass


class MFF1:
    """
    神经网络 - 单隐层前馈网络

    配置：
        二次代价函数
        逻辑神经元

    学习核心：
        误差逆传播(BP)算法：最小均方算法+梯度下降法
    """

    def __init__(self, data, label, hidden_neuron_count=4):
        self.m = data.shape[1]
        self.n = data.shape[0]
        self.data = pre.normalize(data)    # 正则化
        label = np.array(label)
        classes = np.unique(label)[: : -1]
        self.label = np.zeros((len(label), len(classes)))
        for i in range(len(classes)):
            self.label[: , i] = label == classes[i]
        self.hidden_layer = _Layer(self.n, hidden_neuron_count)
        self.output_layer = _Layer(hidden_neuron_count, len(set(label)))
        self.cycle_i = 0
        # 迭代监测
        self.feedback_win = post.PlotAniNet('Training of Neural-network (BP Alg.)',
                                         'Iterration', 'Accumulate Error')

    def _BP_standard(self, learning_rate=1):
        error = 0.0
        update_list = list(range(self.m))
        np.random.shuffle(update_list)
        while update_list:
            k = update_list.pop()
            # 输入正向传播，计算当前网络的输出值
            hidden_y = self.hidden_layer(self.data[: , k])
            output_y = self.output_layer(hidden_y)
            # 误差逆向传播，计算出每个神经元上的误差负梯度
            error += np.linalg.norm(self.label[k, : ] - output_y)
            output_g = output_y * (1.0 - output_y) * (self.label[k, : ] - output_y)
            hidden_g = hidden_y * (1.0 - hidden_y) * np.dot(self.output_layer.w, output_g)
            # 更新每层的连接权重与阈值
            self.output_layer.w += learning_rate * np.outer(hidden_y, output_g)
            self.output_layer.b -= learning_rate * output_g
            self.hidden_layer.w += learning_rate * np.outer(self.data[: , k], hidden_g)
            self.hidden_layer.b -= learning_rate * hidden_g
        return error / (2 * self.m)

    def _BP_accumulate(self, learning_rate=3):
        error = 0.0
        # 输入正向传播，计算当前网络的输出值
        hidden_y = self.hidden_layer(self.data)
        output_y = self.output_layer(hidden_y)
        # 误差逆向传播，计算出每个神经元上的误差负梯度
        error += np.mean(np.linalg.norm(self.label.T - output_y, axis=0))
        output_g = output_y * (1.0 - output_y) * (self.label.T - output_y)
        hidden_g = hidden_y * (1.0 - hidden_y) * np.dot(self.output_layer.w, output_g)
        # 更新每层的连接权重与阈值
        self.output_layer.w += learning_rate / self.n * np.dot(hidden_y, output_g.T)
        self.output_layer.b -= learning_rate * np.mean(output_g, axis=1)
        self.hidden_layer.w += learning_rate / self.n * np.dot(self.data, hidden_g.T)
        self.hidden_layer.b -= learning_rate * np.mean(hidden_g, axis=1)
        return error / 2

    def train(self, cycle_count=100, step_adapt=False):
        if step_adapt:
            step = lambda i: 8.0 / (1.0 + 0.01 * i)
        else:
            step = lambda i: 2.0
        total_cycle_count = self.cycle_i + cycle_count
        while self.cycle_i < total_cycle_count:
            iter = len(self.feedback_win.xdata[0])
            self.feedback_win.add(0, iter, self._BP_standard(step(iter)))
            self.feedback_win.update()
            self.cycle_i += 1

    def classify(self, x):
        return self.output_layer(self.hidden_layer(x))


class SOM:
    """
    神经网络/聚类 - 自组织映射网络

    学习核心：
        无
    """

    def __init__(self, data, network_size=10):
        self.n, self.m = data.shape
        self.net_size = network_size
        self.data = pre.normalize(data)
        self.network = 0.2 * np.random.random((network_size, network_size, self.n)) - 0.1
        self.cycle_i = 0

        index = np.arange(self.net_size)
        self._Y, self._X = np.meshgrid(index, index)

    def train(self, cycle_count=100):
        sigma_0 = 0.5 * self.net_size
        tau_1 = 1e3 / np.log(sigma_0)
        ita_0 = 0.1
        tau_2 = 1e3
        total_cycle_count = self.cycle_i + cycle_count
        while self.cycle_i < total_cycle_count:
            k = np.random.randint(self.m)
            simple = self.data[: , k]
            distance = np.sum((self.network - simple)**2, axis=2)
            # 得到获胜神经元的位置
            pos = np.unravel_index(np.argmin(distance), distance.shape)
            # 计算拓扑邻域宽度
            sigma = sigma_0 * np.exp(-self.cycle_i / tau_1)
            # 计算临近神经元的调整权重
            h = np.exp(-0.5 / sigma**2 * ((self._X - pos[0])**2 + (self._Y - pos[1])**2))
            # 动态更新学习率
            ita = ita_0 * np.exp(-self.cycle_i / tau_2)
            self.network -= ita * ((self.network - simple).T * h.T).T
            self.cycle_i += 1

    def classify(self, data):
        data = pre.normalize(data)
        if data.ndim == 1:
            distance = np.sum((self.network - data)**2, axis=2)
        else:
            m = data.shape[1]
            distance = np.zeros((self.net_size, self.net_size, m))
            i = 0
            while i < m:
                distance[:, :, i] = np.sum((self.network - data[: , i])**2, axis=2)
                i += 1
        return distance == np.max(distance, axis=(0, 1))


def test_bp():
    """
    Test of BP Algorithm

        易陷入局部最优。考虑对错分样本单独学习，而不是乱序对所有样本进行学习
    """
    dataname = 'watermelon_3.0'
    data, label, tag = pre.read(dataname)
    print('database: %s' % dataname, 'count: %i' % len(label), 'tags: %s' % tag, sep='\n')
    n = MFF1(data, label, hidden_neuron_count=4)
    post.item_print('initial error', 0.5 * np.mean(np.linalg.norm(n.label.T - n.classify(data), axis=0)))
    n.train(200, True)
    post.item_print('hidden-layyer w', n.hidden_layer.w, True)
    post.item_print('hidden-layyer b', n.hidden_layer.b, True)
    post.item_print('output-layyer w', n.output_layer.w, True)
    post.item_print('output-layyer b', n.output_layer.b, True)
    post.item_print('final error', 0.5 * np.mean(np.linalg.norm(n.label.T - n.classify(data), axis=0)))


def test_som():
    """
    Test of SOM Network
    """
    dataname = 'watermelon_3.0alpha'
    data, label, tag = pre.read(dataname)
    print('database: %s' % dataname, 'count: %i' % len(label), 'tags: %s' % tag, sep='\n')
    n = SOM(data, network_size=10)
    n.train(cycle_count=2000)
    post.mat_scatter(n.classify(data), label)


if __name__ == '__main__':
    test_bp()
    #test_som()


