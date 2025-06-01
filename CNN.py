# # # # # # # # # # # # # # # # # # # # # # # #
#
#    算法：卷积神经网络
#
# # # # # # # # # # # # # # # # # # # # # # # #

import os
import pickle
import gzip
from time import time

import numpy as np
import torch
import torch.nn.functional as tf
from .post import PlotAniNet


'''问题
1. 弃权下的逆传播是怎么样的
2. 卷积池化复合层（最大池化）下BP算法里的神经元输入值计算
'''


# # # # # # # # # # # #  加载数据  # # # # # # # # # # # #

def load_data(path, encoding='ASCII'):
    """
    加载压缩的序列化数据（如MNIST数据集）

    :param path: 数据文件路径
    :param encoding: 序列化编码字符集
    :return: 反序列化对象
    """
    with gzip.open(path) as zf:
        return pickle.load(zf, encoding=encoding)


def save_data(object, path):
    """
    保存为压缩的序列化数据

    :param object: 需要保存的Python对象
    :param path: 数据文件路径
    :param encoding: 序列化编码字符集
    """
    with gzip.open(path, 'wb') as zf:
        pickle.dump(object, zf)


def load_mnist():
    """
    加载格式预处理的MNIST手写图像数据集 (float32)

    :return: 训练数据、验证数据、测试数据的元组
    """
    data_path = r".\dataBase\mnist_formated.pkl.gz"
    if os.path.exists(data_path):
        return load_data(data_path)
    else:
        formated_data = []
        for data in load_data(r".\dataBase\mnist.pkl.gz", 'latin1'):
            #reshaped_x = np.reshape(data[0].T, (28, 28, -1))
            reshaped_x = data[0]
            m = len(data[1])
            vectorized_y = np.zeros((m, 10), dtype=np.float32)
            vectorized_y[np.arange(m), data[1]] = 1.0
            formated_data.append((reshaped_x, vectorized_y))
        formated_data = tuple(formated_data)
        save_data(formated_data, data_path)
        return formated_data


# # # # # # # # # # # #  人工神经元的激活函数  # # # # # # # # # # # #

class sigmoid:

    @staticmethod
    def fn(z):
        if isinstance(z, torch.Tensor):
            return tf.sigmoid(z)
        else:
            return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def d(a):
        return np.multiply(a, 1.0 - a)


class tanh:

    @staticmethod
    def fn(z):
        if isinstance(z, torch.Tensor):
            return tf.tanh(z)
        else:
            return np.tanh(z)

    @staticmethod
    def d(a):
        return 1.0 - a**2


class relu:

    @staticmethod
    def fn(z):
        if isinstance(z, torch.Tensor):
            return tf.relu(z)
        else:
            z[z < 0] = 0.0
            return z

    @staticmethod
    def d(a):
        a[a > 0] = 1.0
        return a


# # # # # # # # # # # #  前馈网络类，用于连接各层构成网络以及训练  # # # # # # # # # # # #

class Network:

    def __init__(self, layers):
        """
        :param layers: 网络层对象的有序列表，描述了网络的结构
        """
        self.layers = layers
        self.layer_n = len(layers)
        self.dof = 0 # 网络参数量
        self.velocity = [] # momentum速度项
        for layer in layers:
            if isinstance(layer, ConvPoolLayer):
                self.dof += (np.prod(layer.filter_shape[1: ]) + 1) * layer.filter_shape[0]
            else:
                self.dof += (layer.n_in + 1) * layer.n_out
            self.velocity.append((np.zeros(layer.w.shape, dtype=np.float32),
                                  np.zeros(layer.b.shape, dtype=np.float32)))
        self.epoch = 0
        self.best_test_accr = (0, 0.0, 0.0)
        self.monitor = PlotAniNet('Training of CNN', 'Epoch', 'Accuracy (%)', 2,
                                  ['test data', 'training data'])

    def _forword(self, input, dropout=False):
        """
        计算给定输入值下当前网络的输出

        :param input: 网络的输入值，为按行排布的二维numpy数组
        :param dropout: 是否用于训练
        """
        layer_i = 0
        value_forword = input.T
        while layer_i < self.layer_n:
            self.layers[layer_i].propagate(value_forword, dropout=dropout)
            value_forword = self.layers[layer_i].output
            layer_i += 1

    def _backword(self, output, eta=1.0, lmbda=0.0, mu=0.2):
        """
        由预期输出更新网络各层

        :param input: 网络的预期输出值，为按行排布的二维numpy数组
        :param eta: 学习率（调整步长）
        :param lmbda: 规范化参数
        :param mu: momentum梯度下降法的摩擦力项
        """
        layer_i = self.layer_n - 1
        value_backword = output.T
        decay_factor = lmbda * eta / self.num_training_batch # 规范化衰减项
        while layer_i >= 0:
            current_layer = self.layers[layer_i]
            current_layer.feedback(value_backword)
            value_backword = current_layer.input_g
            # 根据梯度调整权重与偏置
            v_w, v_b = self.velocity[layer_i]
            v_w = mu * v_w + eta * current_layer.grad_w - decay_factor * current_layer.w
            v_b = mu * v_b + eta * current_layer.grad_b
            current_layer.w += v_w
            current_layer.b += v_b
            self.velocity[layer_i] = (v_w, v_b)
            layer_i -= 1

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda, test_data):
        """
        :param training_data: 训练样本及其真实值的元组，均为按行排布的二维numpy数组
        :param epochs: 迭代回合数
        :param mini_batch_size: 小批量集的大小
        :param eta: 学习率
        :param test_data: 测试样本集，格式同training_data
        """
        self.eta = eta
        training_x, training_y = training_data
        num_training = len(training_y)
        split_index = np.arange(mini_batch_size, num_training, mini_batch_size)
        training_batch_x = np.vsplit(training_x, split_index)
        training_batch_y = np.vsplit(training_y, split_index)
        self.num_training_batch = int(num_training / mini_batch_size) # 若不能整除，则舍去部分样本
        test_x, test_y = test_data
        # 在训练集上迭代以训练网络，使用BP算法（随机梯度下降）
        print("Start training network (DOF: %d)" % self.dof)
        epochs += self.epoch; timer = time()
        while self.epoch < epochs:
            batch_indices_list = np.arange(self.num_training_batch)
            #np.random.shuffle(batch_indices_list) # 每回合按不同顺序训练小批量集
            for mini_batch_index in batch_indices_list:
                iteration = self.num_training_batch * self.epoch + mini_batch_index
                if iteration % 10000 == 0:
                    print("Training mini-batch number %d (~%.1f ms/iter)" %
                          (iteration, (time() - timer) / 10))
                    timer = time()
                # 输入值前向传播
                self._forword(training_batch_x[mini_batch_index], dropout=True)
                # 输出值误差反向传播
                eta = self.learning_rate(iteration)
                self._backword(training_batch_y[mini_batch_index], eta=eta, lmbda=lmbda)
            # 使用测试集评估网络
            self._forword(test_x)
            test_cost, test_accr = self.layers[-1].accuracy(test_y.T)
            self._forword(training_x)
            training_cost, training_accr = self.layers[-1].accuracy(training_y.T)
            weight_cost = 0.0    # 代价函数规范化项
            for layer in self.layers:
                weight_cost += np.sum(layer.w**2)
            weight_cost = lmbda * weight_cost / (2 * self.num_training_batch)
            print("Epoch %d: test accuracy %.2f%%, cost %.2f" %
                  (self.epoch, test_accr, test_cost + weight_cost))
            if test_accr > self.best_test_accr[1]:
                self.best_test_accr = (self.epoch, test_accr, training_accr)
            self.monitor.add(0, self.epoch, test_accr)
            self.monitor.add(1, self.epoch, training_accr)
            self.monitor.update()
            self.epoch += 1
        print("Finished training network.")
        print("Best test accuracy of %.2f%% obtained at epoch %d\n"
              "Corresponding training accuracy of %.2f%%" %
              (self.best_test_accr[1], self.best_test_accr[0], self.best_test_accr[2]))

    def learning_rate(self, iteration):
        return self.eta


# # # # # # # # # # # #  各类隐藏层和输出层  # # # # # # # # # # # #

class FullyConnectedLayer:
    """
    全连接层
    """

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        """
        :param n_in: 输入向量的长度
        :param n_out: 输出向量的长度（即该层神经元的个数）
        :param activation_fn: 激活函数
        :param p_dropout: 弃权率
        """
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # 初始化权重与偏差
        self.w = np.random.normal(
            loc=0.0, scale=1.0 / np.sqrt(n_in), size=(n_out, n_in)).astype(np.float32)
        self.b = np.random.normal(loc=0.0, scale=1.0, size=(n_out, 1)).astype(np.float32)

    def propagate(self, input, dropout=False, **kwargs):
        """
        :param input: 上一层的输出值，为按列排布的二维numpy数组
        :param dropout: 是否用于训练，用来决定是否进行弃权
        """
        self._input = input
        if dropout: p_dropout = self.p_dropout
        else: p_dropout = 0.0
        self.output_origin = self.activation_fn.fn(np.dot(self.w, input) + self.b)
        self.output, self.dropout_mask = dropout_layer(self.output_origin, p_dropout)

    def feedback(self, output_g):
        """
        :param output_g: 该层输出值的误差负梯度，为按列排布的二维numpy数组
        """
        error = np.multiply(self.activation_fn.d(self.output_origin), output_g)
        error_dropout = dropout_layer(error, self.p_dropout, self.dropout_mask)
        self.input_g = np.dot(self.w.T, error_dropout)
        self.grad_w = np.zeros(self.w.shape, dtype=np.float32)
        i = 0; n = output_g.shape[1]
        while i < n: self.grad_w += np.outer(error_dropout[:, i], self._input[: , i]); i += 1
        self.grad_w /= n
        self.grad_b = np.mean(error_dropout, axis=1, keepdims=True)


class ConvPoolLayer:
    """
    卷积池化复合层
    """

    def __init__(self, filter_shape, image_shape, pooling='max', poolsize=(2, 2), activation_fn=sigmoid):
        """
        :param filter_shape: 4元数组，分别为滤镜的个数、通道数、高度、宽度
        :param image_shape: 3元数组，分别为图片的通道数、高度、宽度
        :param pooling: 池化层类型
        :param poolsize: 2元数组，为池化单元的尺寸
        :param activation_fn: 激活函数
        """
        self.filter_shape = filter_shape
        self._padding = (filter_shape[2] - 1, filter_shape[3] - 1)
        self.image_shape = (-1, *image_shape)
        self.pooling = pooling
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # 初始化权重与偏差
        n_in = np.prod(filter_shape[1: ])
        self.w = np.random.normal(
            loc=0.0, scale=1.0 / np.sqrt(n_in), size=filter_shape).astype(np.float32)
        self.b = np.random.normal(loc=0.0, scale=1.0, size=(filter_shape[0], )).astype(np.float32)

    def propagate(self, input, **kwargs):
        """
        :param input: 上一层的输出值，为按列排布的二维numpy数组
        """
        self._w = torch.tensor(self.w)
        self._b = torch.tensor(self.b)
        self._input = torch.tensor(input.T.reshape(self.image_shape)) # 卷积层输入（tensor）
        self.num_mini_batch = len(self._input)
        # 卷积（valid）与非线性化
        self._input_filterd = self.activation_fn.fn(
            tf.conv2d(self._input, self._w, bias=self._b)) # 卷积层输出（tensor）
        # 池化（无匹配值时不池化）
        if self.pooling == 'max':
            self._input_shrinked, self._max_pool_ind = \
                tf.max_pool2d(self._input_filterd, self.poolsize, return_indices=True) # 池化层输出（tensor）
        elif self.pooling == 'L2':
            self._input_shrinked = tf.lp_pool2d(self._input_filterd, 2, self.poolsize)
        elif self.pooling == 'average':
            self._input_shrinked = tf.avg_pool2d(self._input_filterd, self.poolsize)
        else:
            self._input_shrinked = self._input_filterd
        #output = self._input_shrinked.detach().cpu().numpy()
        self.output = self._input_shrinked.reshape((self.num_mini_batch, -1)).T.numpy()

    def feedback(self, output_g):
        """
        :param output_g: 该层输出值的误差负梯度，按列排布的二维numpy数组
        """
        error_pool = torch.tensor(output_g).reshape(self._input_shrinked.shape)
        # 上采样
        if self.pooling == 'max':
            error_upsample = tf.max_unpool2d(error_pool, self._max_pool_ind, self.poolsize)
        elif self.pooling == 'L2':
            error_upsample = tf.interpolate(error_pool / self._input_shrinked,
                               size=self._input_filterd.shape[-2: ]) * self._input_filterd
        elif self.pooling == 'average':
            error_upsample = (1.0 / self.poolsize[0] * self.poolsize[1]) * \
                             tf.interpolate(error_pool, size=self._input_filterd.shape[-2: ])
        else:
            error_upsample = error_pool
        # 卷积层梯度计算
        error_conv = self.activation_fn.d(self._input_filterd) * error_upsample
        input_g = tf.conv2d(error_conv, torch.flip(self._w, (2, 3)).permute(1, 0, 2, 3),
                            padding=self._padding)
        self.input_g = input_g.reshape((self.num_mini_batch, -1)).T.numpy()
        grad_w = tf.conv2d(self._input.permute((1, 0, 2, 3)),
                               error_conv.permute((1, 0, 2, 3))).permute((1, 0, 2, 3)) \
                 / self.num_mini_batch
        self.grad_w = grad_w.numpy()
        grad_b = torch.mean(torch.sum(error_conv, dim=(2, 3)), dim=0)
        self.grad_b = grad_b.numpy()


class SoftmaxLayer:
    """
    柔性最大值层（输出层）
    """

    def __init__(self, n_in, n_out):
        """
        :param n_in: 输入向量的长度
        :param n_out: 输出向量的长度（即该层神经元的个数）
        """
        self.n_in = n_in
        self.n_out = n_out
        # 初始化权重与偏差
        self.w = np.zeros((n_out, n_in), dtype=np.float32)
        self.b = np.zeros((n_out, 1), dtype=np.float32)

    def _softmax(self, z):
        z_exp = np.exp(z)
        return z_exp / np.sum(z_exp, axis=0)

    def propagate(self, input, **kwargs):
        """
        :param input: 上一层的输出值，为按列排布的二维numpy数组
        """
        self._input = input
        self.output = self._softmax(np.dot(self.w, input) + self.b)
        self.output_class = np.argmax(self.output, axis=0)

    def feedback(self, y):
        """
        对数似然代价函数

        :param y: 样本真实值，为按列排布的二维numpy数组
        """
        error = y - self.output
        self.input_g = np.dot(self.w.T, error)
        self.grad_w = np.zeros(self.w.shape, dtype=np.float32)
        i = 0; n = y.shape[1]
        while i < n: self.grad_w += np.outer(error[:, i], self._input[:, i]); i += 1
        self.grad_w /= n
        self.grad_b = np.mean(error, axis=1, keepdims=True)

    def accuracy(self, y):
        """
        计算当前网络在输入集合上的性能

        :param y: 样本真实值，为按列排布的二维numpy数组
        :return: 代价函数值和分类准确率
        """
        y_class = np.argmax(y, axis=0)
        cost = - np.mean(np.log(self.output[y_class, np.arange(y.shape[-1])]))
        accuracy =  100 * np.mean(y_class == self.output_class)
        return cost,accuracy


def dropout_layer(layer, p_dropout, dropout_mask=None):
    """
    弃权技术，用于删除层上的部分神经元

    :param layer: 待弃权的层
    :param p_dropout: 需删除的神经元的占比
    :param dropout_mask: （可选）弃权掩膜，用于强制指定需保留的神经元
    :return: 处理后的层，形状与处理前相同；若未提供dropout_mask参数，则同时返回保留神经元的索引
    """
    if isinstance(dropout_mask, type(None)):
        n = len(layer)
        if p_dropout == 0: return layer, np.arange(n)
        active_neural = np.random.choice(np.arange(n), int(n * (1 - p_dropout)))
        layer_dropout = np.zeros(layer.shape, dtype=np.float32)
        layer_dropout[active_neural , : ] = layer[active_neural , : ]
        return layer_dropout / (1.0 - p_dropout), active_neural
    else:
        if p_dropout == 0: return layer
        layer_dropout = np.zeros(layer.shape, dtype=np.float32)
        layer_dropout[dropout_mask, :] = layer[dropout_mask, :]
        return layer_dropout / (1.0 - p_dropout)


if __name__ == '__main__':
    training_data, validation_data, test_data = load_mnist()
    training_data_small = (training_data[0][: 5000, : ], training_data[1][: 5000, : ])
    test_data_small = (test_data[0][: 1000, :], test_data[1][: 1000, :])
    '''
    net1 = Network([ConvPoolLayer((20, 1, 5, 5), (1, 28, 28), pooling='max', activation_fn=relu),
                   ConvPoolLayer((40, 20, 5, 5), (20, 12, 12), pooling='max', activation_fn=relu),
                   FullyConnectedLayer(40*4*4, 1000, activation_fn=relu, p_dropout=0.5),
                   FullyConnectedLayer(1000, 1000, activation_fn=relu, p_dropout=0.5),
                   SoftmaxLayer(1000, 10)])
    '''
    net2 = Network([ConvPoolLayer((20, 1, 5, 5), (1, 28, 28), pooling='max', activation_fn=relu),
                    ConvPoolLayer((40, 20, 5, 5), (20, 12, 12), pooling='max', activation_fn=relu),
                    FullyConnectedLayer(40 * 4 * 4, 100, activation_fn=relu, p_dropout=0.0),
                    FullyConnectedLayer(100, 100, activation_fn=relu, p_dropout=0.0),
                    SoftmaxLayer(100, 10)])
    '''
    net = Network([FullyConnectedLayer(784, 30, activation_fn=sigmoid, p_dropout=0.0),
                   SoftmaxLayer(30, 10)])
    '''
    #net1.SGD(training_data, 200, 10, 0.002, 0.1, test_data)
    net2.SGD(training_data, 200, 10, 0.002, 0.1, test_data)


