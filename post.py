# # # # # # # # # # # # # # # # # # # # # # # #
#
#    后处理模块
#
# # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from matplotlib import pyplot as plt
from time import sleep


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def item_print(tag, obj, newline=False, indent=2):
    if newline:
        text = '\n'.join(map(lambda s: ' ' * indent + s, str(obj).split('\n')))
        print('%s:\n%s' % (tag, text))
    else:
        print('%s: %s' % (tag, obj))


def interact():
    """
    模拟控制台交互
    """
    while True:
        cmd = input(">>> ")
        if cmd == 'exit':
            break
        try:
            value = eval(cmd)
        except SyntaxError:
            exec(cmd)
        else:
            if not isinstance(value, type(None)):
                print(value)


def plot(data, label, line=None, title=None, tag=None):
    """
    绘制带标签数据的散点图，可额外以参数定义一条直线
    """
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    ax = fig.add_subplot(111)
    if tag:
        ax.set_xlabel(tag[0])
        ax.set_ylabel(tag[1])
    ax.scatter(data[0, : ], data[1, : ], marker='o', c=label, cmap=plt.get_cmap('seismic'))
    if hasattr(line, '__iter__'):
        a, b, c = line.reshape(-1)
        if a * b <= 0:
            x_min = max(np.min(data[0, : ]), - (b * np.min(data[1, : ]) + c) / a)
            x_max = min(np.max(data[0, :]), - (b * np.max(data[1, :]) + c) / a)
        else:
            x_min = max(np.min(data[0, :]), - (b * np.max(data[1, :]) + c) / a)
            x_max = min(np.max(data[0, :]), - (b * np.min(data[1, :]) + c) / a)
        line_x = np.linspace(x_min, x_max, 2)
        line_y = - (a * line_x + c) / b
        ax.plot(line_x, line_y, 'y-.')
    plt.grid()
    plt.show()


class PlotAniNet:
    """
    以交互模式在同一子图中动态绘制多条曲线。

    用于：神经网络训练可视化
    """

    def __init__(self, title='', xlabel='x', ylabel='y', line_count=1, legend=None):
        self.fig, self.ax = plt.subplots()
        self.fig.suptitle(title)
        self.line = []
        self.xdata = []
        self.ydata = []
        self.n = line_count
        for i in range(line_count):
            self.line.append(self.ax.plot([], [], '.-')[0])
            self.xdata.append([])
            self.ydata.append([])
        self.ax.autoscale()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid()
        if legend:
            self.ax.legend(legend)
        self.fig.show()

    def add(self, ind, x, y):
        self.xdata[ind].append(x)
        self.ydata[ind].append(y)

    def update(self):
        i = 0
        while i < self.n:
            self.line[i].set_data(self.xdata[i], self.ydata[i])
            i += 1
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.001)


class PlotAniOptim:
    """
    以交互模式在不同子图中动态绘制曲线，子图可包含二元常量函数的云图。

    用于：优化算法可视化
    """

    def __init__(self, title, layout, subtitle):
        self.subtitle = subtitle
        self.fig = plt.figure(figsize=(12, 6))
        self.fig.suptitle(title)
        self.ax_list = []
        self.contour_list = []
        self.line_list = []
        for i in range(layout[0]):
            ax = self.fig.add_subplot(layout[1], layout[2], i + 1)
            ax.set_title(subtitle[i])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid()
            self.ax_list.append(ax)
            self.contour_list.append(None)
            line, = ax.plot([], [], 'r-o', linewidth='0.8', markersize='1')
            self.line_list.append(line)
        self.fig.tight_layout()
        self.fig.show()

    def plotFunc(self, i, func, xrange, yrange, nline=3):
        m = nline * 5
        x = np.linspace(*xrange, m)
        y = np.linspace(*yrange, m)
        X, Y = np.meshgrid(x, y)
        _x = np.vstack((X.reshape((1, -1)), Y.reshape((1, -1))))
        Z = func(_x).reshape((m, m))
        ax = self.ax_list[i]
        contour = self.contour_list[i]
        if contour:
            contour.remove()
        contour = ax.contourf(X, Y, Z, nline, cmap='RdBu_r', linestyles='dashed', zorder=1)
        self.contour_list.append(contour)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.autoscale_view()
        plt.pause(0.001)

    def addPoint(self, i, x, y):
        line = self.line_list[i]
        xdata, ydata = line.get_data()
        line.set_data(np.append(xdata, x), np.append(ydata, y))
        self.ax_list[i].set_title(f"{self.subtitle[i]} - iter {len(xdata)}")
        plt.pause(0.001)


def mat_scatter(matrix, label, title='Data Distribution', xlabel='x', ylabel='y', colors=None):
    """
    用于绘制som结果分布的散点图
    """
    if colors is None:
        colors = ['red', 'blue']
    fig, ax = plt.subplots()
    fig.suptitle(title)
    label = np.array(label)
    classes = np.unique(label)
    class_n = len(classes)
    m, n, count = matrix.shape
    X, Y = np.meshgrid(np.arange(n), np.arange(m))
    for i in range(class_n):
        matrix_i = np.sum(matrix[:, :, label == classes[i]], axis=2)
        data = 1e5 * matrix_i / (count * (m + n))
        ax.scatter(X.reshape(-1), Y.reshape(-1), s=data.reshape(-1),
                   marker='o', c=colors[classes[i] % class_n], alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.margins(0.2)
    ax.grid()
    fig.show()


class PlotAniSVM:
    """
    以交互模式绘制数据散点图和一条直线（或闭合曲线），还可对部分数据点进行标注

    用于：支持向量机训练可视化
    """

    def __init__(self, dataArr, labelArr, dim_1=0, dim_2=1, interval=0.01, divide=100):
        self.dataArr = np.array(dataArr)
        self.n, self.m = self.dataArr.shape
        self.labelArr = np.array(labelArr).flatten()
        self.t = interval
        self.dim(dim_1, dim_2, divide)
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.window.setWindowTitle('My Plotter')
        self._startPlot()

    def dim(self, dim_1, dim_2, divide):
        self.dim1 = dim_1
        lower = np.min(self.dataArr[dim_1, :])
        upper = np.max(self.dataArr[dim_1, :])
        x = np.linspace(1.2 * lower - 0.2 * upper, 1.2 * upper - 0.2 * lower, divide)
        self.dim2 = dim_2
        lower = np.min(self.dataArr[dim_2, :])
        upper = np.max(self.dataArr[dim_2, :])
        y = np.linspace(1.2 * lower - 0.2 * upper, 1.2 * upper - 0.2 * lower, divide)
        X, Y = np.meshgrid(x, y)
        dataArr = np.zeros((self.n, divide, divide))
        dataArr[dim_1, :, :] = X
        dataArr[dim_2, :, :] = Y
        dataArr = dataArr.reshape((self.n, -1))
        self._x = x
        self._X = X
        self._Y = Y
        self._shape = (divide, divide)
        self._dataArr = dataArr

    def _startPlot(self):
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlabel('$dim_%s$' % self.dim1)
        self.ax.set_ylabel('$dim_%s$' % self.dim2)
        self.sv, = self.ax.plot([], [], 'ro', markerfacecolor='none')
        self.line, = self.ax.plot([], [], c='b')
        self._contour = None
        plt.grid(True)
        self.count = 0

    def scatter(self):
        self.ax.cla()
        self._startPlot()
        self.ax.scatter(self.dataArr[self.dim1, :],
                        self.dataArr[self.dim2, :],
                        c=self.labelArr)
        self.ax.set_title("count: %i" % self.m)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        sleep(self.t)

    def plotLine(self, w, b, sv=None):
        self.count += 1
        self.line.set_xdata(self._x)
        self.line.set_ydata(- (w[self.dim1, 0] * self._x + b) / w[self.dim2, 0])
        if len(sv):
            self.sv.set_xdata(self.dataArr[self.dim1, sv])
            self.sv.set_ydata(self.dataArr[self.dim2, sv])
        self.ax.set_title("iteration: %i" % self.count)
        # self.ax.relim()
        # self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        sleep(self.t)

    def plotContour(self, classify, sv=None):
        self.count += 1
        labels = classify(self._dataArr).reshape(self._shape)
        if self._contour:
            self._contour.remove()
        self._contour = self.ax.contour(self._X, self._Y, labels, [0.], colors='b')
        if len(sv):
            self.sv.set_xdata(self.dataArr[self.dim1, sv])
            self.sv.set_ydata(self.dataArr[self.dim2, sv])
        self.ax.set_title("iteration: %i" % self.count)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        sleep(self.t)


if __name__ == '__main__':
    data = np.zeros((4,4,2))
    data[1,1,0] = 1
    data[2,3,1] = 1
    mat_scatter(data, [0,1])


