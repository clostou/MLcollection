# # # # # # # # # # # # # # # # # # # # # # # #
#
#    算法：贝叶斯优化
#
# # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import torch
import math
from time import sleep

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


torch.set_default_dtype(torch.float64)


class GPR:
    """
    高斯过程回归 (Gaussian Process Regression)
    """
    
    def __init__(self, initial_point, values):
        self.point = None
        self.value = None
        self.mean = None
        self.conv = None
        self._mean_args = (0.0, )
        self._kernel_args = (1.0, 1.1)
        
        self.set_point(initial_point, values)
        
        print("-"*20 + " GPR INFO " + "-"*20 + "\n\
  Mean Function:\n      constant_func (C = %.1f)\n\
  Convariance Function:\n      gauss_kernel (alpha = %.1f, sigma = %.2f)\n\
  Count of initial points: %i\n" % (
              *self._mean_args, *self._kernel_args, self.point.shape[1])
              + "-"*50)
    
    def mean_func(self, x):
        # 常数均值函数 (2D -> 1D)
        return np.ones(np.shape(x)[1]) * self._mean_args[0]
    
    def kernel_func(self, x1, x2, diag_only=False):
        # 高斯核函数 (2D -> 2D/1D)
        if diag_only:
            m = min(x1.shape[1], x2.shape[1])
            delta = np.empty(m, dtype=np.float64)
            for i in range(m):
                dx = x1[: , i] - x2[: , i]
                delta[i] = np.dot(dx, dx)
        else:
            delta = np.sum(np.power(x1.T[: , : , np.newaxis] - x2[np.newaxis, : , : ], 2), axis=1)
        return self._kernel_args[0] * np.exp(delta / (- 2 * self._kernel_args[1]**2))
    
    @staticmethod
    def point_input(points):
        # 将输入点集格式化为按列排布的二维数组
        shape = np.shape(points)
        if len(shape) == 0:
            pointArr = np.array([[points]], dtype=np.float64)
        elif len(shape) == 1:
            pointArr = np.array(points, dtype=np.float64).reshape(1, shape[0])
        elif len(shape) == 2:
            pointArr = np.array(points, dtype=np.float64)
        else:
            raise ValueError("Array of inputed points should have a dimension equaling or less than 2 .")
        return pointArr
    
    def set_point(self, points, values):
        self.point = self.point_input(points)
        self.value = np.array(values, dtype=np.float64).flatten()
        self.mean = self.mean_func(self.point)
        self.conv = self.kernel_func(self.point, self.point)
    
    def add_point(self, points, values):
        new_point = self.point_input(points)
        self.mean = np.append(self.mean, self.mean_func(new_point))
        conv_12 = self.kernel_func(self.point, new_point)
        conv_22 = self.kernel_func(new_point, new_point)
        self.conv = np.block([[self.conv, conv_12], [conv_12.T, conv_22]])
        self.point = np.concatnate(self.point, new_point)
        self.value = np.append(self.value, np.array(values, dtype=np.float64))
    
    def regress(self, points):
        predict_point = self.point_input(points)
        Kinv = np.linalg.inv(self.conv)
        k_cross = self.kernel_func(self.point, predict_point)
        k = self.kernel_func(predict_point, predict_point, diag_only=True)
        mu = k_cross.T.dot(Kinv).dot(self.value - self.mean) + self.mean_func(predict_point)
        sigma = k - np.einsum('ji, jk, ki -> i', k_cross, Kinv, k_cross)    # 爱因斯坦求和约定实现同时计算多个二次型
        return mu, sigma


class BOA:
    """
    贝叶斯优化（Bayesian Optimization Algorithm)

      *使用了Pytorch库的自动微分功能
    """

    def __init__(self, func, candidate_points, lower=None, upper=None, plot=False):
        # 函数句柄func输入为列向量，输出为该点的函数值
        self.func = func
        self.point = None    # 观测点
        self.value = None    # 函数值
        self.mean = None    # 观测点的均值
        self.conv = None    # 观测点的协方差
        self.step_n = 0

        self._mean_args = (0.0,)
        self._kernel_args = (10, 5)    # 高斯核的缩放因子及方差

        self.add_point(candidate_points)

        if not isinstance(lower, type(None)) and np.size(lower) == len(self.point):
            self.lower_x = torch.Tensor(lower).reshape((-1, 1))
        else:
            self.lower_x = torch.ones((len(self.point), 1)) * float('-inf')
        if not isinstance(upper, type(None)) and np.size(upper) == len(self.point):
            self.upper_x = torch.Tensor(upper).reshape((-1, 1))
        else:
            self.upper_x = torch.ones((len(self.point), 1)) * float('inf')

        self._with_plot = plot
        # 二维绘图变量
        self.fig = None
        self.ax_1 = None
        self.ax_2 = None
        self.line = [None] * 5
        # 三维绘图变量
        self.fig3D = None
        self.ax3D_1 = None
        self.ax3D_2 = None
        self.line3D = [None] * 4

    def mean_func(self, x):
        return torch.ones(x.shape[1]) * self._mean_args[0]

    def kernel_func(self, x1, x2, diag_only=False):
        if diag_only:
            m = min(x1.shape[1], x2.shape[1])
            delta = torch.empty(m)
            for i in range(m):
                dx = x1[: , i] - x2[: , i]
                delta[i] = torch.dot(dx, dx)
        else:
            delta = torch.sum(torch.pow(torch.unsqueeze(x1.T, 2) - torch.unsqueeze(x2, 0), 2), axis=1)
        return self._kernel_args[0] * torch.exp(delta / (- 2 * self._kernel_args[1] ** 2))

    def add_point(self, points):
        if isinstance(self.point, type(None)):
            self.point = torch.Tensor(points)
            self.value = torch.Tensor(self.func(points))
            self.mean = self.mean_func(self.point)
            self.conv = self.kernel_func(self.point, self.point)
        else:
            new_point = torch.Tensor(points)
            self.value = torch.cat((self.value, torch.Tensor(self.func(points))))
            self.mean = torch.cat((self.mean, self.mean_func(new_point)))
            conv_12 = self.kernel_func(self.point, new_point)
            conv_22 = self.kernel_func(new_point, new_point)
            self.conv = torch.cat((
                torch.cat((self.conv, conv_12), dim=1),
                torch.cat((conv_12.T, conv_22), dim=1)
            ))
            self.point = torch.cat((self.point, new_point), dim=1)

    def regress(self, points):
        predict_point = torch.Tensor(points)
        Kinv = torch.inverse(self.conv)
        k_cross = self.kernel_func(self.point, predict_point)
        k = self.kernel_func(predict_point, predict_point, diag_only=True)
        mu = torch.matmul(torch.mm(k_cross.T, Kinv), (self.value - self.mean)) + self.mean_func(predict_point)
        sigma = k - torch.einsum('ji, jk, ki -> i', k_cross, Kinv, k_cross)    # 爱因斯坦求和约定实现同时计算多个二次型
        return mu, sigma

    def _normal_pdf(self, x, loc=0, scale=1):
        return torch.exp(- torch.pow(x - loc, 2) / (2 * scale ** 2)) / (math.sqrt(2 * math.pi) * scale)

    def _normal_cdf(self, x, loc=0, scale=1):
        return 0.5 * (1 + torch.erf((x - loc) / (math.sqrt(2) * scale)))

    def EI(self, mean_x, var_x, eps=0.01):
        # 采集函数：期望改进 (expected improvement)
        # 特点：目标函数为一维函数时，易在点密集处产生接近0的平坦区域，难以优化
        #      目标函数为二维函数时，若初始点集仅包含一个点，也易在该点附近产生平坦区域
        u = (mean_x - (torch.max(self.value) + eps)) / var_x
        cdf = self._normal_cdf(u)
        pdf = self._normal_pdf(u)
        v = u * cdf + pdf
        EI = var_x * v
        # logEI = torch.log10(EI)
        return EI

    def UCB(self, mean_x, var_x, lamb=3):
        # 采集函数：上置信边界 (Upper Confidence Bound)
        # 特点：目标函数为一维函数时，易陷入局部最小值，难以优化
        UCB = mean_x + lamb * var_x
        return UCB

    def acquisition_func(self, x):
        mu = self.mean_func(x)
        k = self.kernel_func(self.point, x)
        Kinv = torch.inverse(self.conv)
        mean_x = torch.matmul(torch.mm(k.T, Kinv), self.value - self.mean) - mu
        var_x = torch.sqrt(self.kernel_func(x, x, diag_only=True) - torch.einsum('ji, jk, ki -> i', k, Kinv, k))
        #return self.UCB(mean_x, var_x)
        return self.EI(mean_x, var_x, 0.1)

    class Adam:
        ''' 一阶优化器：Adam (Adaptive Moment Estimation) '''

        def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
            self.alpha = alpha    # 全局学习率: 0.0001 ~ 0.1
            self.beta1 = beta1    # 动量项衰减系数（惯性）: 0.8 ~ 0.99
            self.beta2 = beta2    # 梯度历史衰减系数（阻尼）: 0.95 ~ 0.9999
            self.eps = eps
            self._m = None
            self._v = None
            self._k = 0

        def __call__(self, grad):
            if isinstance(self._m, type(None)):
                self._m = torch.zeros(grad.shape)
                self._v = torch.zeros(grad.shape)
            self._m = self.beta1 * self._m + (1 - self.beta1) * grad    # 移动指数加权平均
            self._v = self.beta2 * self._v + (1 - self.beta2) * grad ** 2
            self._k += 1
            alpha = self.alpha * math.sqrt(1 - self.beta2 ** self._k) / (1 - self.beta1 ** self._k)
            return alpha * self._m / (torch.sqrt(self._v) + self.eps)

    def clipX(self, x):
        # 值裁剪
        ind = x < self.lower_x
        x[ind] = self.lower_x[ind]
        ind = x > self.upper_x
        x[ind] = self.upper_x[ind]
        return x

    def step(self, init_x, precision=1e-6, max_iter=200):
        # 使用自写的Adam优化器
        optimizer = self.Adam(1, 0.8, 0.95, 1e-30)
        x = torch.Tensor(init_x).reshape((-1, 1))
        x.requires_grad_(True)    # 设为叶节点，构造计算图并获取梯度
        iter = 0
        min_iter = int(0.2 * max_iter)
        last_acquire = float('inf')
        while iter < max_iter:
            acquire = self.acquisition_func(x)
            acquire.backward()
            if iter >= min_iter and abs(acquire.item() - last_acquire) <= precision:
                break
            else:
                dx = optimizer(x.grad)
                x.grad.zero_()
                with torch.no_grad():
                    x += dx
                    self.clipX(x)
                last_acquire = acquire
                iter += 1
        x_star = x.detach().numpy()
        if iter == max_iter:
            print("Step %i ended with:" % (self.step_n + 1), x_star.flatten(), "(iterN: %i)" % iter)
        else:
            self.add_point(x_star)
            self.step_n += 1
            print("Step %i converged to:" % self.step_n, x_star.flatten(), "(iterN: %i)" % iter)

    def step2(self, init_x, iter_n=100):
        # 使用torch库内部的L-BFGS优化器
        x = torch.Tensor(init_x).reshape((-1, 1))
        x = torch.autograd.Variable(x, requires_grad=True)
        optimizer = torch.optim.LBFGS([x])
        iter = 0
        while iter < iter_n:
            optimizer.zero_grad()
            loss = - self.acquisition_func(x)
            loss.backward()
            optimizer.step(lambda : - self.acquisition_func(x))
            iter += 1
        x_star = x.detach().numpy()
        self.add_point(x_star)
        self.step_n += 1
        print("Step %i ended with:" % self.step_n, x_star.flatten(), "(iterN: %i)" % iter)

    def _simple_range(self, dim, count=200, margin=0.2):
        x_simple = self.point[dim, :]
        x_min = x_simple.min()
        x_max = x_simple.max()
        margin = 1.0 if x_min == x_max else margin * (x_max - x_min)
        x = torch.linspace(x_min - margin, x_max + margin, count)
        return x

    def plot(self, dim=0, plot_target=False):
        if isinstance(self.fig, type(None)):
            self.fig = plt.figure(figsize=(10, 6))
            self.fig.subplots_adjust(wspace=0.2, hspace=0.2)
            gs = gridspec.GridSpec(2, 1, height_ratios=(2, 1))
            # 子图1，用于高斯过程回归的可视化
            self.ax_1 = self.fig.add_subplot(gs[0, 0])
            self.line[0], = self.ax_1.plot([], [], 'r-', label="Target")
            self.line[1], = self.ax_1.plot([], [], 'r.',
                                           markerfacecolor=(0,0,0,0), markeredgecolor=(1,0,0.2,1),
                                           markersize=10, label="Observation")
            self.line[2], = self.ax_1.plot([], [], 'b--', label="Prediction")
            self.line[3] = self.ax_1.fill_between([], [], facecolor='blue', alpha=0.2, label="95% confidence interval")
            self.ax_1.set_ylabel("$f(\\vec x)$", fontsize=16)
            self.ax_1.legend()
            self.ax_1.grid()
            # 子图2，用于采集函数的可视化
            self.ax_2 = self.fig.add_subplot(gs[1, 0])
            self.line[4], = self.ax_2.plot([], [], 'k-', label="EI")
            self.ax_2.set_xlabel("$x_%i$" % dim, fontsize=16)
            self.ax_2.set_ylabel("Utility", fontsize=16)
            #self.ax_2.set_yscale('log')
            self.ax_2.legend()
            self.ax_2.grid()
            #self.fig.tight_layout()
            self.fig.show()
        self.fig.suptitle("Gaussian Process and Utility Function (Step %i)" % self.step_n, fontsize=18)
        x = self._simple_range(dim)
        x_ = torch.mean(self.point, dim=1, keepdim=True).repeat(1, len(x))    # 未可视化的维度使用均值替代
        x_[dim, : ] = x
        y, y_var = self.regress(x_)
        dy = 2 * torch.sqrt(y_var)
        acquires = self.acquisition_func(x_)
        if plot_target:
            self.line[0].set_data(x, self.func(x_.numpy()))    # 对于测试时使用的简单函数，可以直接绘制
        self.line[1].set_data(self.point[dim, : ], self.value)
        self.line[2].set_data(x, y)
        self.line[3].remove()
        self.line[3] = self.ax_1.fill_between(x, y + dy, y - dy, facecolor='blue',
                                              alpha=0.2, label="95% confidence interval")
        self.line[4].set_data(x, acquires)
        self.ax_1.relim()
        self.ax_1.autoscale_view()
        self.ax_2.relim()
        self.ax_2.autoscale_view()
        self.ax_2.set_xlim(self.ax_1.get_xlim())
        plt.pause(0.1)

    def plot3D(self, dim_1=0, dim_2=1, plot_target=False):
        if isinstance(self.fig3D, type(None)):
            self.fig3D = plt.figure(figsize=(10, 6))
            self.ax3D_1 = self.fig3D.add_subplot(121, projection='3d')
            self.ax3D_1.set_xlabel("$x_%i$" % dim_1, fontsize=16)
            self.ax3D_1.set_ylabel("$x_%i$" % dim_2, fontsize=16)
            self.ax3D_1.set_zlabel("$f(\\vec x)$", fontsize=16)
            self.ax3D_2 = self.fig3D.add_subplot(122, projection='3d')
            self.ax3D_2.set_xlabel("$x_%i$" % dim_1, fontsize=16)
            self.ax3D_2.set_ylabel("$x_%i$" % dim_2, fontsize=16)
            self.ax3D_2.set_zlabel("Utility", fontsize=16)
            self.fig3D.show()
        else:
            if plot_target:
                self.line3D[0].remove()
            self.line3D[1].remove()
            self.line3D[2].remove()
            self.line3D[3].remove()
        self.fig3D.suptitle("Gaussian Process and Utility Function (Step %i)" % self.step_n, fontsize=18)
        n = 30    # 3D绘图网格密度
        X, Y = torch.meshgrid(self._simple_range(dim_1, count=n), self._simple_range(dim_2, count=n), indexing='ij')
        x_ = torch.mean(self.point, dim=1, keepdim=True).repeat(1, n**2)
        x_[dim_1, :] = X.flatten()
        x_[dim_2, :] = Y.flatten()
        if plot_target:
            value = self.func(x_.numpy()).reshape((n, n))
            self.line3D[0] = self.ax3D_1.contourf(X, Y, value, zdir='z', offset=0, cmap='coolwarm')
        value_pred = self.regress(x_)[0].view((n, n))
        acquires = self.acquisition_func(x_).view((n, n))
        self.line3D[1] = self.ax3D_1.scatter3D(self.point[dim_1, : ], self.point[dim_2, : ],
                                               self.value, c='red', s=20)
        self.line3D[2] = self.ax3D_1.plot_wireframe(X, Y, value_pred, color='yellow', linewidth=0.8, alpha=0.6)
        self.line3D[3] = self.ax3D_2.plot_surface(X, Y, acquires, cmap='coolwarm')
        self.ax3D_1.relim()
        self.ax3D_1.autoscale_view()
        self.ax3D_2.relim()
        self.ax3D_2.autoscale_view()
        plt.pause(0.5)

    def search(self, init_x, precision=1e-6, max_iter=200):
        optimizer = self.Adam(1, 0.8, 0.95, 1e-30)
        x = torch.Tensor(init_x).reshape((-1, 1))
        x.requires_grad_(True)
        iter = 0
        last_value_pred = float('inf')
        while iter < max_iter:
            value_pred = self.regress(x)[0]
            value_pred.backward()
            if abs(value_pred.item() - last_value_pred) <= precision:
                break
            else:
                dx = optimizer(x.grad)
                x.grad.zero_()
                with torch.no_grad():
                    x += dx
                    self.clipX(x)
                last_value_pred = value_pred
                iter += 1
        x_star = x.detach().numpy()
        if iter == max_iter:
            print("Max iteration reached when searching!")
        else:
            print("Extreme value of GPR with %i points:" % self.point.shape[1], x_star.flatten())
        return x_star


def GPR_test():
    f = lambda x: x * np.sin(x)
    x_simple = np.array([1, 3, 5, 6, 7, 8])
    y_simple = f(x_simple)
    
    x = np.linspace(0, 10, 100)
    y = f(x)
    y_pred, y_var = GPR(x_simple, y_simple).regress(x)
    dy = 2 * np.sqrt(y_var)
    #print(y_pred, y_var, sep='\n')
    
    plt.plot(x, y, 'r-', label="f(x) = xsin(x)")
    plt.plot(x_simple, y_simple, 'r.', markerfacecolor=(0,0,0,0), markeredgecolor=(1,0,0.2,1),
             markersize=10, label="Observation")
    plt.plot(x, y_pred, 'b--', label="Prediction")
    plt.fill_between(x, y_pred + dy, y_pred - dy, alpha=0.3, label="95% confidence interval")
    
    plt.title("Gaussian Process Regression")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    #plt.ylim((-10, 20))
    plt.legend()
    plt.grid()
    plt.show()


def test_func(x):
    # -8 ~ 20
    x = x[0, : ]
    return - 0.001 * np.power(x, 4) + 0.02 * np.power(x, 3) + \
        0.01 * np.power(x, 2) - 0.8 * x + 2 * np.cos(x + 1) + 5


def test_func2(x):
    # -10 ~ 10
    x = x[: 2, : ]
    gauss = lambda mu, sigma: np.exp(- np.sum(np.power(x - mu, 2), axis=0) / sigma ** 2)
    return 2 * gauss(np.array([[4], [2]]), 5) + 1.8 * gauss(np.array([[-8], [-5]]), 6) + \
        1.5 * gauss(np.array([[-8], [8]]), 8) + 1.5 * gauss(np.array([[6], [8]]), 8) + \
        1.2 * gauss(np.array([[8], [-6]]), 6) + 1.2 * gauss(np.array([[1], [-8]]), 6)


if __name__ == '__main__':
    GPR_test()
    #boa = BOA(test_func, np.array([-6]).reshape(1, -1), plot=True)
    #boa.plot(plot_target=True)
    #boa.step([0])
    #boa = BOA(test_func2, np.array([-10, -10, 10, 8]).reshape((2, -1), order='F'), upper=[float('inf'), 0], plot=True)
    '''boa.plot3D(plot_target=True)
    sleep(5)
    print(123)
    sleep(1)
    for i in range(10):
        sleep(1)
        boa.step([0, 0])
        boa.plot3D(plot_target=True)
    boa.search([0, 0])'''


