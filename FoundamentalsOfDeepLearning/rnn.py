# 循环神经网络
# 通过进行输入、循环的权重公式进行
# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from cnn import element_wise_op
from activators import ReluActivator, IdentityActivator
from functools import reduce


class RecurrentLayer(object):
    def __init__(self, input_width, state_width, activator, learning_rate):
        self.gradient = None
        self.gradient_list = None
        self.delta_list = None
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate
        self.times = 0  # 当前时刻初始化为t0
        self.state_list = []  # 保存各个时刻的state
        self.state_list.append(np.zeros((state_width, 1)))
        self.U = np.random.uniform(-1e-4, 1e-4, (state_width, input_width))  # 初始化U
        self.W = np.random.uniform(-1e-4, 1e-4, (state_width, state_width))  # 初始化W

    def forward(self, input_array):
        '''
        根据 st=f(Uxt+Wst-1) 进行计算
        :param input_array:
        :return:
        '''
        self.times += 1
        state = (np.dot(self.U, input_array) + np.dot(self.W, self.state_list[-1]))
        element_wise_op(state, self.activator.forward)
        self.state_list.append(state)

    def backward(self, sensitivity_array, activator):
        '''
        实现BPTT算法
        :param sensitivity_array:
        :param activator:
        :return:
        '''
        self.calc_delta(sensitivity_array, activator)
        self.calc_gradient()

    def update(self):
        '''
        按照梯度下降，更新权重
        :return:
        '''
        self.W -= self.learning_rate * self.gradient

    def calc_delta(self, sensitivity_array, activator):
        self.delta_list = []  # 用来保存各个时刻的误差项
        for i in range(self.times):
            self.delta_list.append(np.zeros((self.state_width, 1)))
        self.delta_list.append(sensitivity_array)
        # 迭代计算每个时刻的误差项
        for k in range(self.times - 1, 0, -1):
            self.calc_delta_k(k, activator)

    def calc_delta_k(self, k, activator):
        '''
        根据k+1时刻的delta计算k时刻的delta
        :param k:
        :param activator:
        :return:
        '''
        state = self.state_list[k + 1].copy()
        element_wise_op(self.state_list[k + 1], activator.backward)
        self.delta_list[k] = np.dot(np.dot(self.delta_list[k + 1].T, self.W), np.diag(state[:, 0])).T

    def calc_gradient(self):
        self.gradient_list = []  # 保存各个时刻的权重梯度
        for t in range(self.times + 1):
            self.gradient_list.append(np.zeros((self.state_width, self.state_width)))
        for t in range(self.times, 0, -1):
            self.calc_gradient_t(t)
        # 实际的梯度是各个时刻梯度之和
        self.gradient = reduce(
            lambda a, b: a + b, self.gradient_list, self.gradient_list[0]
        )  # [0]被初始化为0且没有被修改过

    def calc_gradient_t(self, t):
        '''
        计算每个时刻t权重的梯度
        :param t:
        :return:
        '''
        gradient = np.dot(self.delta_list[t], self.state_list[t - 1].T)
        self.gradient_list[t] = gradient

    def reset_state(self):
        self.times = 0
        self.state_list = []
        self.state_list.append(np.zeros((self.state_width, 1)))


def data_set():
    x = [
        np.array([[1], [2], [3]]),
        np.array([[2], [3], [4]])
    ]
    d = np.array([[1], [2]])
    return x, d


def gradient_check():
    '''
    梯度检查
    '''
    # 设计一个误差函数，取所有节点输出项之和

    error_function = lambda o: o.sum()

    r1 = RecurrentLayer(3, 2, IdentityActivator(), 1e-3)

    # 计算forward值
    x, d = data_set()
    r1.forward(x[0])
    r1.forward(x[1])

    # 计算sensitivity map
    sensitivity_array = np.ones(r1.state_list[-1].shape, dtype=np.float64)

    # 计算梯度
    r1.backward(sensitivity_array, IdentityActivator())

    # 检擦梯度
    epsilon = 10e-4
    for i in range(r1.W.shape[0]):
        for j in range(r1.W.shape[1]):
            r1.W[i, j] += epsilon
            r1.reset_state()
            r1.forward(x[0])
            r1.forward(x[1])
            err1 = error_function(r1.state_list[-1])
            r1.W[i, j] -= 2 * epsilon
            r1.reset_state()
            r1.forward(x[0])
            r1.forward(x[1])
            err2 = error_function(r1.state_list[-1])
            expect_grad = (err1 - err2) / (2 * epsilon)
            r1.W[i, j] += epsilon
            print('weights(%d,%d):expected - actural %f - %f' % (i, j, expect_grad, r1.gradient[i, j]))


def test():
    l = RecurrentLayer(3, 2, ReluActivator(), 1e-3)
    x, d = data_set()
    l.forward(x[0])
    l.forward(x[1])
    l.backward(d, ReluActivator())
    gradient_check()
    return l


if __name__ == '__main__':
    test()
