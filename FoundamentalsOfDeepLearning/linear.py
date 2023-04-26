#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from perceptron import Perceptron

# 定义线性单元激活函数（只有线性单元的激活函数和感知器的不同）
f = lambda x: x


class LinearUnit(Perceptron):
    '''调用Perceptron类，只需修改激活函数'''

    def __init__(self, input_num):
        '''初始化线性单元，将线性单元的激活函数传入感知器类'''
        Perceptron.__init__(self, input_num, f)


def get_training_dataset():
    '''输入数据和样本值'''
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    '''训练线性单元'''
    # 定义类，并且参数个数为1 年限
    lu = LinearUnit(1)
    input_vecs, labels = get_training_dataset()
    # 因为linearUnit直接调用的perceptron类，所以可以继承人家的函数
    # 迭代轮数10，学习率0.01
    lu.train(input_vecs, labels, 10, 0.01)
    return lu


def plot(linear_unit):
    import matplotlib.pyplot as plt
    imput_vecs, labels = get_training_dataset()  # 对已知样本值绘图
    fig = plt.figure()

    # 子图上为一行一列的图，序号为1

    ax = fig.add_subplot(111)  # type: ignore
    ax.scatter(list(map(lambda x: x[0], imput_vecs)), labels)  # 坑：注意map外加list生成列表
    weights = linear_unit.weights
    bias = linear_unit.bias
    x = range(0, 12, 1)
    y = list(map(lambda x: weights[0] * x + bias, x))  # 因为此次是一个权重参数（年） 所以取weights[0]
    ax.plot(x, y)
    plt.show()


if __name__ == '__main__':
    '''训练线性单元'''
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print(linear_unit)
    # 测试
    print('Work 3.4 years,monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years,monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years,monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years,monthly salary = %.2f' % linear_unit.predict([6.3]))
    plot(linear_unit)
