

import numpy as np
from activators import ReluActivator, IdentityActivator


# 获取卷积区域
def get_patch(input_array, i, j, filter_width, filter_height, stride):
    '''
  输入数组中获取本次卷积区域
  自动适配输入为2D和3D的情况
  '''
    # 返回相应的数组类型或者大小
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[
               start_i:start_i + filter_height,
               start_j:start_j + filter_width]
    elif input_array.ndim == 3:
        return input_array[:,
               start_i: start_i + filter_height,
               start_j: start_j + filter_width]


# 获取一个2D区域的最大值所在的index
def get_max_index(array):
    '''
    获取一个2D区域的最大值所在的索引
    '''
    max_i = 0
    max_j = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_value = array[i, j]
                max_i, max_j = i, j
    return max_i, max_j


# 计算卷积
def conv(input_array, kernel_array, output_array, stride, bias):
    # '''计算卷积,自动适配输入为2D和3D的情况'''
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]  # [-1] is the last one
    kernel_height = kernel_array.shape[-2]  # [-2] is the second on the last side
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                                         get_patch(input_array, i, j, kernel_width,
                                                   kernel_height, stride) * kernel_array
                                 ).sum() + bias


# 为数组增加Zero padding
def padding(input_array, zp):
    # '''为数组增加zero padding 自动适配输入为2D和3D的情况'''
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            # 将数组赋0
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zp,
                input_width + 2 * zp
            ))
            padded_array[:,
            zp: zp + input_height,
            zp: zp + input_width
            ] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp
            ))
            padded_array[zp: zp + input_height, zp: zp + input_width] = input_array
            return padded_array


# 对numpy数组进行element wise操作 即矩阵对应位置的元素进行相应的op操作
def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


class Filter(object):
    def __init__(self, width, height, depth):
        # 随机生成三维数组filter的权重w
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return ('filter weightrs:\n%s\nbias:\n%s' % (repr(self.weights), repr(self.bias)))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


class ConvLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride, activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = \
            ConvLayer.calculate_output_size(
                self.input_width, filter_width, zero_padding, stride
            )
        self.output_height = \
            ConvLayer.calculate_output_size(
                self.input_height, filter_height, zero_padding, stride
            )
        self.output_array = np.zeros((
            self.filter_number, self.output_height, self.output_width
        ))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width, filter_height, self.channel_number))
        self.activator = activator
        self.learining_rate = learning_rate

    def forward(self, input_array):
        '''计算卷积层的输出'''
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array,
                 filter.get_weights(), self.output_array[f],
                 self.stride, filter.get_bias())
        element_wise_op(self.output_array, self.activator.forward)

    def backward(self, input_array, sensitivity_array, activator):
        '''
    计算传递给前一层的误差项 
    前一层的误差项保存在self.delta_array  ——w
    梯度保存在filter对象的weights_grad 梯度
    '''
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)  # delta
        self.bp_gradient(sensitivity_array)  # 'Ew

    def update(self):
        '''按照梯度下降，更新权重'''
        for filter in self.filters:
            filter.update(self.learining_rate)

    def bp_sensitivity_map(self, sensitivity_array, activator):
        '''
    计算传递到上一层的sensitivity map
    公式: delta(l-1) = delta(l)*w(l)·f'(net)
    '''
        # delta
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        # full卷积，对sensitivity map及逆行zero padding
        # 虽然源时输入的zero padding单元也会获得残差
        # 但是这个残差不需要继续向上传递，因此就不计算了
        expanded_width = expanded_array.shape[2]  # type: ignore
        zp = (self.input_width + self.filter_width - 1 - expanded_width) / 2
        padded_array = padding(expanded_array, zp)
        # 初始化delta_array,用于保存传递到上一层的
        # sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说，最终传递到上一层的
        # sensitivity map相当于所有的filter的
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter权重反转180度
            flipped_weights = np.array(
                map(
                    lambda i: np.rot90(i, 2), filter.get_weights()
                )
            )
            # 计算与一个filter对应的elta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)  # type: ignore
            self.delta_array += delta_array
        # 将计算的结果与激活函数的偏导数做element-wise操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        self.delta_array *= derivative_array

    def bp_gradient(self, sensitivity_array):
        # 处理卷积的步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        expanded_array = list(expanded_array)  # type: ignore
        for f in range(self.filter_number):
            # 计算每个权重的梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                self.padded_input_array = list(self.padded_input_array)
                conv(self.padded_input_array[d], expanded_array[f], filter.weights_grad[d], 1, 0)
            # 计算偏置项的梯度
            filter.bias_grad = expanded_array[f].sum()

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时，sensitivity map大小
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity map
        expand_array = np.zeros((
            depth, expanded_height, expanded_width
        ))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_height):  # type: ignore
            for j in range(self.output_width):  # type: ignore
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((
            self.channel_number, self.input_height, self.input_width
        ))

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size - filter_size + 2 * zero_padding) / stride + 1


class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = (input_width - filter_width) / self.stride + 1
        self.output_height = (input_height - filter_height) / self.stride + 1
        self.output_array = np.zeros((int(self.channel_number),
                                      int(self.output_height), int(self.output_width)))

    def forward(self, input_array):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (
                        get_patch(input_array[d], i, j, self.filter_width, self.
                                  filter_height, self.stride).max()  # type: ignore
                    )

    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(int(self.output_height)):
                for j in range(int(self.output_width)):
                    patch_array = get_patch(
                        input_array[d], i, j, self.filter_width, self.filter_height, self.stride
                    )
                    k, l = get_max_index(patch_array)
                    self.delta_array[d, i * self.stride + k, j * self.stride + l] = sensitivity_array[d, i, j]


def init_test():
    # 3 dimension
    a = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],
         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]]
    )
    b = np.array(
        [[[0, 1, 1],
          [2, 2, 2],
          [1, 0, 0]],
         [[1, 0, 2],
          [0, 0, 0],
          [1, 2, 1]]]
    )
    cl = ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)
    cl.filters[0].weights = np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],
         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],
         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]], dtype=np.float64)
    cl.filters[0].bias = 1
    cl.filters[1].weight = np.array(
        [[[1, 1 - 1],
          [-1, -1, 1],
          [0, -1, 1]],
         [[0, 1, 0],
          [-1, 0, -1],
          [-1, 1, 0]],
         [[-1, 0, 0],
          [-1, 0, 1],
          [-1, 0, 0]]], dtype=np.float64)
    return a, b, cl


def test():
    a, b, cl = init_test()
    cl.forward(a)
    print(cl.output_array)


def test_bp():
    a, b, cl = init_test()
    cl.backward(a, b, IdentityActivator())
    cl.update()
    print(cl.filters[0])
    print(cl.filters[1])


def gradient_check():
    '''梯度检查'''
    error_function = lambda o: o.sum()
    # 计算forward的值
    a, b, cl = init_test()
    cl.forward(a)

    # 求取sensitivity map
    sensitivity_array = np.ones(cl.output_array.shape, dtype=np.float64)

    # 计算梯度
    epsilon = 10e-4
    for d in range(cl.filters[0].weight_grad.shape[0]):
        for i in range(cl.filters[0].weight_grad.shape[1]):
            for j in range(cl.filters[0].weight_grad.shape[2]):
                cl.filters[0].weights[d, i, j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output_array)
                cl.filters[0].weights[d, i, j] -= 2 * epsilon
                cl.forward(a)
                err2 = error_function(cl.output_array)
                expect_grad = (err1 - err2) / (2 * epsilon)
                cl.filters[0].weights[d, i, j] += epsilon
                print('weights(%d,%d,%d):expected - actural %f - %f' %
                      (d, i, j, expect_grad, cl.filters[0].weights_grad[d, i, j]))


def init_pool_test():
    a = np.array(
        [[[1, 1, 2, 4],
          [5, 6, 7, 8],
          [3, 2, 1, 0],
          [1, 2, 3, 4]],
         [[0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 0, 1],
          [3, 4, 5, 6]]], dtype=np.float64)

    b = np.array(
        [[[1, 2],
          [2, 4]],
         [[3, 5],
          [8, 2]]], dtype=np.float64)

    mpl = MaxPoolingLayer(4, 4, 2, 2, 2, 2)
    return a, b, mpl


def test_pool():
    a, b, mpl = init_pool_test()
    mpl.forward(a)
    print('input array:\n%s\noutput array:\n%s' % (a, mpl.output_array))


def test_pool_bp():
    a, b, mpl = init_pool_test()
    mpl.backward(a, b)
    print('input array:\n%s\nsensitivity array:\n%s\ndelta array:\n%s' % (a, b, mpl.delta_array))


if __name__ == '__main__':
    test_pool_bp()