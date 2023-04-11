# -*- coding: UTF-8 -*-

from __future__ import print_function
from functools import reduce

class VectorOp(object):
  '''
  实现向量操作
  '''
  @staticmethod
  def dot(x,y):
    '''
    计算两个向量的内积
    '''
    # 将通过计算每个对应位置的元素的乘积向量list，将list中每个元素用reduce进行相加
    # reduce函数就是进行将list表中元素相加，例如，list=[1,2,3,4] reduce(lambda a,b:a+b,list,0.0)
    # 得到（（（1+2）+3）+4）
    # lambda函数进行运算的函数
    return reduce(lambda a,b:a+b,VectorOp.element_multiply(x,y),0.0)
  
  @staticmethod
  def element_multiply(x,y):
    '''
    将两个向量中的每个对应位置元素相乘
    '''
    # 首先将两个向量各个位置的元素打包zip到一起，得到[(x1,y1),(x2,y2),(x3,y3)]
    # 然后用map函数计算[x1*y1,x2*y2,x3*y3]
    # 得到的仍是一个向量list表（求两个向量对应位置相乘类似）
    # map函数进行就是map(f(),args)就是将参数给f（）进行计算 map的作用是进行迭代每个位置元素进行相应的操作
    return list(map(lambda x_y:x_y[0]*x_y[1],zip(x,y)))
  
  @staticmethod
  def element_add(x,y):
    '''
    将两个元素的对应位置上元素相加
    '''
    # 同element_multiply
    return list(map(lambda x_y:x_y[0]+x_y[1],zip(x,y)))
  
  @staticmethod
  def scala_multiply(v,s):
    '''
    将向量v中的每个元素和标量s（数）相乘
    '''
    return map(lambda e:e*s,v)    

class Perceptron(object):
  def __init__(self, input_num, activator):
    '''
    初始化感知器，设置输入参数(x个数)的个数，激活函数
    激活函数的类型为double
    需要最后得到权重向量和偏置值
    '''
    # 权重向量初始化为0（input_num个元素的数组）
    self.weights = [0.0]*input_num
    # 激活函数
    self.activator =activator
    # 偏置初始化为0
    self.bias = 0.0
  
  def __str__(self):
    '''
    打印学习到的权重、偏置项（运行完类，最后返回的是要得到的字符串，进行输出）
    '''
    return 'weights\t:%s\nbias\t:%f\n' % (self.weights,self.bias)
  def predict(self,input_vec):
    '''
    输入向量，输出感知器的激活函数计算结果
    '''
    # 原理式子：x1*w1+x2*w2+b_0 结果和0进行比较
    return self.activator(VectorOp.dot(input_vec,self.weights)+self.bias)
  
  def train(self,input_vecs,labels,iteration,rate):
    '''
    输入训练数据：一组向量，每个向量对应的label，训练轮数，学习率
    '''
    for i in range(iteration):
      self._one_iteration(input_vecs,labels,rate)
  
  def _one_iteration(self,input_vecs,labels,rate):
    '''
    一次迭代，把所有的训练数据过一遍（难点）
    '''
    # 把输入和输出打包到一起，成为样本的列表[(input_vec,label),(),...]
    # 每个训练样本是（input_vec,label）
    samples = zip(input_vecs,labels)
    # 对每个样本，按照传感器规则更新权重
    for (input_vec,label) in samples:
      # 计算感知器每次权重下的输出    
      output = self.predict(input_vec)
      # 更新权重
      self._update_weights(input_vec,output,label,rate)
  
  def _update_weights(self,input_vec,output,label,rate):
    '''
    按照感知器规则更新权重和偏置值（难点）
    '''
    # 计算本次的delta
    # 把input_vec[x1,x2,x3,...]向量中的每个值乘上delta，得到每个权重 
    # 最后把权重更新按元素加到原来的weights[w1,w2,w3,...]
    delta = label - output # 样本和输出的差值 
    # 更新权重 将样本和实际输出的差值乘以学习率，再与每次输入的值相乘，加到权重上（为啥这么写呢？？先学后边的）
    # ps：解答 第二章线性单元和梯度讲了运用求梯度下降最大的式子求权重参数[详看第二章式子3]
    self.weights = VectorOp.element_add(self.weights,VectorOp.scala_multiply(input_vec,rate*delta))
    # 更新bias=学习率乘以（样本和输出的差值）
    self.bias += rate*delta
  
def f(x):
  '''定义激活函数'''
  return 1 if x > 0 else 0
  
def get_and_training_dataset():
  '''基于and真值表构建训练数据'''
  # 构建训练数据
  # 输入向量列表
  input_vecs = [[0,0],[0,1],[1,0],[1,1]]
  # 期望输出列表 【0，0，0，1】，labels为已知样本结果
  labels = [0,0,0,1]
  return input_vecs,labels

def train_and_perceptron():
  '''使用and真值表训练感知器（这里返回的是一个类）'''
  # 创建感知器，输入参数为2（and是二元函数），激活函数为f
  p = Perceptron(2,f)
  # 训练，迭代10轮，学习速率为0.1
  input_vecs,labels = get_and_training_dataset()
  p.train(input_vecs,labels,10,0.1)
  return p

if __name__=='__main__':
  # 训练and感知器
  and_perceptron = train_and_perceptron()
  # 打印训练获得的权重 (ps:实际打印的是Percetron的__str__)
  print(and_perceptron)
  # test
  print('0 and 0 = %d' % and_perceptron.predict([0,0]))
  print('0 and 1 = %d' % and_perceptron.predict([0,1]))
  print('1 and 0 = %d' % and_perceptron.predict([1,0]))
  print('1 and 1 = %d' % and_perceptron.predict([1,1])) 
      
    