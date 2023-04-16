#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from functools import reduce
import random
from numpy import * # type: ignore

# 激活函数sigmoid
def sigmoid(inX):
  return 1.0/(1+exp(-inX))

# 节点类，负责记录和维护自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算
class Node(object):
  def __init__(self, layer_index,node_index):
    '''
    构造节点对象
    '''
    self.layer_index = layer_index # 节点所属层的编号
    self.node_index = node_index # 节点的编号
    self.downstream = [] #下游节点 (会有多个节点)存储的是connection
    self.upstream = [] #上游节点 (会有多个节点)存储的是connection
    self.output = 0 # 输出
    self.delta = 0

  def set_output(self,output):
    '''设置节点的输出值（节点是输入层时）'''
    self.output = output
  
  def append_downstream_connection(self,conn):
    '''添加一个到下游节点的连接'''
    self.downstream.append(conn)
  
  def append_upstream_connection(self,conn):
    '''添加一个到上游节点的连接'''
    self.upstream.append(conn) 
    
  def calc_output(self):
    '''根据式1：y=sigmoid(w*x)计算输出'''
    output = reduce(lambda ret,conn:ret+conn.upstream_node.output*conn.weight,self.upstream,0.0)
    self.output = sigmoid(output)  
  
  def calc_hidden_layer_delta(self):
    '''隐藏层时，根据4式：delta=a（1-a）求和（wk*deltak）求delta'''
    # 这里用到的是下游节点的delta值乘以
    downstream_delta = reduce(lambda ret,conn : ret + conn.downstream_node.delta*conn.weight,self.downstream,0.0)
    self.delta = self.output*(1-self.output)*downstream_delta
    
  def calc_output_layer_delta(self,label):
    '''输出层时，根据3式：delta=y*(1-y)*(t-y)，计算delta'''
    self.delta = self.output*(1-self.output)*(label-self.output)
  
  def __str__(self):
    '''打印node节点信息'''
    # %u：表示无符号的十进制数 \t表示4个空格
    node_str = '%u-%u: output:%f delta:%f' % (self.layer_index,self.node_index,self.output,self.delta)
    dowmstream_str = reduce(lambda ret,conn:ret+'\n\t'+str(conn),self.downstream,'')
    upstream_str = reduce(lambda ret,conn:ret+'\n\t'+str(conn),self.upstream,'') 
    return node_str + '\n\tdownstream:' + dowmstream_str +'\n\tupstream:' + upstream_str 
    
    
    
class ConstNode(object):
  def __init__(self,layer_index,node_index):
    '''构造节点对象（偏置值需要，此时要求节点输出恒为一）'''
    # 注意：当偏置节点输出恒为一时，此时它的值不受上游节点的影响，但是需要用到下游节点计算delta,影响下游节点
    self.layer_index = layer_index
    self.node_index = node_index
    self.downstream = [] # important 存储的是connection
    self.upstream = []
    self.output = 1
    self.delta = 0
  
  def append_upstream_connection(self,conn):
    '''添加一个到下游节点的连接'''
    self.upstream.append(conn)
 
  def append_downstream_connnection(self,conn):
    '''添加一个到下游节点的连接'''
    self.downstream.append(conn)
    
  def calc_hidden_layer_delta(self):
    '''节点属于隐藏层时，根据4式：delta=a（1-a）求和（wki*deltak） 求delta'''
    downstream_delta = reduce(lambda ret,conn : ret + conn.downstream_node.delta * conn.weight,self.downstream,0.0)
    self.delta = self.output * (1-self.output)*downstream_delta
  
  def __str__(self):
    '''打印节点信息'''
    node_str = '%u-%u: output: 1 delta:%f' % (self.layer_index,self.node_index,self.delta)
    dowmstream_str = reduce(lambda ret,conn:ret+'\n\t'+str(conn),self.downstream,'')
    upstream_str = reduce(lambda ret,conn:ret+'\n\t'+str(conn),self.upstream,'') 
    return node_str + '\n\tdownstream:' + dowmstream_str +'\n\tupstream:' + upstream_str 
  
  
  
class  Layer(object):
  def __init__(self,layer_index,node_count):
    '''
    初始化一层
    layer_index:层编号
    node_count:层所含的结点数
    '''
    self.layer_index = layer_index # 层标识
    self.nodes = [] # 层内所包含的节点，类型为nodes类的数组
    # 将非偏置值节点加入到这一层中
    for i in range(node_count):# 注意 for循环这里 最后是不包含node_count的
      self.nodes.append(Node(layer_index,i))
    # 将偏置值节点加入层中
    self.nodes.append(ConstNode(layer_index,node_count))
    
  def set_output(self,data):
    '''设置层的输出，当是输入层时会用到'''     
    data2 = list(data) 
    for i in range(len(data2)):
      self.nodes[i].set_output(data2[i])
      
  def calc_output(self):
    '''计算层的输出向量'''
    for node in self.nodes[:-1]: # [:-1]表示除了最后一个取全部 因为最后一个节点是偏置值节点 偏置值节点的输出永远为1
      node.calc_output()
  
  def dump(self):
    '''打印层的信息'''
    for node in self.nodes:
      print (node)
      
      
      
class Connection(object):
  '''主要记录连接的权重，以及这个连接所关联的上下游节点（其实就是图中的连线,那条边）''' 
  def __init__(self,upstream_node,downstream_node):
    '''
    初始化连接，权重初始化为是一个很小的随机数
    upstream_node 连接的上游节点值
    downstream_node 连接的下游节点值
    '''
    # 注意 误差是根据后边节点计算出的，而梯度是根据误差和前边节点求出的
    self.upstream_node = upstream_node # 连接的上游节点
    self.downstream_node = downstream_node # 连接的下游节点
    self.weight = random.uniform(-0.1,0.1)# 从-0.1到0.1中取一个实数
    self.gradient = 0.0 # 梯度 其实就是delta（由下游节点决定的）*ai（上游节点）
    
  def calc_gradient(self):
    '''计算当前的梯度'''
    # 梯度 = 下游节点的delta*上游节点值
    self.gradient = self.downstream_node.delta * self.upstream_node.output
    
  def get_gradient(self):
    '''获取当前梯度'''
    return self.gradient
  
  def update_weight(self,rate):
    '''根据梯度下降算法更新权重'''
    self.calc_gradient()
    self.weight += rate * self.gradient
    
  def __str__(self):
    '''打印节点的值'''
    return '(%u-%u)->(%u-%u) = %f' % (
      self.upstream_node.layer_index,
      self.upstream_node.node_index,
      self.downstream_node.layer_index,
      self.downstream_node.node_index,
      self.weight)

class Connections(object):
  '''提供对Connection的集合操作'''
  def __init__(self):
    self.connections = [] # connections是connection的数组
  
  def add_connection(self,connection):
    self.connections.append(connection)
    
  def dump(self):
    for conn in self.connections:
      print (conn)
      
class Network(object):
  def __init__(self,layers):
    '''
    初始化一个全连接神经网络
    layers:存放层节点
    '''
    self.connections = Connections()
    self.layers = [] # 数组，每个元素存放每层结点数
    layer_count = len(layers) # 表示层数
    # node_count = 0 # 节点总数
    # 初始化层数
    for i in range(layer_count):# 0到(层数-1)
      self.layers.append(Layer(i,layers[i]))
    # 初始化连接边
    for layer in range(layer_count-1):# 表示最后一层不包括进去
      connections = [Connection(upstream_node,downstream_node) 
                     for upstream_node in self.layers[layer].nodes
                     for downstream_node in self.layers[layer+1].nodes[:-1]]
      
      for conn in connections:
        self.connections.add_connection(conn)
        conn.upstream_node.append_upstream_connection(conn)
        conn.downstream_node.append_downstream_connection(conn)
        
  def train(self,labels,data_set,rate,iteration):
    '''
    训练神经网络
    label: 数组，训练的样本标签，每个元素是一个样本的标签
    data_set: 二维数组，训练样本特征
    '''
    for i in range(iteration):
      for d in range(len(data_set)):
        self.train_one_sample(labels[d],data_set[d],rate)
        
  def train_one_sample(self,label,sample,rate):
    '''内部函数，用一个样本训练网络'''
    self.predict(sample)
    self.calc_delta(label)
    self.update_weight(rate)
  
  def calc_delta(self,label):
    '''内部函数，计算每个节点的delta'''
    output_nodes = self.layers[-1].nodes # [-1]表示最后一层数据
    label2 = list(label)
    for i in range(len(label2)):
      output_nodes[i].calc_output_layer_delta(label2[i])
    for layer in self.layers[-2::-1]: # [-2::-1]表示 将layers列表反过来，然后除去之前的最后一元素
      for node in layer.nodes:
        node.calc_hidden_layer_delta()
        
  def update_weight(self,rate):
    '''内部函数，更新每个连接权重'''
    for layer in self.layers[:-1]:
      for node in layer.nodes:
        for conn in node.downstream:
          conn.update_weight(rate)
  
  def calc_gradient(self):
    '''计算每个连接的梯度'''
    for layer in self.layers[:-1]:
      for node in layer.nodes:
        for conn in node.downstream:
          conn.calc_gradient()
  
  def get_gradient(self,sample,label):
    '''
    获得网络在一个样本下，每个连接上的梯度
    label：样本标签
    sample：样本输入
    '''
    self.predict(sample)
    self.calc_delta(label)
    self.calc_gradient()
  
  def predict(self,sample):
    '''
    根据输入的样本预测输出值
    sanple：样本特征，网络输入向量
    '''
    self.layers[0].set_output(sample)
    for i in range(1,len(self.layers)):
      self.layers[i].calc_output()
    return map(lambda node:node.output,self.layers[-1].nodes[:-1])# [-1]最后一个元素，[:-1]除最后一个元素的所有元素
  
  def dump(self):
    '''打印网络信息'''
    for layer in self.layers:
      layer.dump()

class Normalizer(object):
  '''正则化???规则之后会学到'''
  def __init__(self):
    self.mask = [0x1,0x2,0x4,0x8,0x10,0x20,0x40,0x80]
  def norm(self,number): # 正则化 解决过拟合问题
    return map(lambda m:0.9 if number & m else 0.1,self.mask) 
  def denorm(self,vec): #降低正则化 解决欠拟合问题
    binary  = list(map(lambda i:1 if i>0.5 else 0,vec))
    mask2 = list(map(int,self.mask))
    for i in range(len(mask2)):
      binary[i] = binary[i]*mask2[i] 
    return reduce(lambda x,y: x+y,binary)

def mean_square_error(vec1,vec2):
  '''目标函数（损失函数）'''
  return  0.5*reduce(lambda a,b: a+b,map(lambda v: (v[0]-v[1])*(v[0]-v[1]),zip(vec1,vec2)))
    
def gradient_check(network,sample_feature,sample_label):
  '''
  梯度检查
  network:神经网络对象
  sample_feature:样本的特征
  sample_label: 样本标签
  '''
  # 计算网络误差(这里是一个函数)   这里\是换行的意思并且\后边不能有其他
  network_error = lambda vec1,vec2:\
    0.5*reduce(lambda a,b:a+b,map(lambda v:(v[0]-v[1])*(v[0]-v[1]),zip(vec1,vec2)))
  
  # 获取网络在当前样本下每个连接的梯度
  network.get_gradient(sample_feature,sample_label)
  
  # 对每个权重做梯度检查
  for conn in network.connections.connections:
    # 获取指定连接的梯度
    actual_gradient = conn.get_gradient()
    
    #增加一个很小的值，计算网络的误差
    epsilon = 0.0001
    conn.weight +=epsilon
    error1 = network_error(network.predict(sample_feature),sample_label)
    conn.weight -=2*epsilon
    error2 = network_error(network.predict(sample_feature),sample_label)
    expected_gradient = (error2-error1)/2/epsilon
    print('expected gradient:\t%f\nactural gradient:\t%f' % (expected_gradient,actual_gradient))
      
def train_data_set():
  normalizer = Normalizer()
  data_set = []
  labels = []
  for i in range(0,256,8):# 从0到256，不包括256，步长为8，每8个为一个间隔
    n = normalizer.norm(int(random.uniform(0,256)))# ???这里涉及正则化的之后会学到  其实就是正则化生成一些数据
    data_set.append(n)
    labels.append(n)
  return labels,data_set
  
def train(network):
  labels,data_set = train_data_set()
  network.train(labels,data_set,0.3,50)
  
def test(network,data):
  normalizer = Normalizer()
  norm_data = normalizer.norm(data)
  predict_data = network.predict(norm_data)
  print ('\ttestdata(%u)\tpredict(%u)' % (data,normalizer.denorm(predict_data)))
  
def correct_ratio(network):
  normalizer = Normalizer()
  correct = 0.0
  for i in range(256):
    if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
      correct += 1.0
  print('correct_ratio:%2f%%' % (correct/256*100))

def gradient_check_test():
  net = Network([2,2,2]) 
  sample_feature = [0.9,0.1]
  sample_label = [0.9,0.1]
  gradient_check(net,sample_feature,sample_label)
  
if __name__ == '__main__':
  net = Network([8,3,8])
  train(net)
  net.dump()
  correct_ratio(net)
    
      
      
      
      
      
        
      
    
      
    
    
    
  
    
      
  
      
      
        
  
  
  