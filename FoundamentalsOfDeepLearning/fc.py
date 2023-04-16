import random
import numpy as np
from backpropagation import Network
from activators import SigmoidActivator,IdentityActivator
from functools import reduce

# 全连接层实现类
class FullConnectedLayer(object):
  def __init__(self,input_size,output_size,activator):
    '''
    构造函数
    input_size:本层输入向量的维度
    output_size:本层输出向量的维度
    activator:激活函数
    '''
    self.input_size = input_size
    self.output_size = output_size
    self.activator = activator
    # 权重数组w
    self.w = np.random.uniform(-0.1,0.1,(output_size,input_size))  
    # 偏置项
    self.b = np.zeros((output_size,1))
    # 输出向量
    self.output = np.zeros((output_size,1))
    
  def forward(self,input_array):
    '''向前计算'''
    self.input = input_array # 输入向量
    self.output = self.activator.forward(np.dot(self.w,input_array)+self.b)
    
  def backward(self,delta_array):
    '''反向计算w和b的梯度'''
    self.delta = self.activator.backward(self.input)*np.dot(self.w.T,delta_array)
    self.w_grad = np.dot(delta_array,self.input.T)
    self.b_grad = delta_array
    
  def update(self,learning_rate):
    '''使用梯度下降更新权重'''
    self.w += learning_rate*self.w_grad
    self.b += learning_rate*self.b_grad
    
  def dump(self):
    print('w: %s\nb:%s' % (self.w,self.b))
   
class Nerwork(object):
  def __init__(self,layers):
    '''构造函数'''
    self.layers = []
    for i in range(len(layers)-1):
      self.layers.append(FullConnectedLayer(layers[i],layers[i+1],SigmoidActivator()))
    
  def predict(self,sample):
    '''使用神经网络实现预测'''
    output = sample
    for layer in self.layers:
      layer.forward(output)
      output = layer.output
    return output
             
  def train(self,labels,data_set,rate,epoch):
    '''训练函数'''
    for i in range(epoch):
      for d in range(len(data_set)):
        self.train_one_sample(labels[d],data_set[d],rate)
        
  def train_one_sample(self,label,sample,rate):
    self.predict(sample)
    self.calc_gradient(label)
    self.update_weight(rate)
    
  def calc_gradient(self,label):
    delta = self.layers[-1].activator.backward(self.layers[-1].output)*(label-self.layers[-1].output)
    for layer in self.layers[::-1]:
      layer.backward(delta)
      delta = layer.delta
    return delta
  
  def update_weight(self,rate):
    for layer in self.layers:
      layer.update(rate)
  
  def dump(self):
    for layer in self.layers:
      layer.dump()
        
  def loss(self,output,label):
    return 0.5*((label-output)*(label-output)).sum()
  
  def gradient_check(self,sample_feature,sample_label):
    '''梯度检查'''
    # 获取网络在当前样本下每个连接的梯度
    self.predict(sample_feature)
    self.calc_gradient(sample_label)
    
    # 梯度检查
    epsilon = 10e-4
    for fc in self.layers:
      for i in range(fc.w.shape[0]):
        for j in range(fc.w.shape[1]):
          fc.w[i,j] += epsilon
          output = self.predict(sample_feature)
          err1 = self.loss(sample_feature,output)
          fc.w[i,j] -= 2*epsilon
          output = self.predict(sample_feature)
          err2 = self.loss(sample_label,output)
          expect_grad = (err1-err2)/(2*epsilon)
          fc.w[i,j] += epsilon
          print('weights(%d,%d):expected - actural %.4e - %.4e' % (i,j,expect_grad,fc.w_grad[i,j]))

# from backpropagation import train_data_set
   
def transpose(args):
  return map(lambda arg:map(lambda line:np.array(line).reshape(len(line),1),arg),args)


class Normalizer(object):
  def __init__(self):
    self.mask = [0x1,0x2,0x4,0x8,0x10,0x20,0x40,0x80]
    
  def norm(self,number):
    data = map(lambda m:0.9 if number & m else 0.1,self.mask)
    return np.array(data).reshape(8,1)
  
  def denorm(self,vec):
    binary = map(lambda i : 1 if i>0.5 else 0,vec[:,0])
    binary = list(binary)
    for i in range(len(list(self.mask))):
      binary[i] = binary[i]*self.mask[i]
    return reduce(lambda x,y: x + y, binary)

def train_data_set():
  normalizer = Normalizer()
  data_set = []
  labels = []
  for i in range(0,256):
    n = normalizer.norm(i)
    data_set.append(n)
    labels.append(n)
  return labels,data_set

def correct_ratio(network):
  normalizer = Normalizer()
  correct = 0.0
  for i in range(256):
    if normalizer.denorm(network.predict(normalizer.norm(i)))==i:
      correct+=1.0
  print('correct_ratio:%.2f%%' % (correct/256*100))
  
def test():
  labels,data_set = transpose(train_data_set())
  net = Nerwork([8,3,8])
  rate = 0.5
  mini_batch = 20
  epoch = 10
  for i in range(epoch):
    net.train(labels,data_set,rate,mini_batch)
    labels = list(labels)
    data_set = list(data_set)
    print('after epoch %d loss:%f' % ((i+1),net.loss(labels[-1],net.predict(data_set[-1]))))
    rate /= 2
  correct_ratio(net)

def gradient_check():
  '''梯度检查'''
  labels,data_set = transpose(train_data_set())
  labels = list(labels)
  data_set = list(data_set)
  net = Network([8,3,8])
  net.get_gradient(data_set[0],labels[0]) 
  return net
     
    
      
      
  
           
      
      
           
       
      