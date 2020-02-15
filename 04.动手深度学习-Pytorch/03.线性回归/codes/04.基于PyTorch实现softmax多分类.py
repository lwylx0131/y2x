# -*- encoding: utf-8 -*-

import torch
import numpy as np
import sys
sys.path.append('../..')
import dl_common_pytorch as dl
import torch.nn as nn

# 使用Fashion-MNIST数据集
batch_size = 256
train_iter, test_iter = dl.load_data_fashion_mnist(batch_size)

# 数据返回的每个batch样本X的形状为(batch_size, 1, 28, 28)，所以先用view()将X的形状转换成(batch_size, 784)送入全连接层
num_inputs = 784
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    
    def forward(self, x): # x: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y

net = LinearNet(num_inputs, num_outputs)
from collections import OrderedDict

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

net = nn.Sequential(OrderedDict([
    ('flatten', FlattenLayer()),
    ('linear', nn.Linear(num_inputs, num_outputs))
]))

# 使用均值为0，标准差为0.01的正态随机初始化模型的权重参数
nn.init.normal_(net.linear.weight, mean=0.0, std=0.01)
nn.init.constant_(net.linear.bias, val=0.0)

# pytorch提供一个包括softmax运算和交叉熵损失计算的函数，它的数值稳定性比我们自定义实现的更好
loss = nn.CrossEntropyLoss()

# 使用学习率为0.1的小批量随机梯度下降作为优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型
num_epochs = 5
dl.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)