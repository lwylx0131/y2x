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

# 和softmax回归唯一不同在于，多加一个全连接层作为隐藏层，隐藏层单元个数为256，并且使用ReLU函数作为激活函数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
    dl.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs)    
)

for params in net.parameters():
    nn.init.normal_(params, mean=0, std=0.01)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
num_epochs = 5
dl.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
'''
第1次正在训练.
epoch 1, loss 0.0032, train acc 0.696, test acc 0.801
第2次正在训练.
epoch 2, loss 0.0019, train acc 0.819, test acc 0.826
第3次正在训练.
epoch 3, loss 0.0017, train acc 0.843, test acc 0.820
第4次正在训练.
epoch 4, loss 0.0015, train acc 0.855, test acc 0.828
第5次正在训练.
epoch 5, loss 0.0015, train acc 0.863, test acc 0.807
'''

'''
在基于Fashion-MNIST数据集的实验中，在评价机器学习模型在训练数据集和测试集上时，发现当模型在训练数据集上更准确时，在测试集上却不一定更准确，为什么？
训练误差（training error）：模型在训练数据集上表现出的误差。
泛化误差（generalization error）：模型在任意一个测试数据样本上表现出的误差的期望，并常常通过测试数据集上的误差来近似。
计算训练误差和泛化误差可以使用之前的损失函数，如线性回归用到的平方损失函数和softmax回归用到的交叉熵损失函数。
'''