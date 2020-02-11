# -*- encoding: utf-8 -*-

import torch
from time import time

a = torch.ones(1000)
b = torch.ones(1000)

start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)

start = time()
d = a + b
print(time() - start)

a = torch.ones(3)
b = 10
print(a + b)


import numpy as np
from matplotlib import pyplot as plt
import random

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))
print(features[0], labels[0])

import sys
sys.path.append('../..')
from dl_common_pytorch import *

set_figsize()
#plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
#plt.show()

batch_size = 10
##for X, y in data_iter(batch_size, features, labels):
##    print(X, y)
##    break

# 将权重初始化均值为0，标准差为0.01的正态随机数，偏差则初始化为0
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
# 由于之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此设置梯度requires_grad=True属性
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

'''训练模型'''
# 其实迭代周期num_epochs和学习率lr都是超参数
# 并且在实践中，大多超参数都需要通过反复试错来不断调节超参数
lr = 0.03
num_epochs = 3
net = linreq
loss = squared_loss
for epoch in range(num_epochs):
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次
    # x和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        # l是有关小批量X和y的损失
		# 由于变量l不是一个标量，为了避免张量对张量求导，所以需要将l使用.sum()变成一个标量，再运行l.backward()得到该变量有关模型参数的梯度
        l = loss(net(X, w, b), y).sum()
        # 小批量的损失对模型参数求梯度
        l.backward()
        # 使用小批量随机梯度下降迭代模型参数
        sgd([w, b], lr, batch_size)
		# 每次遍历需要将参数梯度清零，否则会累加梯度
        w.grad.data.zero_()
        b.grad.data.zero_()
	# 最终将以上迭代训练的wb模型参数来验证整个数据集的损失，如果损失慢慢变小，说明模型参数wb有效
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' %(epoch + 1, train_l.mean().item()))
# 最终将训练的模型参数wb与真实的wb对比，应该很接近
print(true_w, '\n', w)
print(true_b, '\n', b)