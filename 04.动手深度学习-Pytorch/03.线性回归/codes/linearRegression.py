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
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

# 将权重初始化均值为0，标准差为0.01的正态随机数，偏差则初始化为0
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
# 由于之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此设置梯度requires_grad=True属性
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)