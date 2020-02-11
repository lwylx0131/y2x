# -*- encoding: utf-8 -*-

import torch
import numpy as np

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))
#print(features[0], labels[0])

# 使用pytorch提供的data包来读取数据
import torch.utils.data as torchData

batch_size = 10
# 将训练数据的特征和标签组合
dataSet = torchData.TensorDataset(features, labels)
# 随机读取小批量数据
dataIter = torchData.DataLoader(dataSet, batch_size, shuffle=True)

# 使用pytorch定义线性回归 
# neural networks 神经网络
import torch.nn as nn 

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y	

net = LinearNet(num_inputs)
print(net) # 打印网络结构

# 查看模型所有的可学习参数
#for param in net.parameters():
#    print(param)		

'''
LinearNet(
  (linear): Linear(in_features=2, out_features=1, bias=True)
)
Parameter containing:
tensor([[-0.5042, -0.4775]], requires_grad=True)
Parameter containing:
tensor([-0.0988], requires_grad=True)
'''

# 使用Sequential搭建网络，这是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加到计算图中
#net = nn.Sequential(nn.Linear(num_inputs, 1))

net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
#net.add_module()

#import collections.OrderedDict
#net = nn.Sequential(OrderedDict([('linear', nn.Linear(num_inputs, 1))]))

print(net)
#print(net[0]) 
'''
LinearNet(
  (linear): Linear(in_features=2, out_features=1, bias=True)
)
Sequential(
  (linear): Linear(in_features=2, out_features=1, bias=True)
)
'''

print('---网络模型参数初始化之前---')
for param in net.parameters():
    print(param)		

# 初始化模型参数，将权重参数每个元素初始化为随机采样于均值为0，标准差为0.01的正态分布
nn.init.normal_(net[0].weight, mean=0.0, std=0.01)
# 偏差初始化为0
nn.init.constant_(net[0].bias, val=0.0)

print('---网络模型参数初始化之后---')
for param in net.parameters():
    print(param)		
'''
---网络模型参数初始化之前---
Parameter containing:
tensor([[ 0.0828, -0.3488]], requires_grad=True)
Parameter containing:
tensor([-0.0739], requires_grad=True)
---网络模型参数初始化之后---
Parameter containing:
tensor([[-0.0241, -0.0059]], requires_grad=True)
Parameter containing:
tensor([0.], requires_grad=True)
'''

# 定义损失函数
loss = nn.MSELoss()

# 定义一个学习率为0.03的小批量随机梯度下降(SGD)为优化算法
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
#print(optimizer)

# 调整学习率为之前的0.1倍
#for param_group in optimizer.param_groups:
#    param_group['lr'] *= 0.1
	
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in dataIter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        # 梯度清零，等价net.zero_grad()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('第%d次训练完毕，其loss为: %f' %(epoch, l.item()))