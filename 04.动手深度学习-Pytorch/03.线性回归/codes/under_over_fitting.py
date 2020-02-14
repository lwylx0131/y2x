# -*- encoding: utf-8 -*-

import torch
import numpy as np
import sys
sys.path.append('../..')
import dl_common_pytorch as dl
import torch.nn as nn

# 与线性回归相同，多项式函数拟合也使用平方损失函数。特别地，一阶多项式函数拟合又叫做线性函数拟合
# 因为高阶多项式函数模型参数更多，模型函数的选择空间更大，所以高阶多项式函数比低阶多项式函数的复杂度更高。
# 所以，高阶多项式函数比低阶多项式函数更容易在相同的训练数据集上得到更低的训练误差。
# 给定训练数据集，如果模型复杂度过低，很容易出现欠拟合；如果过高，很容易出现过拟合。
# 影响欠拟合和过拟合的另外一个重要因素是训练数据集的大小。
# 一般来说，如果训练数据集中样本数过少，特别是比模型参数数量（按元素计）更少时，过拟合更容易发生。
# 此外，泛化误差不会随训练数据集里样本数量增加而增大。

# 为理解模型复杂度和训练数据集大小对欠拟合和过拟合的影响，使用多项式函数拟合为例实验
# 生成多项式数据集：y = 1.2*x - 3.4*x^2 + 5.6*x^3 + 5 + e，其中噪声项e服从均值为0，标准差为0.01的正态分布。
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
# 1 按列追加拼接矩阵
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
'''
>>> a
tensor([[ 0.8746],
        [-1.8988],
        [ 1.1329],
        [ 0.6410],
        [-0.3390]])
>>> torch.cat((a, torch.pow(a, 2), torch.pow(a, 3)), 1)
tensor([[ 0.8746,  0.7649,  0.6689],
        [-1.8988,  3.6056, -6.8465],
        [ 1.1329,  1.2836,  1.4542],
        [ 0.6410,  0.4109,  0.2634],
        [-0.3390,  0.1149, -0.0390]])
'''
labels = true_w[0]*poly_features[:, 0] + true_w[1]*poly_features[:, 1] + true_w[2]*poly_features[:, 2] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

num_epochs = 100
loss = torch.nn.MSELoss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
        print('第%d次迭代：训练集损失值 = %f, 测试集损失值 = %f.' %(epoch+1, train_ls[-1], test_ls[-1]))
    print('weight: ', net.weight.data)
    print('bias: ', net.bias.data)

# 实验表明：这个模型的训练误差和在测试数据集的误差都比较低
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])
'''
weight:  tensor([[ 1.2004, -3.3983,  5.6003]])
bias:  tensor([4.9981])
'''