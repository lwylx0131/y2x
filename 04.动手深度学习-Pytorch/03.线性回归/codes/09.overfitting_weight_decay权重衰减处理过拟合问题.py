# -*- encoding: utf-8 -*-

import torch
import numpy as np
import sys
sys.path.append('../..')
import dl_common_pytorch as dl
import torch.nn as nn

'''
过拟合现象中，即模型的训练误差远小于它在测试集上的误差，虽然增大训练数据集可能会减轻过拟合，但是获取额外的训练数据往往代价高昂。
以下将应对过拟合问题使用的常用方法：权重衰减（weight decay）

权重衰减等价于L2范数正则化。正则化通过为模型损失函数添加惩罚项使学出的模型参数值较小，是应对过拟合的常用手段。

L2范数正则化在模型原损失函数基础上添加L2范数惩罚项，从而得到训练所需要最小化的函数。
L2范数惩罚项指的是模型权重参数每个元素的平方和与一个正的常数的乘积。比如以线性回归损失函数为例：
l(w1, w2, b) = 1/n 求和符号 1/2(x1w1 + x2w2 + b - y)^2
将权重参数W = [w1, w2]表示，带有L2范数惩罚项的新损失函数为：
l(w1, w2, b) + (r/2n)||W||^2
其中超参数r>0。当权重参数均为0时，惩罚项最小。当r较大时，惩罚项在损失函数中的比重较大，这通常会使学到的权重参数的元素较接近0.
当r设为0时，惩罚项完全不起作用。上式中L2范数平方||W||^2展开后得到w1^2 + w2^2。有了L2范数惩罚项后，在小批量随机梯度下降中，将线性回归权重w1和w2的迭代方式更改为：
可见，L2范数正则化令权重w1和w2先自乘小于1的数，再减去不含惩罚项的梯度。因此，L2范数正则化又叫权重衰减。
权重衰减通过惩罚绝对值较大的模型参数为需要学习的模型增加了限制，这可能对过拟合有效。
'''

# 实验：以高维线性回归为例来引入一个过拟合问题，并使用权重衰减来应对过拟合。设数据样本特征维度为p：
# y = 0.05 + 求和(i->p) 0.01xi + e

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 从0开始实现权重衰减方法，通过再目标函数后添加L2范数惩罚项来实现权重衰减
# 初始化模型参数，定义随机初始化模型参数的函数。该函数为每个函数都附上梯度
def init_params():
    W = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [W, b]

# 定义L2范数惩罚项，此处只惩罚模型的权重参数
def l2_penalty(W):
    return (W ** 2).sum() / 2

# 训练数据集和测试数据集，并在计算最终的损失函数时添加L2范数惩罚项
batch_size, num_epochs, lr = 1, 100, 0.003
net = dl.linreq
loss = dl.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot(lambd):
    W, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加L2范数惩罚项
            l = loss(net(X, W, b), y) + lambd * l2_penalty(W)
            l = l.sum()
            
            if W.grad is not None:
                W.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            dl.sgd([W, b], lr, batch_size)
        train_ls.append(loss(net(train_features, W, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, W, b), test_labels).mean().item())
    print("最后一次训练集损失值为: %.3f, 测试集损失值为: %.3f, L2 norm of W: %f." %(train_ls[-1], test_ls[-1], W.norm().item()))

# 训练并测试高维线性回归模型
# 当lambd设为0时，即没有使用权重衰减，结果训练误差远小于测试集上的误差，这是典型的过拟合现象
fit_and_plot(lambd=0)
'''
最后一次训练集损失值为: 0.000, 测试集损失值为: 88.312, L2 norm of W: 13.665776.
'''
# 当使用权重衰选，即lambd不为0，可以看出训练误差虽然有所提高，但是测试集上的误差有所下降，过拟合现象得到一定程度的缓解。
# 另外，权重参数的L2范数比不使用权重衰减时更小，此时的权重参数更接近0。
fit_and_plot(lambd=3)
'''
最后一次训练集损失值为: 0.000, 测试集损失值为: 0.010, L2 norm of W: 0.041722.
'''

# 使用PyTorch框架简洁实现：直接在构造优化器实例时通过weight_decay参数来指定权重衰减超参数。
# 默认情况下，PyTorch会对权重和偏差同时衰减。我们可以分别对权重和偏差构造优化器实例，从而只对权重衰减。
def fit_and_plot_pytorch(wd):
    # 对权重参数衰减，权重名称一般是以weight结尾
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    # 对权重参数衰减
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd)
    # 不对偏差参数衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)
	
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            l.backward()
            
            # 对两个optimizer实例分别调用step函数，从而分别更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    print("[pytorch]最后一次训练集损失值为: %.3f, 测试集损失值为: %.3f, L2 norm of W: %f." %(train_ls[-1], test_ls[-1], net.weight.data.norm().item()))
fit_and_plot_pytorch(0)
fit_and_plot_pytorch(3)
'''
最后一次训练集损失值为: 0.000, 测试集损失值为: 99.505, L2 norm of W: 13.098402.
最后一次训练集损失值为: 0.000, 测试集损失值为: 0.015, L2 norm of W: 0.035413.
[pytorch]最后一次训练集损失值为: 0.000, 测试集损失值为: 85.904, L2 norm of W: 13.225693.
[pytorch]最后一次训练集损失值为: 0.002, 测试集损失值为: 0.076, L2 norm of W: 0.088188.
'''

'''
1）正则化通过为模型损失函数添加惩罚项使学出的模型参数值较小，是应对拟合的常用手段。
2）权重衰减等价于L2范数正则化，通常会使学到的权重参数的元素较接近0。
3）权重衰减可以通过优化器中的weight_decay超参数来指定。
4）可以定义多个优化器实例对不同的模型参数使用不同的迭代方式。
'''