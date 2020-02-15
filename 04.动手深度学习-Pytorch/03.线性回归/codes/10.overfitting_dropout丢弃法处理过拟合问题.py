# -*- encoding: utf-8 -*-

import torch
import numpy as np
import sys
sys.path.append('../..')
import dl_common_pytorch as dl
import torch.nn as nn

'''
除了权重衰减可以处理过拟合问题，在深度学习模型还经常使用丢弃法（dropout）应对过拟合问题。
丢弃法有一些不同的变体，下面以倒置丢弃法（inverted dropout）为例。

多层感知机：含有一个单隐藏层的多层感知机，其中数人个数为4，隐藏单元个数为5，且隐藏单元hi(i=1~5)的计算表达式：
hi = Q(x1w1i + x2w2i + x3w3i + x4w4i + bi)
其中Q为激活函数，x1x2x3x4是输入，隐藏单元i的权重参数为w1i,w2i,w3i,w4i（第i个隐藏单元权重参数，一个有5个隐藏单元），偏差参数为bi。
当对该隐藏层使用丢弃法时，该层的隐藏单元将有一定概率被丢弃掉。设丢弃概率wiep，那么有p的概率hi会被清零，有1-p的概率hi会除以1-p做拉伸。
丢弃概率是丢弃法的超参数。具体来说，设随机变量ei为0和1的概率分别为p和1-p，使用丢弃法时我们计算新的隐藏单元h'i:
h'i = (ei / (1-p)) * hi
由于E(ei) = 1 - p，因此：
E(h'i) = (E(ei) / (1-p)) * hi = hi，即丢弃法不改变其输入的期望值
'''
def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都抛弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    # 随机一个与X一致矩阵，对比里面的每个元素是否小于keep_prob，然后将符合条件的使用.float()转换为1
    mask = (torch.randn(X.shape) < keep_prob).float()
    # 将mask与X相乘，就会去掉一些隐藏的单元，剩下的元素再除以keep_prob
    return mask * X / keep_prob

# 丢弃概率分别为0、0.5和1
X = torch.arange(16).view(2, 8)
print(dropout(X, 0))
print(dropout(X, 0.5))
print(dropout(X, 1.0))

# 定义一个包含两个隐藏层的多层感知机，其中两个隐藏层的输出个数都是256
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)
params = [W1, b1, W2, b2, W3, b3]

# 定义模型将全连接层和激活函数ReLU串起来，并对每个激活函数的输出使用丢弃法。
# 可以分别设置各个层的丢弃概率。通常的建议是把靠近输入层的丢弃概率设置得小一点。
# 在此实验中，我们将第一个隐藏层的丢弃概率设置为0.2，第二个隐藏层的丢弃概率设为0.5。
# 通过参数is_training函数来判断运行模式为训练还是测试，并只需在训练模型下使用丢弃法。
drop_prob1, drop_prob2 = 0.2, 0.5

def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    # 第一隐藏层函数其实就是将原函数通过激活relu函数输出到第二层隐藏层
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training: # 只在训练模型时使用丢弃法
        H1 = dropout(H1, drop_prob1) # 在第一层全连接后添加丢弃层
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training: # 只在训练模型时使用丢弃法
        H1 = dropout(H2, drop_prob2) # 在第二层全连接后添加丢弃层
    return torch.matmul(H2, W3) + b3

# 在模型评估时，不应该进行丢弃
num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = dl.load_data_fashion_mnist(batch_size)
dl.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
'''
第1次正在训练.
epoch 1, loss 0.0047, train acc 0.534, test acc 0.724
第2次正在训练.
epoch 2, loss 0.0023, train acc 0.784, test acc 0.769
第3次正在训练.
epoch 3, loss 0.0019, train acc 0.825, test acc 0.829
第4次正在训练.
epoch 4, loss 0.0017, train acc 0.840, test acc 0.825
第5次正在训练.
epoch 5, loss 0.0016, train acc 0.849, test acc 0.833
'''

# PyTorch的简洁实现，只需要在全连接层后添加Dropout层并指定丢弃概率
# 在训练模型时，Dropout层将以指定的丢弃概率随机丢弃上一层的输出元素；在测试模型时（即model.eval()后），Dropout层并不发挥作用。
net = nn.Sequential(
    dl.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2, num_outputs)
)

for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)
    
# 训练并测试模型
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
dl.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
'''
第1次正在训练.
epoch 1, loss 0.0044, train acc 0.557, test acc 0.763
第2次正在训练.
epoch 2, loss 0.0022, train acc 0.787, test acc 0.819
第3次正在训练.
epoch 3, loss 0.0019, train acc 0.820, test acc 0.743
第4次正在训练.
epoch 4, loss 0.0018, train acc 0.837, test acc 0.834
第5次正在训练.
epoch 5, loss 0.0016, train acc 0.848, test acc 0.828
'''