# -*- encoding: utf-8 -*-

import torch
from IPython import display
from matplotlib import pyplot as plt
import random

'''
生成第二个特征features[:,1]和标签labels的散点图，观察两者间的线性关系
'''
def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(4.5, 3.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

# 遍历数据集并不断读取小批量数据样本，每次返回batch_size个随机样本的特征和标签
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 对列表indices中的所有元素随机打乱顺序洗牌，函数返回的是None
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)
'''
features = torch.from_numpy(np.random.normal(0, 0.1, (5, 2)))
tensor([[ 0.0049, -0.0851],
        [ 0.1637, -0.0807],
        [-0.0296, -0.0461],
        [ 0.1592,  0.0013],
        [ 0.1071,  0.1362]], dtype=torch.float64)
labels = 2*features[:, 0] + (-3.4)*features[:, 1] + 4.2
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))
indices[0: 2]
[4, 3, 0, 1, 2] -> [4, 3]
j = torch.LongTensor(indices[0:2])
features.index_select(0, j)
tensor([[0.1071, 0.1362],
        [0.1592, 0.0013]], dtype=torch.float64)
'''
		
# 线性回归矢量计算表达式，使用mm函数做矩阵乘法
def linreq(X, w, b):
    w = w.to(torch.float64) # tensor double -> float64
    return torch.mm(X, w) + b

# 平方损失定义线性回归的损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

# 小批量随机梯度下降算法，通过不断迭代模型参数来优化损失函数
# 此处自动求梯度模块计算得来的梯度是一个批量样本的梯度和，将它除以批量大小来得到平均值
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# ########################## 5.6 #########################3
import sys
import torchvision
def load_data_fashion_mnist(batch_size, resize=None, root='FashionMNIST2065'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

import torch.nn as nn
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)
		
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        print("第%d次正在训练." %(epoch+1))
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
        
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
        
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()
        
            train_l_sum += l.item() # 得到损失函数的总值
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %(epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))