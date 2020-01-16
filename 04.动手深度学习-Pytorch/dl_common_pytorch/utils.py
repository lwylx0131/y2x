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