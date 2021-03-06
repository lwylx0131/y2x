# -*- encoding: utf-8 -*-

import torch
import numpy as np
import sys
sys.path.append('../..')
import dl_common_pytorch as dl
import torch.nn as nn

'''
深度模型有关数值稳定性的典型问题就是衰减（vanishing）和爆炸（explosion）。
当神经网络的层数较多时，模型的数值稳定性容易变差。假设一个层数为L的多层感知机的第l层H(l)的权重参数为W(l)，输出层H(L)的权重参数为W(L)。
为便于讨论，不考虑偏差参数，且设所以隐藏层的激活函数为恒等映射Q(x) = x。
给定输入X，多层感知机的第l层的输出H(l) = XW(1)W(2)...W(L)。此时如果层数l较大，H(l)的计算可能会出现衰减或爆炸。举个例子：
假设输入和所有层的权重参数都是标量，如权重参数为0.2和5，多层感知机的第30层输出为输入X分别于0.2^30 = 1*10^(-21)衰减和5^30 = 9*10^20爆炸。
类似地，当层数较多时，梯度计算也很容易出现衰减或爆炸。

在神经网络中，通常需要随机初始化模型参数。原因如下：
在多层感知机中，假设输出层只保留一个输出单元o1（删去o2和o3以及指向它们的箭头），且隐藏层使用相同的激活函数。
如果将每个隐藏单元的参数都初始化为相等的值，那么在正向传播时每个隐藏单元将根据相同的输入计算出相同的值，并传递至输出层。
在反向传播中，每个隐藏单元的参数梯度值相等。
因此，这些参数在使用基于梯度的优化算法迭代后值依然相等。之后的迭代也是如此。
在这个情况下，无论隐藏单元有多少，隐藏层本质上只有一个隐藏单元在发挥作用。
因此，正如在前面的实验中所做的那样，我们通常将神经网络的模型参数，特别是权重参数，进行随机初始化。
'''

'''
随机初始化模型参数的方法有很多。在线性回归的简洁实现中，我们使用torch.nn.init.normal_()使模型net的权重参数采用正态分布的随机初始化方式。
不过，PyTorch中nn.Module的模块参数都采取了较为合理的初始化策略，因此一般不用我们考虑。

Xavier随机初始化：
还有一种比较常用的随机初始化方法叫作Xavier随机初始化。 假设某全连接层的输入个数为a，输出个数为b，Xavier随机初始化将使该层中权重参数的每个元素都随机采样于均匀分布
它的设计主要考虑到，模型参数初始化后，每层输出的方差不该受该层输入个数影响，且每层梯度的方差也不该受该层输出个数影响。
'''