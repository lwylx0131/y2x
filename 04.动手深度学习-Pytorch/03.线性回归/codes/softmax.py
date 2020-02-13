# -*- encoding: utf-8 -*-

import torch
import numpy as np

'''
softmax回归跟线性回归一样将输入特征与权重做线性叠加。
与线性回归的主要不同在于，softmax回归的输出值个数等于标签里的类别数：
o1 = x1w11 + x2w21 + x3w31 + x4w41 + b1
o2 = w1w12 + x2w22 + x3w32 + x4w42 + b2
o3 = w1w13 + x2w23 + x3w33 + x4w43 + b3
softmax回归同线性回归一样，也是一个单层神经网络。
由于每个输出o1,o2,o3的计算都要依赖于所有的输入x1,x2,x3,x4，softmax回归的输出层也是一个全连接层。
'''

'''
torchvision主要由以下几部分构成：
torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
torchvision.utils: 其他的一些有用的方法。
'''
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import time

mnist_train = torchvision.datasets.FashionMNIST(root='FashionMNIST2065', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='FashionMNIST2065', train=False, download=True, transform=transforms.ToTensor())
'''
root（string）– 数据集的根目录，其中存放processed/training.pt和processed/test.pt文件。
train（bool, 可选）– 如果设置为True，从training.pt创建数据集，否则从test.pt创建。
download（bool, 可选）– 如果设置为True，从互联网下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。
transform（可被调用 , 可选）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：transforms.RandomCrop。
target_transform（可被调用 , 可选）– 一种函数或变换，输入目标，进行变换。
'''

# 通过下标访问任何一个样本，访问第一个样本的特征和标签
feature, label = mnist_train[0]
print(feature.shape, label)

'''
一个维度，28*28，表示一张图片（9）的像素组成
tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ... ,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ... ,0.4980, 0.2431, 0.2118, 0.0000, 0.0000, 0.0000, 0.0039, 0.0118],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ... ,0.6902, 0.5255, 0.5647, 0.4824, 0.0902, 0.0000, 0.0000, 0.0000],
		 ...
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ... ,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]) 9
'''

X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0]) # 将第i个feature加到X中
    y.append(mnist_train[i][1]) # 将第i个label加到y中

# 在实践中，读取数据经常是训练的性能瓶颈，特别当模型较简单或者计算硬件性能较高时，
# pytorch的DataLoader中一个很方便的功能是允许使用多进程来加速数据读取，这里使用num_workers设置4个进程读取数据
batch_size = 256
num_workers = 0 # win必须0，表示不用额外的进程来加速读取数据
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

import sys
sys.path.append('../..')
import dl_common_pytorch as dl
#dl.load_data_fashion_mnist(256)

#start = time.time()
#for X, y in train_iter:
#    continue
#print('train data set iter %.2f sec' %(time.time() - start))

# 跟线性回归一样，使用向量表示每个样本。
# 已知每个样本输入是高和宽均为28像素的图像。模型的输入向量长度为28*28=784：该向量的每个元素对应图像中的每个像素
# 由于图像有10个类别，单层神经网络输出层的输出个数为10，因此softmax回归的权重和偏差参数分别为784*10和1*10的矩阵
num_inputs = 784
num_outputs = 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 对多维Tensor按维度操作，可以只对其中同一列dim=0或者同一行dim=1的元素求和，并在结果中保留行和列这两个维度keepdim=True
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0), X.sum(dim=0).shape)
print(X.sum(dim=0, keepdim=True), X.sum(dim=0, keepdim=True).shape)
'''
tensor([5, 7, 9]) torch.Size([3])
tensor([[5, 7, 9]]) torch.Size([1, 3])
'''
print(X.sum(dim=1), X.sum(dim=1).shape)
print(X.sum(dim=1, keepdim=True), X.sum(dim=1, keepdim=True).shape)
'''
tensor([ 6, 15]) torch.Size([2])
tensor([[ 6],
        [15]]) torch.Size([2, 1])
'''

'''
为了表达样本预测各个输出的概率，softmax运算会先通过exp函数对每个元素做指数运算，再对exp矩阵同行元素求和，最后令矩阵每行各元素与该行元素之和相除。
最终得到的矩阵每行元素和为1且非负，因此，该矩阵每行都是合法的概率分布。
softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别的预测概率。

这是由于softmax函数的常数不变性，即softmax(x)=softmax(x+c)，推导如下：
上面的exp(c)之所以可以消除，是因为exp(a+b)=exp(a)*exp(b)这个特性将exp(c)提取出来了。
在计算softmax概率的时候，为了保证数值稳定性（numerical stability），我们可以选择给输入项减去一个常数，
比如x的每个元素都要减去一个x中的最大元素。当输入项很大的时候，如果不减这样一个常数，取指数之后结果会变得非常大，发生溢出的现象，导致结果出现inf。

softmax函数为什么要用e的指数形式归一化呢，如果用其他形式可不可以呢（例如直接线性归一化，带入幂函数归一化等），指数形式又有什么好处呢？
目前看来，无论用那种函数形式归一化都能达到归一的目的，但是指数形式在由于导数随着输入变大而变大，
可以把较大的输入域较小的输入拉开得更大，两个较大的输入之间也可以拉开距离，有利于将最大的概率值筛选出来。
那么如果基于归一化和筛选两个特性，只要底大于1的其他指数函数也可以实现这个效果。

https://www.boyuai.com/elites/course/cZu18YmweLv10OeV/video/-m1RzLMiaJHiHvnuIWFwc#comment-wl-7TK1GHcYp11IliwPLs
使用e的指数形式归一化是由于假设未知来源误差服从高斯分布，化简的结果得到的，
如果说明数据与理论标准函数之间的偏差服从其他的概率分布，最后选取得到的归一化函数就不一定是e的指数形式，并不是随便选的

损失函数 - 交叉熵损失函数：https://zhuanlan.zhihu.com/p/35709485

如果说softmax的目的只是得到不同类别概率一个和为1的计算方法，那这种计算方法不是很多吗？为什么一定要用exp呢？
三种概略直接用10/（0.1+10+0.1）不也可以吗？如果说是为了避免分子为负数，也可以求平方之后在计算，也可以得到和为一的情况？
softmax是非线性的，你说的那种是线性的
平方之后就不是线性的了
emmm，如果用平方的话，感觉反向传播求导的时候会容易梯度爆炸

公式并不是拍脑门决定的，包括线性回归，logistic回归，softmax回归这类问题的公式都是有理论根据的。
这类算法属于广义线性模型，都是在假定误差服从指数族分布的情况下，按照极大似然估计的套路推导出来的。
没记错的话，线性回归是在误差服从高斯分布的基础上推导而来的，softmax回归是在误差服从多项式分布的基础上推导而来的
总结一下，你这是个好问题~但从抽象的角度去理解公式的合理性即可，不必分纠结理论。

softmax公式的得出方法大概解释可以解释为：
首先假设样本与理论标准函数的误差（类似于线性回归那一章中生成数据时叠加上的高斯误差）服从正态分布（高斯分布），并且不同样本之间独立同分布，
通过贝叶斯公式计算各个分类的概率，将高斯分布的公式带入公式之后化简得到。
在一些地方softmax函数又被称为归一化指数（normalized exponential）
'''
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition # 此处运用了广播机制

X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1))
'''
tensor([[0.1902, 0.2141, 0.1770, 0.1777, 0.2410],
        [0.2122, 0.2080, 0.2363, 0.0967, 0.2469]]) tensor([1., 1.])
'''

# 通过view函数将每张原始图像改成长度为num_inputs的向量
# X: 28*28=784 (n, 784) * W(784, 10) + b(1, 10) = (n, 10) + (1, 10) 
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

# y_hat是2个样本在3个类别的预测概率 y是这2个样本的标签类别
# 通过gather函数，得到2个样本标签的预测概率
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat)
y = torch.LongTensor([0, 2])
print(y_hat.gather(1, y.view(-1, 1)))
print(y_hat.argmax(dim=1))
print(y_hat.argmax(dim=1) == y)
print((y_hat.argmax(dim=1) == y).float())
print((y_hat.argmax(dim=1) == y).float().mean())
print((y_hat.argmax(dim=1) == y).float().mean().item())
'''
tensor([[0.1000, 0.3000, 0.6000],
        [0.3000, 0.2000, 0.5000]])
tensor([[0.1000],
        [0.5000]]) 
tensor([2, 2])
tensor([False,  True])
tensor([0., 1.])
tensor(0.5000)
0.5
'''

# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
	
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()
	
# y_hat:预测概率 y:真实标签
# 第一个样本预测类别为2（该行最大元素0.6在本行的索引为2），与真实标签0不一致
# 第二个样本预测类别为2（该行最大元素0.5在本行的索引为2），与真实标签2一致。
# 因此这两个样本的分类准确率为0.5
# 类似的我们可以评价模型net在数据集上的准确率
print(accuracy(y_hat, y))
print(dl.evaluate_accuracy(test_iter, net))

dl.train_ch3(net, train_iter, test_iter, cross_entropy, 5, batch_size, [W, b], lr=0.03)