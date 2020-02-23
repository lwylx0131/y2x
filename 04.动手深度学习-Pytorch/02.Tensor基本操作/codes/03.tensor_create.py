# -*- encoding: utf-8 -*-
import torch
import numpy as np

'''
1)torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
  data: 数据，可以是list，numpy的ndarray
  dtype: 数据类型，默认与data的类型一致
  device: 所在设备，gpu/cpu
  requires_grad: 是否需要梯度，因为神经网络结构经常会要求梯度
  pin_memory: 是否存于锁页内存
'''
# 使用numpy创建张量，创建一个3*3的全1张量
arr = np.ones((3, 3))
print('数据类型为: ', arr.dtype)
t = torch.tensor(arr)
print(t)

# 在GPU环境上创建张量
#arr = np.ones((3, 3))
#t = torch.tensor(arr, device='cuda')
#print(t)

# 使用torch.from_numpy(ndarray)创建张量
# 从torch.from_numpy创建的tensor张量和ndarray共享内存，当修改其中一个数据，另外一个也会被修改
arr = np.array([[1, 2, 3], [4, 5, 6]])
t = torch.from_numpy(arr)
print(arr)
print(t)

'''
1)zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
  torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False)
  size：为张量的形状
  out：输出的张量
  dtype: 数据类型
  layout：内存中的布局形式，有strided，sparse_coo等
  device：所在设备，gpu/cpu
  requires_grad:是否需要梯度
'''
# 创建全0张量
# 创建一个t1张量，赋予一个初始值
t1 = torch.tensor([1, 2, 3])
# 将创建的t张量输出到t1，t和t1的id相同，说明指向的内存块也相同
t = torch.zeros((2, 3), out=t1)
print(t, '\n', t1)
print(id(t), id(t1), id(t) == id(t1))

# 通过torch.zeros_like()，根据input形状创建全0的3*5张量
input = torch.empty(3, 5)
print(input)
print(torch.zeros_like(input))
print(torch.ones_like(input))

'''
1)torch.full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
  torch.full_like(input, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
  fill_value: 填充的值
'''
# 根据数值创建一个2*3全部是8的张量
print(torch.full((2, 3), 8))
input = torch.empty(3, 5)
print(input)
# 根据input形状创建一个元素都为8的张量
print(torch.full_like(input, 9))

'''
1)arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
  start:数列的起始值
  end:数列的结束值，取不到，只能取到 end-1
  step:公差（步长），默认为 1
'''
#创建等差的1维张量
print(torch.arange(1, 9, 2))

'''
1)linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
  steps:创建出的一维张量的元素个数
  end：结束位置可以取到
'''
# 创建等间距（均分）的1维张量
print(torch.linspace(1, 9, 5))

'''
1)logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
  base: 对数函数的底，默认为10
'''
# 创建对数均分的1维张量
print(torch.logspace(start=-5, end=10, steps=4))

'''
1)torch.eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
  m:矩阵行数
  n:矩阵列数
'''
# 创建单位对角矩阵
print(torch.eye(4))

'''
1)torch.normal(mean, std, out=None)
  mean:均值
  std:标准差
这种生成正态分布数据的张量创建有4种模式：
（1）mean为张量，std为张量
（2）mean为标量，std为标量
（3）mean为标量，std为张量
（4）mean为张量，std为标量
'''
# 生成正态分布（高斯分布）数据的张量
mean = torch.arange(1, 6, dtype=torch.float)
std = torch.arange(1, 6, dtype=torch.float)
t = torch.normal(mean, std)
print('mean: {}, std: {}'.format(mean, std))
print(t)
# 此时的mean和std都是张量，可以理解为其中的-0.6152是mean为1，std为1的正态分布采样得到，其他对应位置数据同理得到，只是从不同的正态分布中采样得到。
# mean: tensor([1., 2., 3., 4., 5.]), std: tensor([1., 2., 3., 4., 5.])
# tensor([ 0.2580,  0.0412,  5.7647, -0.0075,  3.6192])

# mean为标量，std为标量
print(torch.normal(0.2, 1.0, size=(5, )))
# 这里生成的数据都是通过mean为0.2，std为1.0采样得到的，长度为5的一维张量。

# mean为标量，std为张量
mean = 2
std = torch.arange(1, 4, dtype=torch.float)
t = torch.normal(mean, std)
print('mean: {}, std: {}'.format(mean, std))
print(t)
# mean: 2, std: tensor([1., 2., 3.])
# tensor([2.2420, 3.3668, 4.4485])
# 2.2420是mean为2，std为1的正态分布采样得到，其他对应位置数据同理得到，只是从不同的正态分布中(均值不同，标准差相同）采样得到。

'''
1)torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
  size:张量的形状
'''
# 标准正态分布数据的张量：均值为0，标准差为1
# 创建一个长度为6的标准正态分布张量
print(torch.randn(6))

# 创建一个二维的标准正态分布张量
print(torch.randn(3, 4))

# torch.randn_like()根据张量的形状创建新的标准正态分布张量
ones = torch.ones((3, 4))
print(torch.randn_like(ones))

# 创建在(0, 1]上均匀分布的张量
# 创建一个均匀分布的长度为5的一维张量
print(torch.rand(5))

# 创建一个均匀分布的二维张量
print(torch.rand(3, 4))

'''
1)torch.randint(low=0, high, size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
  在区间[low,high)上生成整数均匀分布数据的张量
'''
# 在区间上创建整数均匀分布数据的张量
# 在[2, 6)上创建均匀分布的整数张量，长度为4的一维张量
print(torch.randint(2, 6, (4, )))
print(torch.randint(2, 6, (4, 2)))

'''
1)torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False)
  torch.bernoulli(input, *, generator=None, out=None)
  生成0~n-1的随机排列一维张量
  input：概率值
'''
print(torch.randperm(6))

# 生成伯努利分布（0-1分布，两点分布）的张量
# 先创建一个张量a，作为概率值输入
a = torch.empty(3, 3).uniform_(0, 1)
print(a)
# 使用上面创建的张量a作为概率值创建伯努利分布
print(torch.bernoulli(a))
