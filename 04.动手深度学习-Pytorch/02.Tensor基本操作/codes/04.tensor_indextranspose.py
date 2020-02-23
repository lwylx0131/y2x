# -*- encoding: utf-8 -*-
import torch

'''
1)torch.masked_select(input, mask, out=None)
  input: 要索引的张量
  mask: 与input同形状的布尔类型张量
'''
# 使用torch.masked_select()进行张量的索引
t = torch.randint(0, 12, size=(4, 3))
# greater than or equal(ge) / greater than(gt)
mask = t.ge(6)
# 将大于等于6的数据挑选出来，返回一维张量
t_select = torch.masked_select(t, mask)
print(t, '\n', mask, '\n', t_select)
'''
tensor([[11,  6,  1],
        [ 0,  4,  4],
        [ 5,  3,  7],
        [ 2,  1,  5]]) 
tensor([[ True,  True, False],
        [False, False, False],
        [False, False,  True],
        [False, False, False]]) 
tensor([11,  6,  7])
'''

# 使用torch.reshape()变换张量的形状
t = torch.randperm(10)
t1 = torch.reshape(t, (2, 5))
print(t, '\n', t1)
'''
tensor([2, 4, 3, 9, 6, 8, 5, 0, 1, 7]) 
tensor([[2, 4, 3, 9, 6],
        [8, 5, 0, 1, 7]])
'''

# -1代表其他维度计算得到
t1 = torch.reshape(t, (-1, 5))
print(t, '\n', t1)
'''
tensor([4, 6, 3, 9, 5, 0, 7, 8, 2, 1]) 
tensor([[4, 6, 3, 9, 5],
        [0, 7, 8, 2, 1]])
'''

# 当张量在内存中是连续时，新张量和input共享数据内存
t = torch.randperm(10)
t[0] = 1024
print(t, '\n', t1)
print(id(t.data), id(t1.data))
'''
tensor([1024,    7,    4,    0,    3,    1,    5,    2,    9,    6]) 
tensor([[9, 7, 1, 2, 4],
        [3, 6, 8, 5, 0]])
110637920 110637920
'''

'''
1)torch.transpose(input, dim0, dim1)
  input：要变换的张量
  dim0：要交换的维度
  dim1：要交换的维度
'''
# 使用torch.transpose()交换张量的两个维度
t = torch.rand((4, 3, 2))
# 交换他们的第0和第1维
t1 = torch.transpose(t, dim0=0, dim1=1)
print(t.shape, t1.shape)
'''
torch.Size([4, 3, 2]) torch.Size([3, 4, 2])
'''

# 使用torch.t()变换张量
# 2维张量转置，对于矩阵而言，等价于torch.transpose(input, 0, 1)
x = torch.randn(3, 2)
print(x)
print(torch.t(x))
'''
tensor([[ 1.8024, -0.1625],
        [ 0.0738,  0.5010],
        [ 0.5289,  0.3570]])
tensor([[ 1.8024,  0.0738,  0.5289],
        [-0.1625,  0.5010,  0.3570]])
'''

'''
1)torch.squeeze(input, dim=None, out=None)
  dim: 若为None，移除所有长度为1的轴；若指定维度，当且仅当该轴长度为1时，可以被移除。
'''
# 使用torch.squeeze()变换，压缩长度为1的维度（轴）
t = torch.rand((1, 2, 1, 1))
t1 = torch.squeeze(t)
t2 = torch.squeeze(t, dim=0)
# 指定的轴长度不为1，不能移除
t3 = torch.squeeze(t, dim=1)
print(t.shape, '\n', t1.shape, t2.shape, t3.shape)
'''
torch.Size([1, 2, 1, 1]) 
torch.Size([2]) torch.Size([2, 1, 1]) torch.Size([1, 2, 1, 1])
'''

# 使用torch.unsqueeze()变换，依据dim扩展维度
x = torch.tensor([1, 2, 3, 4, 5])
print(torch.unsqueeze(x, 0))
print(torch.unsqueeze(x, 1))
'''
tensor([[1, 2, 3, 4, 5]])
tensor([[1],
        [2],
        [3],
        [4],
        [5]])
'''