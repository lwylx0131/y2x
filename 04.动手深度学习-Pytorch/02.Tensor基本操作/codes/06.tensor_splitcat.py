# -*- encoding: utf-8 -*-
import torch
import numpy as np

'''
1)torch.cat(tensors, dim=0, out=None)
  tensors:张量序列
  dim:要拼接的维度
'''
# 使用torch.cat()拼接，将张量按维度dim进行拼接，不会扩张张量的维度
t = torch.ones((3, 2))
# 在第0个维度上拼接
t0 = torch.cat([t, t], dim=0)
# 在第1个维度上拼接
t1 = torch.cat([t, t], dim=1)
print(t0, '\n', t1)
'''
tensor([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]]) 
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
'''

'''
1)torch.stack(tensors, dim=0, out=None)
  tensors:张量序列
  dim:要拼接的维度

规律：将需要拼接的张量个数n，按照dim插入(3, 2)前中后
原始维度为(3, 2)，当拼接2个张量且dim=0，则维度变为(n=2, 3, 2)
原始维度为(3, 2)，当拼接2个张量且dim=1，则维度变为(3, n=2, 2)
原始维度为(3, 2)，当拼接2个张量且dim=2，则维度变为(3, 2, n=2)
原始维度为(3, 2)，当拼接3个张量且dim=0，则维度变为(n=3, 3, 2)
'''
# 使用torch.stack()拼接，在新建的维度dim上进行拼接，会夸张张量的维度
t = torch.ones((3, 2))
# 在新建的维度上进行拼接，拼接完将从2维变成3维
t1 = torch.stack([t, t], dim=2)
print(t1, '\n', t1.shape)
'''
tensor([[[1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.]]]) 
torch.Size([3, 2, 2])
'''

t = torch.ones((3, 2))
# 在新建的维度上进行拼接
# 由于指定是第0维，会把原来的3，2往后移动一格，然后在新的第0维创建新维度进行拼接
t1 = torch.stack([t, t], dim=0)
print(t1, '\n', t1.shape)
'''
tensor([[[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]]]) 
torch.Size([2, 3, 2])
'''

t = torch.ones((3, 2))
# 在新建的维度上进行拼接
# 由于指定是第0维，会把原来的3，2往后移动一格，然后在新的第0维创建新维度进行拼接
t1 = torch.stack([t, t, t], dim=0)
print(t1, '\n', t1.shape)
'''
tensor([[[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]]]) 
torch.Size([3, 3, 2])
'''

'''
1)torch.chunk(input, chunks, dim=0)
  input：要切分的张量
  chunks：要切分的份数
  dim：要切分的维度
'''
# 使用torch.chunk()切分，可以将张量按维度dim进行平均切分
# 如果不能整除，那么最后一份张量小于其他张量
a = torch.ones((5, 2))
# 在5这个维度切分，切分成2个张量
t = torch.chunk(a, dim=0, chunks=2)
for idx, t_chunk in enumerate(t):
    print(idx, t_chunk, t_chunk.shape)
'''
0 tensor([[1., 1.],
        [1., 1.],
        [1., 1.]]) torch.Size([3, 2])
1 tensor([[1., 1.],
        [1., 1.]]) torch.Size([2, 2])
'''

'''
1)torch.split(tensor, split_size_or_sections, dim=0)
  tensor：要切分的张量
  split_size_or_sections：为int时，表示每一份的长度；为list时，按list元素切分
  dim：要切分的维度
'''
# 使用torch.split()切分，将张量按维度dim进行切分
a = torch.ones((5, 2))
# 指定每个张量的长度为2
t = torch.split(a, 2, dim=0)
# 切出3个张量
for idx, t_split in enumerate(t):
    print(idx, t_split, t_split.shape)
'''
0 tensor([[1., 1.],
        [1., 1.]]) torch.Size([2, 2])
1 tensor([[1., 1.],
        [1., 1.]]) torch.Size([2, 2])
2 tensor([[1., 1.]]) torch.Size([1, 2])
'''

a = torch.ones((5, 2))
# 指定了每个张量的长度为列表中的大小[2, 1, 2]
t = torch.split(a, [2, 1, 2], dim=0)
# 切出3个张量
for idx, t_split in enumerate(t):
    print(idx, t_split, t_split.shape)
'''
0 tensor([[1., 1.],
        [1., 1.]]) torch.Size([2, 2])
1 tensor([[1., 1.]]) torch.Size([1, 2])
2 tensor([[1., 1.],
        [1., 1.]]) torch.Size([2, 2])
'''

a = torch.ones((5, 2))
# list中求和不为长度则抛出异常
#t = torch.split(a, [2, 1, 1], dim=0)
# 切出3个张量
#for idx, t_split in enumerate(t):
#    print(idx, t_split, t_split.shape)
'''
builtins.RuntimeError: split_with_sizes expects split_sizes to sum exactly to 5 (input tensor's size at dimension 0), but got split_sizes=[2, 1, 1]
'''