# -*- encoding: utf-8 -*-

import torch
import numpy as np

x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.float64)
x = torch.randn_like(x, dtype=torch.float)

# 索引出来的结果与原数据共享内存，即修改一个，另一个也跟着修改
y = x[0, :]
y += 1
print(y)
# 源tensor也被修改
print(x[0, :])

# 使用view()改变Tensor形状
# view()返回的新tensor与源tensor共享内存（其实是同一个tensor），即修改一个，另一个也跟着修改
y = x.view(15)
# -1所指的维度可以根据其他维度的值退出来
z = x.view(-1, 5)
print(x.size(), y.size(), z.size())
x += 1
print(x, y)

# 如果我们想返回一个真正的新副本（不共享内存），pytorch提供reshape()可以改变形状，但不保证返回的是其拷贝，所以不推荐使用。推荐先用clone创造一个副本再使用view
c_cp = x.clone().view(15)
x -= 1
print(x, x_cp)
# 使用clone还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源tensor

# item()将一个标量tensor转换为一个python number
x = torch.randn(1)
print(x)
print(x.item())

# 当两个形状不同的tensor按元素运算时，可能会触发广播机制：
# 先适当复制元素使这两个tensor形状相同后再按元素运算
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)

# 索引、view不会开辟新内存，而像y=x+y这样的运算则会新开内存，然后将y指向新的内存
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before)

# 将结果指定到原来y的内存，以下是将x+y的结果通过[:]写进y对应内存中
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before)

# 还可以使用运算符全名函数中的out参数或者自加运算符+=（也即add_()）达到上述效果
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y)
print(id(y) == id_before)

# numpy()和from_numpy()将tensor和numpy数组互相转换，不过转换后他们是共享内存，所以转换很快，改变其中一个时另一个也会改变
# 不过将numpy中的array转换成tensor的方法torch.tensor()，它需要消耗更多的时间和空间，因为转换时会进行数据拷贝且返回的tensor和原来的数据不再共享内存
a = torch.ones(5)
b = a.numpy()
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)

a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)

c = torch.tensor(a)
a += 1
print(a, c)

