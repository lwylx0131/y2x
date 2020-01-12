# -*- encoding: utf-8 -*-

import torch
import numpy as np

x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)
y = x + 2
print(y)
print(y.grad_fn)
# 由上面可知，x是直接创建的，所以没有grad_fn，而y是通过加法操作创建，有一个AddBackward的grad_fn
# 像x这种直接创建的称为叶子节点，对应的grad_fn是None
print(x.is_leaf, y.is_leaf)

z = y * y * 3
out = z.mean()
print(z, out)

# 通过.requires_grad_()来用in-place的方式改变requires_grad属性
a = torch.randn(2, 2)
a = (a * 3) / (a - 1)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# out是一个标量，所以调用backward()时不需要制定求导变量
out.backward() # 等价 out.backward(torch.tensor(1.))
# 查看out关于x的梯度：d(out) / dx
print(x.grad)

# grad在反向传播过程中是累加的，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需要把梯度清零
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)

# 为了避免更加复杂的求导问题，不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量。所以必要时，我们要把张量通过所有张量的元素加权求和的方式转换为标量。
# 假设y由自变量x计算而来，w是和y同形的张量，则y.backward(w)的含义是：先计算l = torch.sum(y * w)，则l是个标量，然后求l对自变量x的导数
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)
# 现在y不是一个标量，所以在调用backward时需要传入一个和y同形的权重向量进行加权求和得到一个标量
v = torch.tensor([[1.0, 0.1], [0.01, 0.0001]], dtype=torch.float)
z.backward(v)
print(x, x.grad) # 注意x.grad与x是同形的张量


# 中断梯度追踪
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
print(x.requires_grad)
print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)
# y3对x求梯度
y3.backward()
print(x.grad) # tensor(2.)
# 以上y3对x的梯度为什么是2？y3=y1+y2=x平方+x立方，当x=1时，dy3/dx应该是5
# 事实上，由于y2的定义是被torch.no_grad()包裹，所以与y2有关的梯度不会回传，只有与y1有关的梯度才会回传，即x平方对x的梯度

# 如果想要修改tensor数值，但又不希望被autograd记录（即不会影响反向传播），那么可以对tensor.data操作
x = torch.ones(1, requires_grad=True)
print(x.data) # 还是一个tensor
print(x.data.requires_grad) #但是已经是独立于计算图之外
y = 2 * x
x.data *= 100 # 只改变值，不会记录在计算度，所以不会影响梯度传播
y.backward()
print(x) # 更改data的值也不会影响tensor的值
print(x.grad)

