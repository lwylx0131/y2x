# -*- encoding: utf-8 -*-

import torch
import matplotlib.pyplot as plt

# 随机数种子
torch.manual_seed(10)
# 学习率
lr = 0.1

# 创建训练数据
# torch.rand 大于0
# torch.randn 可以小于0
x = torch.rand(20, 1) * 10 # (20, 1)
y = 2 * x + (5 + torch.randn(20, 1)) # (20, 1)

# 构建线性回归参数
# 随机初始化w和b，通过requires_grad=True属性用到自动梯度求导
w = torch.randn((1), requires_grad=True)
b = torch.randn((1), requires_grad=True)
'''
每次循环先进行前向传播，
计算 y 的预测值，
计算 loss 损失值，
然后反向传播损失，
去更新参数 w、b。
'''
for iteration in range(1000):
    # 前向传播
    wx = torch.mul(w, x) # w * x
    y_pred = torch.add(wx, b) # y = w * x + b
    
    # 计算MES loss
    loss = (0.5 * (y - y_pred) ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    b.data.sub_(lr * b.grad) # b = b - lr * b.grad
    w.data.sub_(lr * w.grad) # w = w - lr * w.grad
    
    if iteration % 20 == 0:
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' %(loss.data.numpy()), fontdict={'size':20, 'color':'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration:{}\nw:{},b:{}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(3.0)   
        
        if loss.data.numpy() < 1:
            break