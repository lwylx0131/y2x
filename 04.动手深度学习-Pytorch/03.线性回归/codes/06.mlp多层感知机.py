# -*- encoding: utf-8 -*-

import torch
import numpy as np
import sys
sys.path.append('../..')
import dl_common_pytorch as dl

# 使用多层感知机对图像进行分类
batch_size = 256
train_iter, test_iter = dl.load_data_fashion_mnist(batch_size)

# 图像形状28*28，类别数为10：输入个数为784，输出个数为10，设置超参数隐藏单元个数为256
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)
params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

# 定义激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

# 定义模型
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1) # 隐藏层其实就是输入层经过非线性激活函数变换输出的结果
    return torch.matmul(H, W2) + b2

# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

# 训练多层感知机与训练softmax回归差不多，可以直接调用train_ch3函数
num_epochs, lr = 5, 100.0
dl.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
'''
第1次正在训练.
epoch 1, loss 0.0030, train acc 0.716, test acc 0.795
第2次正在训练.
epoch 2, loss 0.0019, train acc 0.824, test acc 0.827
第3次正在训练.
epoch 3, loss 0.0017, train acc 0.844, test acc 0.806
第4次正在训练.
epoch 4, loss 0.0015, train acc 0.856, test acc 0.842
第5次正在训练.
epoch 5, loss 0.0014, train acc 0.865, test acc 0.853
'''

'''
评论区：
ReLu激活函数为什么没有梯度消失问题，当输入<=0时，梯度不也变成0了么？
ReLU在x<0部分会有梯度消失问题，但只有一半；所以后续有Leaky ReLU来缓解这个问题
但是相比于sigmoid梯度最大值0.25并且大部分区域都非常小，ReLU只有一半区域还是缓解很多

sigmoid的梯度消失是指输入值特别大或者特别小的时候求出来的梯度特别小，当网络较深，反向传播时梯度一乘就没有了，这是sigmoid函数的饱和特性导致的。
ReLU在一定程度上优化了这个问题是因为用了max函数，对大于0的输入直接给1的梯度，对小于0的输入则不管。
但是ReLU存在将神经元杀死的可能性，这和他输入小于0那部分梯度为0有关，
当学习率特别大，对于有的输入在参数更新时可能会让某些神经元直接失活，以后遇到什么样的输入输出都是0，Leaky ReLU输入小于0的部分用很小的斜率，有助于缓解这个问题。

为什么选择的激活函数普遍具有梯度消失的特点?
开始的时候我一直好奇为什么选择的激活函数普遍具有梯度消失的特点，这样不就让部分神经元失活使最后结果出问题吗？后来看到一篇文章的描述才发现，正是因为模拟人脑的生物神经网络的方法。在2001年有研究表明生物脑的神经元工作具有稀疏性，这样可以节约尽可能多的能量，据研究，只有大约1%-4%的神经元被激活参与，绝大多数情况下，神经元是处于抑制状态的，因此ReLu函数反而是更加优秀的近似生物激活函数。
所以第一个问题，抑制现象是必须发生的，这样能更好的拟合特征。
那么自然也引申出了第二个问题，为什么sigmoid函数这类函数不行？
1.中间部分梯度值过小（最大只有0.25）因此即使在中间部分也没有办法明显的激活，反而会在多层中失活，表现非常不好。
2.指数运算在计算中过于复杂，不利于运算，反而ReLu函数用最简单的梯度
在第二条解决之后，我们来看看ReLu函数所遇到的问题，
1.在负向部分完全失活，如果选择的超参数不好等情况，可能会出现过多神经元失活，从而整个网络死亡。
2.ReLu函数不是zero-centered，即激活函数输出的总是非负值，而gradient也是非负值，在back propagate情况下总会得到与输入x相同的结果，同正或者同负，因此收敛会显著受到影响，一些要减小的参数和要增加的参数会受到捆绑限制。
这两个问题的解决方法分别是
1.如果出现神经元失活的情况，可以选择调整超参数或者换成Leaky ReLu 但是，没有证据证明任何情况下都是Leaky-ReLu好
2.针对非zero-centered情况，可以选择用minibatch gradient decent 通过batch里面的正负调整，或者使用ELU(Exponential Linear Units)但是同样具有计算量过大的情况，同样没有证据ELU总是优于ReLU。
所以绝大多数情况下建议使用ReLu。

多层感知机中最为重要的自然是“多层”，多层中涉及到的隐藏层的目的是为了将线性的神经网络复杂化，更加有效的逼近满足条件的任何一个函数。
因此文中先证明了一个最常见的思路，即两个线性层复合，是不可行的，无论多少层线性层复合最后得到的结果仍然是等价于线性层。这个结果的逻辑来自与线性代数中，H=XW+b 是一个仿射变换，通过W变换和b平移，而O=HW2+b2 则是通过W2变换和b2平移，最终经过矩阵的乘法和加法的运算法则（分配律）得到最终仍然是对X的仿射变换。
在线性层复合不行的情况下，最容易想到的思路就是将隐藏层变成非线性的，即通过一个“激励函数”将隐藏层的输出改变。
因此这里主要讨论一下，为什么添加激励函数后可以拟合“几乎”任意一个函数。
将函数分成三类：逻辑函数，分类函数，连续函数（分类的原则是输入输出形式）
1.通过一个激励函数可以完成简单的或与非门逻辑，因此通过隐藏层中神经元复合就可以完成任何一个逻辑函数拟合。只需要通过神经网络的拟合将真值表完全表示
2.通过之前使用的线性分类器构成的线性边界进行复合便可以得到任意一个分类函数。
3.通过积分微元法的思想可以拟合任何一个普通的可积函数
'''