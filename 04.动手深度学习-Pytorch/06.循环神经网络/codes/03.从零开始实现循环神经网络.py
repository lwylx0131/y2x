# -*- encoding: utf-8 -*-

#%matplotlib inline
import torch
import numpy as np
import pandas as pd
import time
import sys
import math
sys.path.append('../..')
import dl_common_pytorch as dl
from torch import nn, optim
import random
import zipfile
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

'''
使用循环神经网络训练一个语言模型，当模型训练好后，就可以使用该模型创作歌词。
'''
with zipfile.ZipFile('./jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')

# 数据集有6w多个字符，为打印方便，把换行符替换成空格，然后使用前1w个字符来训练模型
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0: 10000]
# 将每个字符映射成一个从0开始的连续整数，又称索引，来方便之后的数据处理。
# 为得到索引，将数据集里所有不同字符取出来，然后将其逐一映射到索引来构造词典。
# 打印vocab_size，即词典中不同字符的个数，又称词典大小。
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
#print('词典大小 = %d' %(vocab_size)) # 词典大小 = 1027
corpus_indices = [char_to_idx[char] for char in corpus_chars]

# 为了将词表示成向量输入到神经网络，一个简单办法是使用one-hot向量
# 词典大小vocab_size，0~N-1索引对应，如果一个字符索引为i，则创建一个全0的长为N的向量，并将其位置为i的元素设置为1，该向量就是对原字符的one-hot向量。
def one_hot(x, n_class, dtype=torch.float32):
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1) # https://www.cnblogs.com/shiyublog/p/10924287.html
    return res
x = torch.tensor([0, 2])
x_one_hot = one_hot(x, vocab_size)
print(x_one_hot)
print(x_one_hot.shape)
print(x_one_hot.sum(axis=1))
'''
tensor([[1., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 1.,  ..., 0., 0., 0.]])
torch.Size([2, 1027])
tensor([1., 1.])
'''

'''
每次采样的小批量的形状是（批量大小, 时间步数）。下面的函数将这样的小批量变换成数个形状为（批量大小, 词典大小）的矩阵，矩阵个数等于时间步数。
也就是说，时间步t的输入为Xt为R(n*d)，其中n为批量大小，d为词向量大小，即one-hot向量长度（词典大小）。
'''
def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

X = torch.arange(10).view(2, 5)
inputs = to_onehot(X, vocab_size)
print(len(inputs), inputs[0].shape) # 5 torch.Size([2, 1027])
'''
tensor([[1., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
tensor([[0., 1., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
tensor([[0., 0., 1.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
'''

device = 'cpu'
# 初始化模型参数，隐藏单元个数num_hiddens是一个超参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])
	
# 根据循环神经网络的计算表达式实现该模型。首先定义init_rnn_state函数来返回初始化的隐藏状态，它返回由一个形状为（批量大小，隐藏单元个数）的值为0的NDArray组成的元祖。
# 使用元祖是为了更便于处理隐藏状态含有多个NDArray的情况。
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# rnn函数定在一个时间步里如何计算隐藏状态和输出。这里激活函数使用tanh函数。
def rnn(inputs, state, params):
    # inputs/outputs都为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, )
# 简单测试：观察输出结果的个数（时间步数）以及第一个时间步的输出层输出的形状和隐藏状态的形状。
state = init_rnn_state(X.shape[0], num_hiddens, device)
inputs = to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(outputs), outputs[0].shape, state_new[0].shape)
# 5 torch.Size([2, 1027]) torch.Size([2, 256])

'''
定义预测函数：
以下函数基于前缀prefix（含有数个字符的字符串）来预测接下来的num_chars个字符。这个函数稍显复杂，
其中我们将循环神经单元rnn设置成了函数参数，这样在后面小节介绍其他循环神经网络时能重复使用这个函数。
'''
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]   # output记录prefix加上预测的num_chars个字符
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y[0].argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])
# 先测试一下predict_rnn函数。我们将根据前缀“分开”创作长度为10个字符（不考虑前缀长度）的一段歌词。因为模型参数为随机值，所以预测结果也是随机的。
print(predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
# 分开莫缓碎将了硬思期祈用

'''
裁剪梯度：
循环神经网络中较容易出现梯度衰减或梯度爆炸，这会导致网络几乎无法训练。
裁剪梯度（clip gradient）是一种应对梯度爆炸的方法。假设我们把所有模型参数的梯度拼接成一个向量 ，并设裁剪的阈值是。裁剪后的梯度：min(e/||g||, 1)g的L2范数不超过e
'''
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

'''
时序数据的采样：
在训练中我们需要每次随机读取小批量样本和标签。时序数据的一个样本通常包含连续的字符。
假设时间步数位5，样本序列为5个字符，即“想要有直升”。该样本的标签序列为这些字符分别在训练集中的下一个字符，即“要有直升机”。
有两种方式对时序数据进行采样：随机采样和相邻采样。
'''
# 随机采样：每次从数据里随机采样一个小批量，其中批量大小batch_size指每个小批量的样本数，num_steps为每个样本所包含的时间步数。
# 每个样本是原始序列上任意截取的一段序列。相邻的两个随机小批量在原始序列上的位置不一定相邻，因此，无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态。
# 在训练模型时，每次随机采样前都需要重新初始化隐藏状态。
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

'''
相邻采样：除对原始序列做随机采样外，还可以令相邻的两个随机小批量在原始序列上的位置相邻。这时就可以用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态，
从而使下一个小批量的输出也取决于当前小批量的输入， 并如此循环下去。此时对实现训练神经网络造成了两个方面的影响：
1）在训练模型时，只需要在每一个迭代周期开始时初始化隐藏状态；
2）当多个相邻小批量通过传递隐藏状态串联起来时，模型参数的梯度计算将依赖所有串联起来的小批量序列。同一迭代周期中，随着迭代次数的增加，梯度的计算开销会越来越大。
为了使模型参数的梯度计算只依赖一次迭代读取的小批量序列，可以在每次读取小批量前将隐藏状态从计算图中分离出来。
'''
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

'''
困惑度：我们通常使用困惑度（perplexity）来评价语言模型的好坏。回忆一下“softmax回归”一节中交叉熵损失函数的定义。困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，
最佳情况下，模型总是把标签类别的概率预测为1，此时困惑度为1；
最坏情况下，模型总是把标签类别的概率预测为0，此时困惑度为正无穷；
基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数。
显然，任何一个有效模型的困惑度必须小于类别个数。在本例中，困惑度必须小于词典大小vocab_size。

定义模型训练函数
跟之前章节的模型训练函数相比，这里的模型训练函数有以下几点不同：
1）使用困惑度评价模型。
2）在迭代模型参数前裁剪梯度。
3）对时序数据采用不同采样方法将导致隐藏状态初始化的不同。
'''
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach_()
            # inputs是num_steps个形状为(batch_size, vocab_size)的矩阵
            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成形状为
            # (num_steps * batch_size,)的向量，这样跟输出的行一一对应
            y = torch.flatten(Y.T)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())

            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            dl.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
            
# 训练模型并创作歌词，现在我们可以训练模型了。首先，设置模型超参数。
# 我们将根据前缀“分开”和“不分开”分别创作长度为50个字符（不考虑前缀长度）的一段歌词。我们每过50个迭代周期便根据当前训练的模型创作一段歌词。
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
# 采用随机采样训练模型并创作歌词
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, device, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)

# 采用相邻采样训练模型并创作歌词
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, device, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)