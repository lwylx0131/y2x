# -*- encoding: utf-8 -*-

#%matplotlib inline
import torch
import numpy as np
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
在循环神经网络的梯度计算方法中，发现当时间步数较大或者时间步较小时，循环神经网络的梯度较容易出现衰减或爆炸。
虽然裁剪梯度可以应对梯度爆炸，但是无法解决梯度衰减的问题。
通常由于这个原因，循环神经网络在实际中较难捕捉时间序列中时间步距离较大的依赖关系。
门控循环神经网络(gated recurrent neural netword)的提出，正是为了更好的捕捉时间序列中时间步距离较大的依赖关系。
它通过可以学习的门来控制信息的流动。其中，门控循环单元(gated recurrent unit, GRU)是一种常用的门控循环神经网络。
它引入了重置门(reset gate)和更新门(update gate)的概念，从而修改了循环神经网络中隐藏状态的计算方式。
'''
# 获取歌词数据集词典
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = dl.load_data_jay_lyrics()

device = 'cpu'
# 初始化模型参数，隐藏单元个数num_hiddens是一个超参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
def get_params():  
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32) #正态分布
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    W_xz, W_hz, b_z = _three()  # 更新门参数
    W_xr, W_hr, b_r = _three()  # 重置门参数
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])

# 根据循环神经网络的计算表达式实现该模型。首先定义init_gru_state函数来返回初始化的隐藏状态，它返回由一个形状为（批量大小，隐藏单元个数）的值为0的NDArray组成的元祖。
# 使用元祖是为了更便于处理隐藏状态含有多个NDArray的情况。
def init_gru_state(batch_size, num_hiddens, device):   #隐藏状态初始化
    return (torch.zeros((batch_size, num_hiddens), device=device), )
	
# 根据门控循环单元的计算表达式定义模型
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + R * torch.matmul(H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)

# 训练模型时暂时只使用相邻采样，设置好超参数后，将训练模型并根据前缀“分开”和“不分开”分别创作长度为50个字符的一段歌词。
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
# 每过40个迭代周期便根据当前训练的模型创作一段歌词
dl.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens, vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
'''
epoch 40, perplexity 150.832332, time 4.72 sec
 - 分开 我想你你的爱爱 我想你你的爱爱女 我想你你的爱爱女 我想你你的爱爱女 我想你你的爱爱女 我想你你的
 - 不分开 我想你你的爱爱 我想你你的爱爱女 我想你你的爱爱女 我想你你的爱爱女 我想你你的爱爱女 我想你你的
epoch 80, perplexity 32.347916, time 4.80 sec
 - 分开 我想要这样 我不要再想 我不要再想 我不要再想 我不要再想 我不要再想 我不要再想 我不要再想 我
 - 不分开 爱不了我 你不了我 我不要再想 我不要再想 我不要再想 我不要再想 我不要再想 我不要再想 我不要
epoch 120, perplexity 4.779588, time 4.82 sec
 - 分开我想要        所有你烦 我有多烦恼  没有你在我有多难多多难恼  没有你烦 我有多烦恼  没
 - 不分开 你已经离不知我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋 后知后觉 我该好好生活 我该好好生
epoch 160, perplexity 1.438006, time 4.80 sec
 - 分开不想  没有你烦 我试著努力向你奔跑 爱才送到 你却已在别人怀抱 就是开不了口让她知道 就是那么简单
 - 不分开 你已经离开我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋 后知后觉 我该好好生活 我该好好生活
'''


# 训练模型并创作歌词，现在我们可以训练模型了。首先，设置模型超参数。
# 我们将根据前缀“分开”和“不分开”分别创作长度为50个字符（不考虑前缀长度）的一段歌词。我们每过50个迭代周期便根据当前训练的模型创作一段歌词。
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
lr = 1e-2 # 注意调整学习率
gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
model = dl.RNNModel(gru_layer, vocab_size).to(device)
dl.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx, 
                              num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
'''
epoch 50, perplexity 1.029284, time 2.79 sec
 - 分开的话像语言暴力 我已无能为力再提起 决定中断熟悉 然后在这里 不限日期 然后将过去 慢慢温习 让我爱
 - 不分开始打呼 管家是一只会说法语举止优雅的猪 吸血前会念约翰福音做为弥补 拥有一双蓝色眼睛的凯萨琳公主 专
epoch 100, perplexity 1.012104, time 2.91 sec
 - 分开的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我
 - 不分开球 想这样没担忧 唱着歌 一直走 我想就这样牵着你的手不放开 爱可不可以简简单单没有伤害 你 靠着我
epoch 150, perplexity 1.009273, time 2.73 sec
 - 分开的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我
 - 不分开球我妈妈 我说的话 你甘会听 不要再这样打我妈妈 难道你手不会痛吗 我叫你爸 你打我妈 这样对吗干嘛
epoch 200, perplexity 1.353881, time 2.60 sec
 - 分开的只剩下回忆 相爱还有  没有错亏我叫你爸我 爸爸 我想揍你已经很久 别想躲 我想揍你已经很久 别想
 - 不分开没心伤透 我知道 我想揍你已经离开我 说没有 我马儿有些瘦 天涯尽头 满脸风霜落寞 近乡情怯的我 相
epoch 250, perplexity 1.012780, time 2.75 sec
 - 分开的只剩下下笔 将真的没有悲哀 我 想带你骑单车 我 想带你骑单车 我 想和你看棒球 想这样对吗干嘛这
 - 不分开没心跳你知道  杵在伊斯坦堡 却只想你和汉堡 我想要你的微笑每天都做得到 但那个人已经不是我不要再想
'''

# 门控循环神经网络可以更好的捕捉时间序列中的时间步距离较大的依赖关系。
# 门控循环单元引入了门的概念，从而修改了循环神经网络中隐藏状态的计算方式。它包括重置门、更新门、候选隐藏状态和隐藏状态。
# 重置门有助于捕捉时间序列里短期的依赖关系。
# 更新门有助于捕捉时间序列里长期的依赖关系。