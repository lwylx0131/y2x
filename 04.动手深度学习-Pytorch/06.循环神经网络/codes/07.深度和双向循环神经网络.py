# -*- encoding: utf-8 -*-

#%matplotlib inline
import torch
import numpy as np
import sys
sys.path.append('../..')
import dl_common_pytorch as dl
from torch import nn
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

# 在深度学习应用里，通常会用到含有多个隐藏层的循环神经网络，也称作深度循环神经网络。
# 隐藏状态的信息不断传递至当前层的下一时间步和当前时间步的下一层。
# 在之前介绍的循环神经网络模型都是假设当前时间步是由前面的较早时间步的序列决定的，因此它们都将信息通过隐藏状态从前往后传递。
# 有时候，当前时间步也可能由后面时间步决定。例如，当我们写下一个句子时，可能会根据句子后面的词来修改句子前面的用词。
# 双向循环神经网络通过增加从后往前传递信息的隐藏层来更灵活的处理这类信息。
# 双向循环神经网络在每个时间步的隐藏状态同时取决于该时间步之前和之后的子序列（包括当前时间步的输入）。

# 获取歌词数据集词典
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = dl.load_data_jay_lyrics()

device = 'cpu'

# 训练模型时暂时只使用相邻采样，设置好超参数后，将训练模型并根据前缀“分开”和“不分开”分别创作长度为50个字符的一段歌词。
num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
lr = 1e-2 # 注意调整学习率

# 深度循环神经网络
gru_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens,num_layers=2)
model = dl.RNNModel(gru_layer, vocab_size).to(device)
dl.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)

gru_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens,num_layers=6)
model = dl.RNNModel(gru_layer, vocab_size).to(device)
dl.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)

# 双向循环神经网络
num_hiddens=128
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e-2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
lr = 1e-2 # 注意调整学习率

gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens,bidirectional=True)
model = dl.RNNModel(gru_layer, vocab_size).to(device)
dl.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)