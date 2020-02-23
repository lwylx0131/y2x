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

'''
长短期记忆long short-term memory :
输入门(input gate)
遗忘门(forget gate):控制上一时间步的记忆细胞 输入门:控制当前时间步的输入
输出门(output gate):控制从记忆细胞到隐藏状态
记忆细胞：某种特殊的隐藏状态的信息的流动，隐藏状态形状相同的记忆细胞，从而记录额外的信息
'''
# 获取歌词数据集词典
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = dl.load_data_jay_lyrics()

device = 'cpu'
# 初始化模型参数，隐藏单元个数num_hiddens是一个超参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), 
            torch.zeros((batch_size, num_hiddens), device=device))

# LSTM计算模型，只有隐藏状态会传递到输出层，而记忆细胞不参与输出层的计算
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)

# 训练模型时暂时只使用相邻采样，设置好超参数后，将训练模型并根据前缀“分开”和“不分开”分别创作长度为50个字符的一段歌词。
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

dl.train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens, vocab_size, device, corpus_indices, idx_to_char,
                         char_to_idx, False, num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
'''
epoch 40, perplexity 213.445825, time 5.69 sec
 - 分开 我不不 我不你 我不你 我不你 我不你 我不你 我不你 我不你 我不你 我不你 我不你 我不你 我
 - 不分开 我不不 我不你 我不你 我不你 我不你 我不你 我不你 我不你 我不你 我不你 我不你 我不你 我
epoch 80, perplexity 63.485917, time 5.81 sec
 - 分开 我想你这你的我 有知你 我想我 我不要 我不要 我不要 我不好 我不好 我不好这 我不好觉 我不不
 - 不分开 我想你你的你 我不要这 我不要 我不要 我不要 我不好 我不好 我不好这 我不好觉 我不好这生 我
epoch 120, perplexity 14.353536, time 5.79 sec
 - 分开 我想你的话笑 一悔  又来了 我想就这样牵着你的手不放开 爱可不可以简单单没有 害  靠是我的肩膀
 - 不分开 我要你的你笑 一定  又情了 我想就这样牵着你的手不放开 爱可不可以简单单没有 害  靠是我的肩膀
epoch 160, perplexity 3.616201, time 5.77 sec
 - 分开 一直是 是你的脚空 老制盘 旧皮了 装属了明信片的铁盒里藏著一片玫瑰花瓣 黄不葛爬满了雕花的门窗 
 - 不分开你你 我想和这样经我妈妈 难道你手 你在一定 一人就这样样我妈妈 感的让心你你你 想要再这样打我妈妈
'''

num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率
lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = dl.RNNModel(lstm_layer, vocab_size)
dl.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx, 
                                  num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
'''
epoch 40, perplexity 1.018091, time 3.24 sec
 - 分开始乡相信命运 感谢地心引力 让我碰到你 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让
 - 不分开 爱爱的你看棒球 想这样没担忧 唱着歌 一直走 我想就这样牵着你的手不放开 爱可不可以简简单单没有伤
epoch 80, perplexity 1.011709, time 3.35 sec
 - 分开 我爱的看不下  以为我较细汉 从小到大只有妈妈的温暖  为什么我爸爸 那么凶 如果真的我有一双翅膀
 - 不分开 爱能看到到你的睡著一直到老 就是开不了口让她知道 就是那么简单几句 我办不到 整颗心悬在半空 我只
epoch 120, perplexity 1.014148, time 3.41 sec
 - 分开 爱能不能够永远单纯没有悲哀 我 想带你骑单车 我 想和你看棒球 想这样没担忧 唱着歌 一直走 我想
 - 不分开 爱能不能够永远单纯没有悲哀 我 想带你骑单车 我 想和你看棒球 想这样没担忧 唱着歌 一直走 我想
epoch 160, perplexity 1.010713, time 3.29 sec
 - 分开 爱能不能够永远单纯没有悲哀 我 想带你骑单车 我 想和你看棒球 想这样没担忧 唱着歌 一直走 我想
 - 不分开 爱才送到 你却已在别人怀抱 就是开不了口让她知道 我一定会呵护著你 也逗你笑 你对我有多重要 我后
'''

# LSTM的隐藏层输出包括隐藏状态和记忆细胞，只有隐藏状态会传递到输出层。
# LSTM的输入门、遗忘门和输出门可以控制信息的流动。
# LSTM可以应对循环神经网络的梯度衰减问题，并更好的捕捉时间序列中时间步距离较大的依赖关系。