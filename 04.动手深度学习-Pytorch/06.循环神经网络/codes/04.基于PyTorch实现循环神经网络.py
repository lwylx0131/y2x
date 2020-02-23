# -*- encoding: utf-8 -*-

#%matplotlib inline
import torch
import sys
sys.path.append('../..')
import dl_common_pytorch as dl
from torch import nn
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

# 获取歌词数据集词典
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = dl.load_data_jay_lyrics()

# 定义模型，PyTorch中的nn模块提供了循环神经网络的实现。下面构造一个含单隐藏层、隐藏单元个数为256的循环神经网络层rnn_layer
num_hiddens = 256
# rnn_layer作为nn.RNN实例，在前向计算后会分别返回输出和隐藏状态h，其中输出指的是隐藏层在各个时间步上计算输出的隐藏状态，它们通常作为后续输出层的输入。
# 该“输出”本身并不涉及输出层的计算，形状为(时间步数, 批量大小, 隐藏单元个数)。
# 而nn.RNN实例在前向计算返回的隐藏状态指隐藏在最后时间步的隐藏状态：当隐藏层有多层时，每一层的隐藏状态都会记录在该变量中。
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
# 使用权重为随机值的模型进行预测
device = 'cpu'
model = dl.RNNModel(rnn_layer, vocab_size).to(device)
predict_str = dl.predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)
print('预测的字符串：' + predict_str)
# 预测的字符串：分开雨原原原雨瓣原原原雨

num_steps = 35
num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
dl.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx, 
                                 num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
'''
epoch 50, perplexity 8.049565, time 1.31 sec
 - 分开的我 你不的 我 你不能再想你 想要你的我我妈你 想你你 我想要你想你我想 你不你 想你你想我不想 
 - 不分开 我想你你你 我不要再想你 想要你的我我不你 想你你 我不要再想你我想要你想你我想你 想你你的 我有
epoch 100, perplexity 1.253759, time 1.32 sec
 - 分开不会  又过 连隔壁 印地安老斑鸠 腿短毛不多 几天都没有喝水也能活 脑袋瓜有一点秀逗 猎物死了它比
 - 不分开 我的爱你已 我想的太爱 你在抽痛 还我一口被废弃的白蛦丘 站着一只饿昏的老斑鸠 印地安老斑鸠 腿短
epoch 150, perplexity 1.062346, time 1.34 sec
 - 分开 我的爸爸你的打我妈妈 想你想我 想你想要 是你听的你爸 你怎么面不要 想你我爱 我的了口被你拆封 
 - 不分开 我的爱你走 我想的这样牵着我的手不放开 爱可不可以简简单单没有伤害 你 靠着我的肩膀 你 在我胸口
epoch 200, perplexity 1.031426, time 1.32 sec
 - 分开 我的爸爸 像打日妈 你想我有多重  我感动这些我编 有想的事爱 有怀讽刺 有伤中的可以女巫默黄许愿
 - 不分开 我的爱你走 我对不会掩护我 选你这种队友 我不了我 不再不要 你在我爱 你的那画面的经 想要你的陪
epoch 250, perplexity 1.019656, time 1.30 sec
 - 分开 我的爸爸种像 它你看 我 你爸 你打我妈妈 我不懂爸不想 我妈没有多说 有有没有兵器我喜欢 双截棍
 - 不分开 我的爱口走 害怕  从能到大你语沉  我想你 你以  但不知不觉 你跟了它前口的响尾蛇 无力的躺在
'''