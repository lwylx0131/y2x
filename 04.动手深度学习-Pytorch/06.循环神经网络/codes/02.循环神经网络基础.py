# -*- encoding: utf-8 -*-

#%matplotlib inline
import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
import dl_common_pytorch as dl
import torch.nn as nn
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
print(corpus_chars[:40])
# 数据集有6w多个字符，为打印方便，把换行符替换成空格，然后使用前1w个字符来训练模型
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0: 10000]
# 将每个字符映射成一个从0开始的连续整数，又称索引，来方便之后的数据处理。
# 为得到索引，将数据集里所有不同字符取出来，然后将其逐一映射到索引来构造词典。
# 打印vocab_size，即词典中不同字符的个数，又称词典大小。
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print('词典大小 = %d' %(vocab_size)) # 词典大小 = 1027

# 将训练数据集中每个字符转化为索引，并打印前20个字符及其对应的索引
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars : ', ''.join([idx_to_char[idx] for idx in sample]))
print('indices : ', sample)
'''
chars :  想要有直升机 想要和你飞到宇宙去 想要和
indices :  [384, 267, 473, 976, 19, 302, 613, 384, 267, 232, 210, 511, 152, 733, 659, 669, 613, 384, 267, 232]
'''

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

my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')

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

for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')