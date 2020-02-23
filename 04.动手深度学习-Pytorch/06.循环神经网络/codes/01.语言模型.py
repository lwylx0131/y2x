# -*- encoding: utf-8 -*-

#%matplotlib inline
import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
import dl_common_pytorch as dl
import torch.nn as nn
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

'''
语言模型：
一段自然语言文本可以看作是一个离散时间序列，给定一个长度为的词的序列w1,w2,...,wt，语言模型的目标就是评估该序列是否合理，即计算该序列的概率：
P(w1,w2,w3,...,wt)

假设序列w1,w2,...,wt中的每个词是依次生成的，则有：
P(w1,w2,w3,...,wt) = P(w1)P(w2|w1)...P(wt|w1w2...wt-1)
列如，一段含有4个词的文本序列的概率：
P(w1,w2,w3,...,wt) = P(w1)P(w2|w1)P(w3|w1,w2)P(w4|w1,w2,w3)
其中，语言模型的参数就是词的概率以及给定前几个词情况下的条件概率。P(w1) = n(w1) / n，
类似地，给定w1，w2的条件概率为：P(w2|w1) = n(w1,w2) / n(w1)，其中n(w1,w2)为语料库中以w1作为第一个词，w2作为第二个词的文本的数量。

n元语法(n-grams)，它是基于n-1阶马尔科夫链的概率语言模型。例如，当n=2，含有4个词的文本序列的概率就可以改写为：
P(w1,w2,w3,...,wt) = P(w1)P(w2|w1)P(w3|w1,w2)P(w4|w1,w2,w3)
                   = P(w1)P(w2|w1)P(w3|w2)P(w4|w3)
其中，当n分别为1,2和3时，分别称作一元语法（unigram）、二元语法（bigram）和三元语法（trigram）。
例如，长度为4的序列21,w2,w3,w4在一元语法、二元语法和三元语法中的概率分别为：
P(w1,w2,w3,w4) = P(w1)P(w2)P(w3)P(w4)
P(w1,w2,w3,w4) = P(w1)P(w2|w1)P(w3|w2)P(w4|w3)
P(w1,w2,w3,w4) = P(w1)P(w2|w1)P(w3|w1,w2)P(w4|w1,w2,w3)
当n较小时，n元语法往往并不准确。例如，在一元语法中，由三个词组成的句子“你走先”和“你先走”的概率是一样的。
然而，当n较大时，n元语法需要计算并存储大量的词频和多词相邻频率。
思考：元语法可能有哪些缺陷？
1）参数空间过大
2）数据稀疏
'''
