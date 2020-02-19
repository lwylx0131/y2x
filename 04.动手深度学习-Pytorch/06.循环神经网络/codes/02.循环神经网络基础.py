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
with zipfile.ZipFile('jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
corpus_chars[:40]