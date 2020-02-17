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

train_data = pd.read_csv('kaggle_house/train.csv')
test_data = pd.read_csv('kaggle_house/test.csv')
print(train_data.shape) # (1460, 81)
print(test_data.shape) # (1459, 80)

# 查看前4个样本的前4个特征、后2个特征和标签(SalePrice)
train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]

# 将所有训练数据和测试数据的79个特征按样本连接到一起
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 对连续数值的特征做标准化：设该特征在整个数据集上的均值为u，标准差为e