# -*- encoding: utf-8 -*-

import torch
import numpy as np

# ReLU(rectified linear unit)函数提供了一个简单的非线性变换，给定元素x，该函数定义为：
# ReLU(x) = max(x, 0) 只保留正数元素，并将负数元素清零。