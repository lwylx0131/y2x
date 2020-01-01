# -*- encoding: utf-8 -*-
import numpy as np

def loadDataSet():
    # 训练数据集100*3的特征矩阵
    dataMatrix = []
    # 训练数据集100*1的标签矩阵
    labelMatrix = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # X0默认为1，X1，X2的特征值
        dataMatrix.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMatrix.append(int(lineArr[2]))
    return dataMatrix, labelMatrix

def sigmoid(wx):
    return 1.0 / (1 + np.exp(-wx))

# Logistic回归梯度上升优化算法
def gradAscent(dataMatrix, labelMatrix):
    # 先将numpy的2维数组转换为矩阵类型
    dataMatrix = np.mat(dataMatrix) # (100,3)
    labelMatrix = np.mat(labelMatrix).transpose() # (100,1) 需要转置为列
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    # 初始化w模型参数为1，因为有3个模型，所以3个参数
    weights = np.ones((n, 1)) # (3,1)
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights) # (100,3)*(3,1)=(100,1)
        error = labelMatrix - h # (100,1)
        # 更新模型参数 (3,1)+(1,)*(3,100)*(100,1)=(3,1)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

dataMatrix, labelMatrix = loadDataSet()
weights = gradAscent(dataMatrix, labelMatrix)
print('Logistic回归梯度上升最优算法模型值:\n{}'.format(weights))