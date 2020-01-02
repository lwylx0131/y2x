# -*- encoding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random

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

def plotBestFit(weights):
    dataMatrix, labelMatrix = loadDataSet()
    # list转换为np.ndarray
    dataArr = np.array(dataMatrix)
    n = np.shape(dataArr)[0]
    # 分类标签为1时，将所有的x1放入到xcord1数组，所有的x2放入到ycord1数组
    # 分类标签为0时，将所有的x1放入到xcord2数组，所有的x2放入到ycord2数组
    # 注意dataMatrix第一列w0为默认1，即x0对应的模型参数值
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMatrix[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 将数据集x1、x2显示在坐标平面
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 60条数据记录
    x = np.arange(-3.0, 3.0, 0.1) 
    # 因为sigmoid函数在0处(0,0.5)是分类标签1/0的分界线，所以需要sigmoid函数的-WX=0
    # 即w0*x0+w1*x1+w2*x2=0，所以推出以下y的公式，即求x2=(-w0-w1*x1)/w2
    y = np.array(((-weights[0]-weights[1]*x) / weights[2]).transpose())
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
#plotBestFit(weights)

'''
梯度上升算法：
  在每次更新回归系数时都需要遍历整个数据集，如果只是处理100个左右数据尚可，但如果处理数十亿样本和成千上万的特征，那计算复杂度太高。
改进方法-随机梯度上升算法：
  一次仅使用一个样本点来更新回归系数，由于可以在新样本到来时对分类器进行增量式更新，因此随机梯度上升算法是一个在线学习算法。
'''
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix) # (100, 3)
    alpha = 0.01
    weights = np.ones(n) # (3,1)
    for i in range(m):
        # 单个样本点 wx=w0x0+w1x1+w2x2=(1,3)*(3*1)=(1,1)
        h = sigmoid(dataMatrix[i] * weights)
        error = classLabels[i] - h # (1, 1)
        # 更新模型参数 (3,1)+(1,)*(1,)*(3,1)=(3,1)
        weights = weights + alpha * error * dataMatrix[i]
    return weights

dataMatrix, labelMatrix = loadDataSet()
weights = stocGradAscent0(dataMatrix, labelMatrix)
print('Logistic回归随机梯度上升最优算法模型值:\n{}'.format(weights))
#plotBestFit(weights)

'''
以上随机梯度上升算法中，根据X0/X1/X2的迭代次数关系图（没画），在大的波动停止后，还有一些小的周期性波动。
产生这种现象的原因是存在一些不能正确分类的样本点（数据集并非线性可分），在每次迭代时会引发系数的剧烈改变。
我们期望算法能够避免来回波动，从而收敛到某个值。另外，收敛速度也需要加快，可通过以下算法改进。
'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        # 临时记录数据数量，当随机获取一条计算之后就删除该条记录
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机从数据集中获取数据记录
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(dataMatrix[randIndex] * weights)
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del[dataIndex[randIndex]]
    return weights

dataMatrix, labelMatrix = loadDataSet()
weights = stocGradAscent1(dataMatrix, labelMatrix, 500)
print('Logistic回归alpha变动随机梯度上升最优算法模型值:\n{}'.format(weights))
plotBestFit(weights)