# -*- encoding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

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
    
plotBestFit(weights)