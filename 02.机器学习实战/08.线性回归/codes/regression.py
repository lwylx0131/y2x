# -*- encoding: utf-8 -*-

import numpy as np

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse.")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

xArr, yArr = loadDataSet('ex0.txt')
ws = standRegres(xArr, yArr)
print(ws)

# 变量ws存放的就是回归系数，在用内积预测y = ws[0]*X0 + ws[1]*X1 X0默认为1
# 使用模型ws来预测yHat
import matplotlib.pyplot as plt

xMat = np.mat(xArr)
yMat = np.mat(yArr)
yHat = xMat * ws

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(xMat[:, 1].flatten().A[0], yMat[:, 0].flatten().A[0])
ax.scatter(xMat[:,1].getA(), yMat.getA()[0])
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
#plt.show()

# 如果其他数据集也可以使用相同的ws建模，那么如何判断这些模型的好坏
# 可以计算预测值yHat序列和真实值y序列的匹配程度，即计算两个序列的相关系数
print(yHat.shape, yMat.shape)
print(np.corrcoef(yHat.T, yMat))
'''
该矩阵包含所有两两组合的相关系数。可以看到，对角线上的数据是1.0，因为鸿社和自己的匹配是最完美的，而别3七和挪3七的相关系数为0.98。
[[1.         0.13653777]
 [0.13653777 1.        ]]
'''

# 局部加权线性回归函数
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('this matrix is singular, cannot do inverse.')
        return 
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

xArr, yArr = loadDataSet('ex0.txt')
print(yArr[0])
print(lwlr(xArr[0], xArr, yArr, 1.0))
print(lwlr(xArr[0], xArr, yArr, 0.001))
yHat = lwlrTest(xArr, xArr, yArr, 0.01)
print('yHat = ', yHat)

xMat = np.mat(xArr)
srtInd = xMat[:, 1].argsort(0)
xSort = xMat[srtInd][:, 0, :]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1], yHat[srtInd])
ax.scatter(xMat[:, 1].getA(), yMat.getA()[0], s=2, c='red')
plt.show()
'''
使用3种不同平滑值绘出的局部加权线性回归结果。上图中的平滑参数&=1.0,
中图* = 0 .0 1 ,下图女=0.003。可以看到，灸=1.0时的模型效果与最小二乘法差
不多，4=0.01时该模型可以挖出数据的潜在规律，而々=0.003时则考虑了太多
的噪声，进而导致了过拟合现象。
'''