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
#plt.show()
'''
使用3种不同平滑值绘出的局部加权线性回归结果。上图中的平滑参数&=1.0,
中图* = 0 .0 1 ,下图女=0.003。可以看到，灸=1.0时的模型效果与最小二乘法差
不多，4=0.01时该模型可以挖出数据的潜在规律，而々=0.003时则考虑了太多
的噪声，进而导致了过拟合现象。
'''

# 预测鲍鱼年龄
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

abX, abY = loadDataSet('abalone.txt')
yHat01 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 0.1)
yHat1 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 1)
yHat10 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 10)
#比较分析预测误差大小
print(rssError(abY[0: 99], yHat01.T))
print(rssError(abY[0: 99], yHat1.T))
print(rssError(abY[0: 99], yHat10.T))
'''
可以看到,使用较小的核将得到较低的误差。那么,为什么不在所有数据集上都使用最小的核呢？
这是因为使用最小的核将造成过拟合，对新数据不一定能达到最好的预测效果。
56.78420911837208
429.89056187030394
549.1181708826065
'''
yHat01 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 0.1)
yHat1 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 1)
yHat10 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 10)
print(rssError(abY[100: 199], yHat01.T))
print(rssError(abY[100: 199], yHat1.T))
print(rssError(abY[100: 199], yHat10.T))
'''
从以下结果可以看到，在上面的三个参数中，核大小等于10时的测试误差最小，但它在训练集上的误差却是最大的。
25119.459111157415
573.5261441895706
517.5711905381745
'''

# 使用简单的线性回归做比较
ws = standRegres(abX[0: 99], abY[0: 99])
yHat = np.mat(abX[100: 199]) * ws
print(rssError(abY[100: 199], yHat.T.A))

'''
本例展示了如何使用局部加权线性回归来构建模型，可以得到比普通线性回归更好的效果。
局部加权线性回归的问题在于，每次必须在整个数据集上运行。也就是说为了做出预测，必须保
存所有的训练数据。下面将介绍另一种提高预测精度的方法，并分析它的优势所在。
'''

'''
如果数据的特征比样本点还多应该怎么办？是否还可以使用线性回归和之前的方法来做预
测？答案是否定的，即不能再使用前面介绍的方法。这是因为在计算^ 乂广的时候会出错。
如果特征比样本点还多（n > m ) ,也就是说输入数据的矩阵乂不是满秩矩阵。非满秩矩阵在
求逆时会出现问题。为了解决这个问题，统计学家引入了岭回归（ridgeregression)的概念。
'''

def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print('this matrix is singular, cannot do inverse.')
        return
    # denom.I 表示求逆矩阵
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    # 方差
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

abX, abY = loadDataSet('abalone.txt')
ridgeWeights = ridgeTest(abX, abY)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
#plt.show()

'''
前向逐步回归算法可以得到与13^ 0差不多的效果，但更加简单。它属于一种贪心算法，即每
一步都尽可能减少误差。一开始，所有的权重都设为1，然后每一步所做的决策是对某个权重增
加或减少一个很小的值。
'''
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # mean(0) 对各列求平均值 mean(1) 对各行求平均值
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xVar = np.var(xMat, 0)
    xMeans = np.mean(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat
    
xArr, yArr = loadDataSet('abalone.txt')
stageWise(xArr, yArr, eps=0.01, numIt=200)
'''
[[0. 0. 0. 0. 0. 0. 0. 0.]]
[[0.   0.   0.   0.01 0.   0.   0.   0.  ]]
[[0.   0.   0.   0.02 0.   0.   0.   0.  ]]
...
[[ 0.04  0.    0.09  0.03  0.31 -0.64  0.    0.36]]
[[ 0.05  0.    0.09  0.03  0.31 -0.64  0.    0.36]]
[[ 0.04  0.    0.09  0.03  0.31 -0.64  0.    0.36]]
上述结果中值得注意的是wl和w6都是0 ,这表明它们不对目标值造成任何影响，也就是说这
些特征很可能是不需要的。另外，在参数eps设置为0.01的情况下，一段时间后系数就已经饱和
并在特定值之间来回震荡，这是因为步长太大的缘故。这里会看到，第一个权重在0.04和0.05之
间来回震荡。

逐步线性回归算法的实际好处并不在于能绘出图8-7这样漂亮的图’ 主要的优点在于它可以帮助
人们理解现有的模型并做出改进。当构建了一个模型后，可以运行该算法找出重要的特征，这样就
有可能及时停止对那些不重要特征的收集。最后，如果用于测试，该算法每100次迭代后就可以构建
出一^ 模型，可以使用类似于10折交叉验证的方法比较这些模型，最终选择使误差最小的模型。
'''