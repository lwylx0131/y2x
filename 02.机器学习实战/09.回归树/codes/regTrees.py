# -*-encoding: utf-8 -*-

import numpy as np

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将每行映射为float浮点类型
        fltLine = np.map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    # nonzero(a)返回数组a中值不为零的元素的下标 
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :][0]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left') = createTree(lSet, leafType, errType, ops)
    retTree['right') = createTree(rSet, leafType, errType, ops)
    return retTree

def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def chooseBestSplit（dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if(S - bestS) < tolS:
        # 如果误差减少不大则退出
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        # 如果切分出的数据集很小则退出
        return None, leafType(dataSet)
    return bestIndex, bestValue

myDat = loadDataSet('ex00.txt')
myMat = np.mat(myDat)
bestIndex, bestValue = createTree(myMat)

