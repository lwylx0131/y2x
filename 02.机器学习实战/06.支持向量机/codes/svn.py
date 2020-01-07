# -*- encoding: utf-8 -*-

import numpy as np
import random

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    # 随机获取一个范围在0~m且不等于i的整数j
    while(j == i):
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj == L
    return aj

'''
1)数据集
2)类别标签
3)常数C
4)容错率
5)取消前最大的循环次数
'''
def smoSimple(dataMat, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMat) # (100, 2)
    labelMatrix = np.mat(classLabels).transpose() # (100, 1)
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1))) # (100, 1)
    iteri = 0
    while(iteri < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # ((100, 1) * (100, 1)).T * ((100, 2) * (1, 2).T) = (1, 100) * (100, 1) = (1,1)
            # 预测的类别
            fXi = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 这个实例的预测结果与真实结果对比，计算误差，如果误差很大，那么可以对该数据实例所对应的alpha值进行优化
            Ei = fXi - float(labelMatrix[i])
            # 检测
            if((labelMatrix[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMatrix[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机获取j，目的是为了随机选择第二个alpha
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMatrix[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 保证alpha在0与C之间
                if(labelMatrix[i] != labelMatrix[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[j] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L == H')
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue
                alphas[i] += labelMatrix[j] * labelMatrix[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if(0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0 < alphas[j] and (C > alphas[j])):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter: %d i: %d, pairs changed %d' %(iteri, i, alphaPairsChanged))
        if(alphaPairsChanged == 0):
            iteri += 1
        else:
            iteri = 0
        print('iteration number: %d' %iteri)
    return b, alphas
                
dataMat, classLabels = loadDataSet('testSet.txt')
b, alphas = smoSimple(dataMat, classLabels, C=0.6, toler=0.001, maxIter=40)
print('========================================')
for i in range(100):
    if alphas[i] > 0.0:
        print(dataMat[i], classLabels[i])
    
    
    
    
    
    