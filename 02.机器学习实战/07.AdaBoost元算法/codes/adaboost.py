# -*- encoding: utf-8 -*-

import numpy as np

def loadSimpData():
    dataMat = np.matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = 1.0
    return retArray

'''
单层决策树生成函数
'''
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMatrix = np.mat(classLabels).T
    m, n = np.shape(dataMatrix) #(5, 2)
    numSteps = 10.0
    # 用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    # 在所有数据集的特征上进行遍历
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        # 通过最大值和最小值计算步长
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                # 设置阈值
                threshVal = rangeMin + float(j) * stepSize
                # 根据数据集，特征，阈值计算返回分类预测结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 构建列向量，如果predicatedVals中的值不等于labelMatrix中的真正类别标签值，那么errArr的相应位置为1
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMatrix] = 0
                # 错误向量errArr和权重向量D的相应元素相乘并求和，得到weightedError
                weightedError = D.T * errArr
                ##print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f' %(i, threshVal, inequal, weightedError))
                # 将当前的错误率与巳有的最小错误率进行对比，如果当前的值较小，那么就在词典bestStump中保存该单层决策树
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst
    
dataArr, classLabels = loadSimpData()
D = np.mat(np.ones((5, 1)) / 5)
bestStump, minError, bestClasEst = buildStump(dataArr, classLabels, D)
##print(bestStump)
##print(minError)
##print(bestClasEst)

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    # 向量D包含每个数据点的权重，一开始初始化相等的值
    D = np.mat(np.ones((m, 1)) / m)
    # 记录每个数据点的类别估计累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        # 利用buildStump()函数找到最佳的单层决策树
        # 利用D得到的具有最小错误率的单层决策树，同时返回的还有最小的错误率以及估计的类别向量
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        ##print('D: ', D.T)
        ##print('error: ', error)
        # 总分类器本次单层决策树输出结果的权重 alpha = 0.5 * log((1.0-error) / error)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        # 将最佳单层决策树加入到单层决策树数组
        weakClassArr.append(bestStump)
        ##print('classEst: ', classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        # 在迭代中，AdaBoost算法会在增加错分数据的权重的同时，降低正确分类数据的权重。D是一个概率分布向量，所有元素之和为1.0
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 通过aggClassESt变量保持一个运行时的类别估计值来实现
        aggClassEst += alpha * classEst
        ##print('aggClassEst: ', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print('total error: ', errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr

dataArr, classLabels = loadSimpData()
classifierArray = adaBoostTrainDS(dataArr, classLabels, 9)
##print(classifierArray)
'''
[{'dim': 0, 'thresh': 1.3, 'ineq': 'lt', 'alpha': 0.6931471805599453}, 
 {'dim': 1, 'thresh': 1.0, 'ineq': 'lt', 'alpha': 0.9729550745276565}, 
{'dim': 0, 'thresh': 0.9, 'ineq': 'lt', 'alpha': 0.8958797346140273}]
'''

'''
利用训练出的多个弱分类器进行分类
'''
def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        # 每个弱分类器的结果以其对应的alpha值作为权重。所有这些弱分类器的结果加权求和就得到了最后的结果
        aggClassEst += classifierArr[i]['alpha'] * classEst
        ##print('aggClassEst: ', aggClassEst)
    # 返回aggClassEst的符号， 即如果aggClassEst大于0则返回+1,而如果小于0则返回-1
    return np.sign(aggClassEst)

dataArr, labelArr = loadSimpData()
classifierArray = adaBoostTrainDS(dataArr, labelArr, 30)
##print(classifierArray)
##print('ada classify1: ', adaClassify([0, 0], classifierArray))
##print('ada classify2: ', adaClassify([[5, 5], [0, 0]], classifierArray))

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMatrix = []
    labelMatrix = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMatrix.append(lineArr)
        labelMatrix.append(float(curLine[-1]))
    return dataMatrix, labelMatrix

dataArr, labelArr = loadDataSet('horseColicTraining.txt')
classifierArray = adaBoostTrainDS(dataArr, labelArr, 10)

testArr, testLabelArr = loadDataSet('horseColicTest.txt')
prediction10 = adaClassify(testArr, classifierArray)
errArr = np.mat(np.ones((len(prediction10), 1)))
numErr = errArr[prediction10 != np.mat(testLabelArr).T].sum()
numRate = float(numErr) / len(prediction10)
print('预测错误数量: %d, 预测错误率: %.2f' %(numErr, numRate))

