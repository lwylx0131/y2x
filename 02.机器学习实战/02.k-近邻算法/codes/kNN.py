# -*- encoding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

'''
1) inX: np.array([0.5, 0.5]) 表示一条记录输入向量，即属性特征值；
2) 该方法相当于一个分类器了，可以完成很多分类任务；
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # tile([1, 2], (m, n)) 将数组[1, 2]横向复制n次，纵向复制m行
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # axis=1表示按列进行计算求和，相当于计算整行的和值
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 将数组的元素升序排序，打印出升序元素的索引号
    # [0.61, 0.5, 0.5, 0.41] => [3, 1, 2, 0]
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortedDistIndicies[i]]
        classCount[label] = classCount.get(label, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    #print(sortedClassCount)
    return sortedClassCount[0][0]
    
#dataSet, labels = createDataSet()
#print(dataSet)
#print(classify0(np.array([0.5, 0.5]), dataSet, labels, 2))

'''
1) 将文本记录转换为numpy；
2) numpy包含两个部分：特征值矩阵向量，标签向量
'''
def file2matrix(fileName):
    fr = open(fileName)
    numberOfLines = len(fr.readlines())
    returnMatrix = np.zeros((numberOfLines, 3))
    fr = open(fileName)
    classLabelVector = []
    index = 0
    for line in fr.readlines():
        lineList = line.strip().split('\t')
        returnMatrix[index, :] = lineList[0:3]
        classLabelVector.append(lineList[-1])
        index += 1
    return returnMatrix, classLabelVector

'''
1) 使用matplotlib绘制散点图
'''
fileName = 'datingTestSet2.txt'
dataSet, classLabelVector = file2matrix(fileName)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataSet[:, 0], dataSet[:, 1])
#plt.show()

'''
1) 归一化特征值：newValue = (oldValue - min) / (max - min)
'''
def autoNorm(dataSet):
    # min中的参数0使得函数可以从列中选取最小值，max则选择最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    row = dataSet.shape[0]
    minDataSet = np.tile(minVals, (row, 1))
    rangesDataSet = np.tile(ranges, (row, 1))
    normDataSet = (dataSet - minDataSet) / rangesDataSet
    return normDataSet, ranges, minVals
    
#print(autoNorm(dataSet))

'''
1) 测试分类器；
'''
def datingClassTest():
    hoRatio = 0.10
    dataSet, labels = file2matrix('datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(dataSet)
    row = normDataSet.shape[0]
    numTestVals = int(row * hoRatio)
    errorCount = 0.0
    for i in range(numTestVals):
        # 输入特征值，进行分类
        inputX = normDataSet[i, :]
        # 获取90%的训练数据
        trainDataSet = normDataSet[numTestVals:row, :]
        trainLabels = labels[numTestVals:row]
        # 对inputX进行分类，得到一个分类标签
        classifierResult = classify0(inputX, trainDataSet, trainLabels, 3)
        print('[%d] inputX的分类结果是: %s, 其真实标签结果是: %s' %(i, classifierResult, labels[i]))
        if(classifierResult != labels[i]):
            errorCount += 1.0
    errorRatio = errorCount / float(numTestVals)
    print('knn分类算法的错误率为: %0.4f' % (errorRatio))
    
datingClassTest()

'''
1) 将图像转换为向量，把一个32*32的二进制图像矩阵转换为1*1024的向量；
'''
def img2vector(fileName):
    returnVector = np.zeros((1, 1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0, 32*i+j] = int(lineStr[j])
    return returnVector

testVector = img2vector('digits/testDigits/0_13.txt')
#print(testVector[0, 0:31])

'''
1) 手写数字识别系统测试；
'''
def handwritingClassTest():
    trainingFileList = os.listdir('digits/trainingDigits')
    m = len(trainingFileList)
    for i in trainingFileList:
        print(i)

handwritingClassTest()