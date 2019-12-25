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
    hwLabels = []
    # 初始化1024列的矩阵，因为一个数字文件有32行32列
    trainingMatrix = np.zeros((m, 32 * 32))
    for i in range(m):
        # 文件名格式为9_102.txt
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        # 从文件名中解析出数字标签9
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将每个文件的内容解析为一行放入到trainingMatrix矩阵中作为训练数据
        trainingMatrix[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        # 从文件名中解析出数字标签真实值
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        # 通过分类器判别出的数字标签值
        classifierResult = classify0(vectorUnderTest, trainingMatrix, hwLabels, 3)
        print('[%d]-数字图像的真实标签值: %d, 分类器判断的数字图像标签值: %d' %(i, classNumStr, classifierResult))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print('分类错误数: %d, 分类错误率: %f' %(errorCount, errorCount / float(mTest)))

handwritingClassTest()
'''
实际使用这个算法时，算法的执行效率并不高。因为算法需要为每个测试向量做2000次距离计算，
每个距离计算包括了1024个维度浮点运算，总计要执行900次，此外，我们还需要为测试向量准备2 M B的存储空间。
是否存在一种算法减少存储空间和计算时间的开销呢？ k决策树就是A-近邻算法的优化版，可以节省大量的计算开销。

k-近邻算法是分类数据最简单最有效的算法，本章通过两个例子讲述了如何使用k-近邻算法构造分类器。
k-近邻算法是基于实例的学习，使用算法时我们必须有接近实际数据的训练样本数据。
k-近邻算法必须保存全部数据集，如果训练数据集的很大，必须使用大量的存储空间。
此外,由于必须对数据集中的每个数据计算距离值，实际使用时可能非常耗时。
k-近邻算法的另一个缺陷是它无法给出任何数据的基础结构信息，因此我们也无法知晓平均实例样本和典型实例样本具有什么特征。
下一章我们将使用概率测量方法处理分类问题，该算法可以解决这个问题
'''