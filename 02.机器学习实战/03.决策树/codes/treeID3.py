# -*- encoding: utf-8 -*-
'''
在划分数据集之前之后信息发生的变化称为信息增益，知道如何计算信息增益，
我们就可以计算每个特征值划分数据集获得的信息增益，
获得信息增益最高的特征就是最好的选择。
集合信息的度量方式称为香农熵或者简称为熵，这个名字来源于信息论之父克劳德•香农。

如果待分类的事务可能划分在多个分类之中则，符号xi的信息定义为：
    l(xi) = -log2p(xi)
p(xi): 表示选择该分类的概率

熵的公式：
H = -[求和符号]p(xi) * log2p(xi)
i: 表示1~n，表示分类的数目
'''
import math
import matplotlib.pyplot as plt

'''
1)详细介绍决策树：https://blog.csdn.net/jiaoyangwm/article/details/79525237#311__83
2)计算给定数据的香农熵（经验熵）;
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 以字典的形式统计标签类的数量: {'yes': 10, 'no': 20} 
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率来计算类别出现的概率
        prob = float(labelCounts[key]) / numEntries
        # 使用香农熵的公式进行计算
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'], [1,1,'yes'], [1,0,'no'], [0,1,'no'], [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

dataSet, labels = createDataSet()
print(calcShannonEnt(dataSet)) # 0.9709505944546686
        
'''
如何划分数据集：
1)首先选取第一个特征的第一个可能取值来筛选信息。
2)再选取第一个特征的第二个可能的取值来划分我们的信息。
3)之后我们再选取第二个特征的第一个可能的取值来划分数据集，以此类推。
e.g: 
[[1, 1, ‘yes’], [1, 1, ‘yes’], [1, 0, ‘no’], [0, 1, ‘no’], [0, 1, ‘no’]] 
如果我们选取第一个特征值也就是需不需要浮到水面上才能生存来划分我们的数据，这里生物有两种可能，1就是需要，0就是不需要。那么第一个特征的取值就是两种。
如果我们按照第一个特征的第一个可能的取值来划分数据也就是当所有的样本的第一列取1的时候满足的样本，那就是如下三个： 
[1, 1, ‘yes’], [1, 1, ‘yes’], [1, 0, ‘no’] 
可以理解为这个特征为一条分界线，我们选取完这个特征之后这个特征就要从我们数据集中剔除，因为要把他理解为分界线。那么划分好的数据就是：
[[1, ‘yes’], [1, ‘yes’], [0, ‘no’]]
如果我们以第一个特征的第二个取值来划分数据集，也就是当所有样本的第二列取1的时候满足的样本，那么就是：
[[1, 1, ‘yes’], [1, 1, ‘yes’], [0, 1, ‘no’], [0, 1, ‘no’]] 
那么得到的数据子集就是下面这个样子： 
[[1,’yes’],[1,’yes’],[1, ‘no’], [1, ‘no’]]

dataSet：待划分的数据集
axis：划分数据集的特征，eg：[1,1,'yes'] axis=0，表示第一个特征，axis=1，表示第二个特征
value：特征的返回值
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # 如果某个特征和我们指定的特征相同，那么除去这个特征创建一个新的子特征
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            # a=[1,2,3] b=[4,5,6] => [1,2,3,4,5,6]
            reducedFeatVec.extend(featVec[axis+1:])
            # a=[1,2,3] b=[4,5,6] => [1,2,3,[4,5,6]]
            retDataSet.append(reducedFeatVec)
    return retDataSet

print(dataSet) # [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
# 划分出第二个特征并指定值为1的数据集
#print(splitDataSet(dataSet, 1, 1)) # [[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']]

'''
1)循环计算香农熵和splitDataSet()函数，找到最好的特征划分方式
'''
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    # 计算整个数据集的原始香农熵
    baseEntropy = calcShannonEnt(dataSet)
    #存储最好的信息增益
    bestInfoGain = 0.0
    bestFeatureIndex = -1
    for i in range(numFeature):
        # 将dataSet所有数据集的第i个特征值一一遍历放到list，再使用set去重
        featValList = [example[i] for example in dataSet]
        uniqueVals = set(featValList)
        # 经验条件熵
        newEntropy = 0.0
        # 将dataSet数据集的第i个特征对应的所有特征值一一划分数据集并计算对应的香农熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            #print('len(subDataSet) = %d, len(dataSet) = %d' %(len(subDataSet), len(dataSet)))
            prob = float(len(subDataSet)) / float(len(dataSet))
            #根据公式计算出经验条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print('第[%d]个特征的信息增益为: %.3f' %(i, infoGain))
        if(infoGain > bestInfoGain):
            #在dataSet数据集中，计算所有特征的经验条件熵得到信息增益后进行比较，找到最大的信息增益
            bestInfoGain = infoGain
            #记录信息增益最大的特征索引值
            bestFeatureIndex = i
            print('相比之后第[%d]个特征的最大特征值的信息增益为: %.3f' %(i, bestInfoGain))
    return bestFeatureIndex

print('相比之后最好的信息增益对应的特征索引值为: %d' %(chooseBestFeatureToSplit(dataSet)))
'''
[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
第[0]个特征的信息增益为: 0.420
相比之后第[0]个特征的最大特征值的信息增益为: 0.420
第[1]个特征的信息增益为: 0.171
相比之后最好的信息增益对应的特征索引值为: 0
'''

'''
https://blog.csdn.net/qq_30638831/article/details/79938967
figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
num:图像编号或名称，数字为编号 ，字符串为名称
figsize:指定figure的宽和高，单位为英寸；
dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80      1英寸等于2.5cm,A4纸是 21*30cm的纸张 
facecolor:背景颜色
edgecolor:边框颜色
frameon:是否显示边框
'''
def createPlot():
    fig = plt.figure(1, facecolor='white')
    # 清空整个绘图区
    fig.clf()
    # 创建单个子图
    ax1 = plt.subplot(111, frameon=False)
    # xy: 坐标，即箭头起始位置
    # xytext: 文本位置，即'a decision note'文本显示的位置
    # xycoords: xy的坐标系 'axes fraction' | 0,0 是轴域左下角，1,1 是右上角
    # textcoords: xytext的坐标系 'axes fraction' | 0,0 是轴域左下角，1,1 是右上角
    # arrowprops: 提供箭头属性字典来绘制从文本到注释点的箭头
    # ha="center"  在水平方向上，方框的中心在为（-2，0）
    # va="center"  在垂直方向上，方框的中心在为（0，-2）
    # bbox 代表对方框的设置 boxstyle方框的类型 fc方框的颜色
    ax1.annotate('a decision node', xy=(0.1, 0.5), xycoords='axes fraction', xytext=(0.5, 0.1), 
                 textcoords='axes fraction', va='center', ha='center', 
                 bbox=dict(boxstyle='sawtooth', fc='r'), arrowprops=dict(arrowstyle='<-'))
    ax1.annotate('a leaf node', xy=(0.3, 0.8), xycoords='axes fraction', xytext=(0.8, 0.1), 
                 textcoords='axes fraction', va='center', ha='center', 
                 bbox=dict(boxstyle='round4', fc='0.8'), arrowprops=dict(arrowstyle='<-'))    
    plt.show()
#createPlot()

'''
1)创建树信息；
'''
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

'''
1)获取叶节点的数目和数的层数；
'''
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth
    
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

decisionNode = dict(boxstyle="sawtooth", fc="b")
leafNode = dict(boxstyle="round4", fc="r")
arrow_args = dict(arrowstyle="<-")
def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

myTree = retrieveTree(1)
print(myTree)
print('get number leaf from tree: %d' %(getNumLeafs(myTree)))
print('get tree depth from tree: %d' %(getTreeDepth(myTree)))
#createPlot(myTree)    

'''
1)使用决策树分类函数；
'''    
def classify(inputTree, featureLabels, testVec):
    # 获取树的第一个节点值'no surfacing'
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 获取'no surfacing'对应的索引值
    featureIndex = featureLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featureIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featureLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
print('=====labels=====')
print(labels)
myTree = retrieveTree(0)
print(myTree)
print('第1个索引标签flippers特征属性值为0的分类为: %s' %(classify(myTree, labels, [1, 0])))
print('第1个索引标签flippers特征属性值为1的分类为: %s' %(classify(myTree, labels, [1, 1])))    
    
'''
1)使用pickle模块存储决策树；
2)使用pickle读取决策树；
'''
def storeTree(inputTree, fileName):
    import pickle
    with open(fileName, 'wb') as fw:
        pickle.dump(inputTree, fw)
    
def grabTree(fileName):
    import pickle
    fr = open(fileName, 'rb')
    return pickle.load(fr)
storeTree(myTree, 'classifierStorage.txt')
readTree = grabTree('classifierStorage.txt')
print(readTree)
    
'''
1)读取文本内容创建树的函数；
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #if len(dataSet[0]) == 1:
    #    return majorityCnt(classList)
    bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeatureIndex]
    print('---最好的特征索引值为: %d, 对应的特征标签名称为: %s' %(bestFeatureIndex, bestFeatureLabel))
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeatureIndex])
    featureValues = [example[bestFeatureIndex] for example in dataSet]
    uniqueValues = set(featureValues)
    for value in uniqueValues:
        subLabels = labels[:]
        print('==================================================')
        print(dataSet)
        subDataSet = splitDataSet(dataSet, bestFeatureIndex, value)
        subTree = createTree(subDataSet, subLabels)
        print(subTree)
        myTree[bestFeatureLabel][value] = subTree
    return myTree

dataSet, labels = createDataSet()
print(dataSet)
print(labels)
myTree = createTree(dataSet, labels)
print(myTree)

'''
1)使用决策树预测隐形眼镜类型；
'''
fr = open('lenses.txt')
lenses = [line.strip().split('\t') for line in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
print('                       ')
print(lensesTree)
createPlot(lensesTree)