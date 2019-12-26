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

'''
1)计算给定数据的香农熵;
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
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

print(dataSet) # [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
# 划分出第二个特征并指定值为1的数据集
print(splitDataSet(dataSet, 1, 1)) # [[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']]


        