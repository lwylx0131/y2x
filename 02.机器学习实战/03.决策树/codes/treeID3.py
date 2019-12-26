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
        

