# -*- encoding: utf-8 -*-
import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1代表侮辱性文字，0代表正常言论
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    # 创建一个空集
    vocabSet = set([])
    for document in dataSet:
        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('单词: %s不在我的词汇表中！' %(word))
    return returnVec

postingList, classList = loadDataSet()
vocabList = createVocabList(postingList)
print(vocabList)
# vocabList是一张词汇表[a, b, c]，postingList[1](ade)表示一篇文章，如果出现一个词则使用1标示[1, 0, 0]
returnVec = setOfWords2Vec(vocabList, postingList[1])
print(returnVec)

'''
1)朴素贝叶斯分类器；
'''
def trainNB0(trainMatrix, trainCategory):
    # 所有文档构成的文档矩阵，获取文档数量，一行代表一篇
    numTrainDocs = len(trainMatrix)
    # 获取文档的单词数，注意这里是指词汇表的数量
    numWords = len(trainMatrix[0])
    # 因为侮辱性使用1表示，所以sum求和为侮辱性文章数量，再除以文章总数量，得到侮辱性文章的概率（先验概率）
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        # 在侮辱性文章中，获取每个单词出现的次数以及出现单词的总次数
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        # 在正常的文章中，获取每个单词出现的次数以及出现单词的总次数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 侮辱性文章的概率向量
    p1Vect = p1Num / p1Denom
    # 正常的文章的概率向量
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive

myVocabList = createVocabList(postingList)
trainMatrix = []
for doc in postingList:
    trainMatrix.append(setOfWords2Vec(myVocabList, doc))
print('======================train matrix=======================')
print(trainMatrix)
p0V, p1V, pAb = trainNB0(trainMatrix, classList)
print(p0V)
print(p1V)
print(pAb)
