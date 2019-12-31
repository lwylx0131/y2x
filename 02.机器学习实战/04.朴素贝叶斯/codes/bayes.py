# -*- encoding: utf-8 -*-
import numpy as np
import math

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
# 以上是将词的出现与否作为一个特征，此称为词集模型（set-of-words model）
returnVec = setOfWords2Vec(vocabList, postingList[1])
print(returnVec)

'''
1)朴素贝叶斯分类器；
2)1代表侮辱性文字，0代表正常言论
classVec = [0, 1, 0, 1, 0, 1]
trainMatrix:
[[1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], 
[0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 
[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
'''
def trainNB0(trainMatrix, trainCategory):
    # 所有文档构成的文档矩阵，获取文档数量，一行代表一篇
    numTrainDocs = len(trainMatrix)
    # 获取文档的单词数，注意这里是指词汇表的数量
    numWords = len(trainMatrix[0])
    # 因为侮辱性使用1表示，所以sum求和为侮辱性文章数量，再除以文章总数量，得到侮辱性文章的概率（先验概率）
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
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
    p1Vect = np.log(p1Num / p1Denom)
    # 正常的文章的概率向量
    p0Vect = np.log(p0Num / p0Denom)
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

'''
计算朴素贝叶斯：
1)给出的某一篇文章词向量:[0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
2)词向量 * 侮辱性文章的概率向量 * 先验概率: 由于侮辱性文章的概率向量已经求过log，直接*词向量即可，再将*先验概率也转换为log: + log(侮辱性先验概率)
3)词向量 * 正常性文章的概率向量 * 先验概率: 由于正常性文章的概率向量已经求过log，直接*词向量即可，再将*先验概率也转换为log: + log(正常性先验概率)
4)比较2和3，看谁的概率大，那么此向量文章就属于哪类
'''
def classifyNB(wordsVec, p0Vec, p1Vec, pAb):
    p1 = np.sum(wordsVec * p1Vec) + np.log(pAb)
    p0 = np.sum(wordsVec * p0Vec) + np.log(1.0 - pAb)
    if(p1 > p0):
        return 1
    else:
        return 0

def testingNB():
    # 加载训练数据以及对应的分类标签
    postingList, classesList = loadDataSet()
    # 根据训练数据创建词表库
    myVocabList = createVocabList(postingList)
    # 以词表库为基础，将训练数据文章转化为矩阵数据集，出现的单词为1，否则为0
    trainMatrix = []
    for doc in postingList:
        trainMatrix.append(setOfWords2Vec(myVocabList, doc))
    # 已训练数据为基础计算出p0Vec（所有训练文章的正常性文章中的词向量概率）和p1Vec（所有训练文章的侮辱性文章中的词向量概率）以及pAb（侮辱性分类的先验概率）
    p0Vec, p1Vec, pAb = trainNB0(trainMatrix, classesList)
    # 测试文章
    testDocment = ['love', 'my', 'dalmation']
    # 将测试文章转换为词向量
    testDocVec = setOfWords2Vec(myVocabList, testDocment)
    print('测试文章被分类为: %s' %(classifyNB(testDocVec, p0Vec, p1Vec, pAb)))
    testDocment = ['stupid', 'garbage']
    testDocVec = setOfWords2Vec(myVocabList, testDocment)
    print('测试文章被分类为: %s' %(classifyNB(testDocVec, p0Vec, p1Vec, pAb)))

testingNB()

'''
  以上朴素贝叶斯分类算法中，是将每个词的出现与否作为一个特征，被称为词集模型（set-of-words model）；
  如果一个词在文档中出现不止一次，可能意味着包含该词是否出现在文档中所不能表达的某种信息，此种被称为词袋模型（bag-of-words model）；
  在词袋中，每个单词可以出现多次，而在词集中，每个词只能出现一次。为适应词袋模型
，需要对函数setOfWords2Vec()进行修改为bagOfWords2Vec().
'''
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):
    import re
    listOfTokens = re.split(r'[\W*]', bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]

def spamText():
    import random
    docList = []
    fullTest = []
    classList = []
    for i in range(1, 26):
        spamEmailText = open('email/spam/%d.txt' %(i)).read()
        # 解析邮件内容进行字符串分割
        wordList = textParse(spamEmailText)
        # 将解析的一篇邮件内容放到docList，docList包含所有邮件内容，一行代表一封邮件
        docList.append(wordList)
        # 将所有的邮件内容一篇篇追加到fullText数组中
        fullTest.extend(wordList)
        # 将解析的spam垃圾邮件标示分类为1
        classList.append(1)
        
        hamEmailText = open('email/ham/%d.txt' %(i)).read()
        wordList = textParse(hamEmailText)
        docList.append(wordList)
        fullTest.extend(wordList)
        # 将解析的ham邮件标示分类为0
        classList.append(0)
        
    # 通过docList创建自己的词汇表
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        # 随机构建训练集数据
        # random.uniform(a, b) 用于生成一个指定范围内的随机浮点数
        randIndex = int(random.uniform(0, len(trainingSet)))
        # randIndex为在0~50取的其中一个值，然后在trainingSet取值
        testSet.append(trainingSet[randIndex])
        # 从训练集数据中随机抽取10个测试集合，抽取之后需要从训练集中删除。这种随机选择数据的一部分作为训练集，
        # 而剩余部分作为测试集的过程称为留存交叉验证（hold-out cross validation)
        del(trainingSet[randIndex])
    trainMatrix = []
    trainClasses = []
    #print(docList)
    for docIndex in trainingSet:
        # 计算出一篇邮件的词向量，并放入到训练集矩阵中
        docVec = setOfWords2Vec(vocabList, docList[docIndex])
        trainMatrix.append(docVec)
        # 同时放入对应的分类标签类型
        trainClasses.append(classList[docIndex])
    # 从训练数据集中计算出p0Vec（正常邮件词向量概率）和p1Vec（垃圾邮件词向量概率）以及pSpam（垃圾邮件分类的先验概率）
    p0Vec, p1Vec, pSpam = trainNB0(trainMatrix, trainClasses)
    # 从测试数据集来测试分类邮件
    errorCount = 0
    for docIndex in testSet:
        testDocVec = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(testDocVec, p0Vec, p1Vec, pSpam) != classList[docIndex]:
            errorCount += 1
    print('从%d封测试邮件中分类错误率为: %.3f' %(len(testSet), errorCount / float(len(testSet))))

spamText()