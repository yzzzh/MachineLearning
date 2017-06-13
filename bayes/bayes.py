# -*- coding: utf8 -*-
from numpy import *
import re
import operator
import feedparser
#生成单词数据集和分类
def loadDataSet():
    wordList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return wordList,classVec

#输入数据集，返回一个不重复的词汇表，用于检测你想检测的单词
def createVocabList(dataSet):
    vocabSet = set([])
    for each in dataSet:
        vocabSet = vocabSet | set(each)
    return list(vocabSet)

#检测单词是否出现在词汇表中，返回检测向量，若出现，则在相应下标标1
def setOfwords2vec(vocabList, words):
    resVec = [0]*len(vocabList)
    for word in words:
        if word in vocabList:
            resVec[vocabList.index(word)] = 1
    return resVec

#考虑了出现的频率
def bagOfwords2vec(vocabList, words):
    resVec = [0]*len(vocabList)
    for word in words:
        if word in vocabList:
            resVec[vocabList.index(word)] += 1
    return resVec

#获取0和1条件下各事件的概率以及1发生的概率
#平滑：将所有值在初始化时都加1，分母加32
#防止下溢：取ln(a*b)而不是a*b
def trainNB(trainMat, trainClass):
    numTrain = len(trainMat)
    numWords = len(trainMat[0])
    p = sum(trainClass)/float(numTrain)
    p0 = ones(numWords)
    p1 = ones(numWords)
    p0Num = numWords
    p1Num = numWords
    for i in range(numTrain):
        if trainClass[i] == 1:
            p1 += trainMat[i]
            p1Num += sum(trainMat[i])
        else:
            p0 += trainMat[i]
            p0Num += sum(trainMat[i])
    p1Vect = log(p1/p1Num)
    p0Vect = log(p0/p0Num)
    return p0Vect,p1Vect,p

#获取条件概率表
def getProbTable():
    listOfWords,classes = loadDataSet()
    vocabList = createVocabList(listOfWords)
    listOfVec = []
    for each in listOfWords:
        listOfVec.append(setOfwords2vec(vocabList, each))
    p0Vec,p1Vec,p1 = trainNB(listOfVec, classes)
    return p0Vec,p1Vec,p1

#分类器
#p(ci|i1,i2...in) = p(i1|c1)*...p(in|ci)*p(ci)
def classifyNB(testVec,vec0,vec1,p1):
    res1 = sum(testVec*vec1) + log(p1)
    res0 = sum(testVec*vec0) + log(1.0-p1)
    if res1>res0:
        return 1
    else:
        return 0

#文本解析，返回文本中的所有单词
def textParse(txt):
    reg = re.compile('\\W*')
    words = reg.split(txt)
    return [word.lower() for word in words if len(word) > 2]

"""
测试：
构造词汇表
构造训练集
根据训练集得到条件概率表
测试
"""
def emailTest():
    emailList = []
    classList = []
    #构建词汇表
    for i in range(1,26):
        wordList = textParse(open('spam/%d.txt'%i).read())
        emailList.append(wordList)
        classList.append(1)
        wordList = textParse(open('ham/%d.txt'%i).read())
        emailList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(emailList)
    trainingSet = list(range(50))
    testSet = []
    #随机30个训练集，20个测试集，各不相同
    for i in range(20):
        index = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[index])
        del trainingSet[index]
    #构造训练集
    trainingMat = []
    trainingClass = []
    for index in trainingSet:
        trainingMat.append(bagOfwords2vec(vocabList,emailList[index]))
        trainingClass.append(classList[index])
    #获取条件概率表
    p0vec,p1vec,p1 = trainNB(trainingMat,trainingClass)
    errorCount = 0
    for index in testSet:
        testVec = bagOfwords2vec(vocabList,emailList[index])
        if classifyNB(testVec,p0vec,p1vec,p1) != classList[index]:
            errorCount += 1
    print('error rate : %f'%(errorCount/len(testSet)))

#计算出现频率最高的30个词
def calcMostFreq(vacabList,fullText):
    freqWords = {}
    for word in vacabList:
        freqWords[word] = fullText.count(word)
    res = sorted(freqWords.items(),key = operator.itemgetter(1),reverse=True)
    return res[:30]

#
def testRSS(feed1,feed0):
    docList = []
    classList = []
    fullText = []
    minlen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minlen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30words = calcMostFreq(vocabList,fullText)
    trainingSet = list(range(2*minlen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = []
    trainClass = []
    for i in trainingSet:
        trainMat.append(bagOfwords2vec(vocabList,docList[i]))
        trainClass.append(classList[i])
    vec0,vec1,p1 = trainNB(trainMat,trainClass)
    errorCount = 0.0
    for i in testSet:
        testVec = bagOfwords2vec(vocabList,docList[i])
        if classifyNB(testVec,vec0,vec1,p1) != classList[i]:
            errorCount+=1
    print('error rate: %f'%(errorCount/len(docList)))
    return vocabList,vec0,vec1

def getTopWords(feed0,feed1):
    vocabList,vec0,vec1 = testRSS(feed0,feed1)
    top0 = []
    top1 = []
    for i in range(len(vec0)):
        if vec0[i] > -5.0:
            top0.append((vocabList[i],vec0[i]))
        if vec1[i] > -5.0:
            top1.append((vocabList[i],vec1[i]))
    sort0 = sorted(top0,key=lambda pair:pair[1],reverse=True)
    sort1 = sorted(top1,key=lambda pair:pair[1],reverse=True)
    print('===sf===')
    for each in sort0:
        print(each[0])
    print('===ny===')
    for each in sort1:
        print(each[0])

if __name__ == '__main__':
    # listOfWords, classes = loadDataSet()
    # vocabList = createVocabList(listOfWords)
    # trainingVec = []
    # for each in listOfWords:
    #     trainingVec.append(setOfwords2vec(vocabList, each))
    # #条件概率表
    # p0Vec, p1Vec, p1 = trainNB(array(trainingVec), array(classes))
    # testVec = ['stupid','garbage']
    # testVec = array(setOfwords2vec(vocabList, testVec))
    # print(classifyNB(testVec,p0Vec,p1Vec,p1))
    # listOPosts, listClasses = loadDataSet()
    # emailTest()
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    fs = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    # testRSS(ny,fs)
    getTopWords(ny,fs)

"""
改进：1.计算条件概率时可以使用频率计算，而不是只要出现了就设定为1
2.处理多个维度时，可以创建一个table,key为类标号，value为一个字典，包含条件概率的向量和类标号的概率
"""