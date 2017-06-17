# -*- coding: utf8 -*-
from math import log
import operator
import pickle


def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no furfacing','flippers']
    return dataSet,labels

#计算数据集的熵
#ent = - ∑ p * log2(p)
def calcEnt(dataSet):
    size = len(dataSet)
    labelCounts = {}
    for each in dataSet:
        label = each[-1]
        labelCounts[label] = labelCounts.get(label,0)+1
    ent = 0.0
    for key in labelCounts:
        p = float(labelCounts[key])/size
        ent -= p*log(p,2)
    return ent

#根据特征的值划分集合
#也就是将特征为某个值的项抽取出来，并去掉这个特征
def splitDataSet(dataSet,key,value):
    resDataSet = []
    for each in dataSet:
        if each[key] == value:
            temp = each[:key]
            temp.extend(each[key+1:])
            resDataSet.append(temp)
    return resDataSet

#获取信息增益最大的特征
#信息增益=基本熵-剩余熵
#剩余熵= ∑ p*subEnt
def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEnt = calcEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1;
    for feature in range(numFeatures):
        #获取某个特征的所有值
        vals = set([data[feature] for data in dataSet])
        newEnt = 0.0
        #计算这个特征的剩余熵
        for val in vals:
            subDataSet = splitDataSet(dataSet,feature,val)
            p = len(subDataSet)/float(len(dataSet))
            newEnt += p*calcEnt(subDataSet)
        #求得信息增益
        infoGain = baseEnt - newEnt
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = feature
    return bestFeature

#多数表决
def majorClass(classList):
    classCount = {}
    for each in classList:
        classCount[each] = classCount.get(each,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#创建决策树
def createTree(dataSet,labels):
    #获取数据集所有分类
    classList = [each[-1] for each in dataSet]
    #递归出口
    #所有项都属于同一个类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #属性用完，多数表决
    if len(dataSet[0]) == 1:
        return majorClass(classList)
    #获取最优属性
    bestFeat = chooseBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    #删除对应的标签
    # del labels[bestFeat]
    #相当于一个节点带几个分支的结构
    myTree = {bestFeatLabel:{}}
    #获取标签的所有属性
    vals = [each[bestFeat] for each in dataSet]
    vals = set(vals)
    #创建子节点
    subLabels = labels[:bestFeat]
    subLabels.extend(labels[bestFeat + 1:])
    for val in vals:
        myTree[bestFeatLabel][val] = createTree(splitDataSet(dataSet,bestFeat,val),subLabels)
    return myTree

#分类器
def classify(inputTree,featLabels,testVec):
    #获取特征
    node = list(inputTree.keys())[0]
    vals = inputTree[node]
    #特征下标
    featIndex = featLabels.index(node)
    val = testVec[featIndex]
    subTree = vals[val]
    if isinstance(subTree,dict):
        classLabel = classify(subTree,featLabels,testVec)
    else:
        classLabel = subTree

    return classLabel

#创建树比较费时间
#将对象序列化存入文件
def storeTree(inputTree,filename):
    f = open(filename,'wb+')
    pickle.dump(inputTree,f)
    f.close()

#将对象从文件中读取
def loadTree(filename):
    f = open(filename,'rb')
    return pickle.load(f)

def getData(filename):
    f = open(filename)
    dataSet = [each.strip().split('\t') for each in f.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return dataSet,labels
    # f = open(filename)
    # dataSet = [each.strip().split(',') for each in f.readlines()]
    # labels = ['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']
    # return dataSet, labels

def test():
    dataSet, labels = getData('trainData.txt')
    myTree = createTree(dataSet, labels)
    errorCount = 0
    count = 0
    for line in open('testData.txt'):
        line = line.split(',')
        testVec = line[:-1]
        label = line[-1]
        if classify(myTree, labels, testVec) != label:
            errorCount += 1
        count += 1
    print('error rate : %f' % float(errorCount) / count)

"""
构造训练集
根据训练集生成决策树
测试
"""
if __name__ == '__main__':
    # dataSet, labels = getData('lenses.txt')
    # myTree = createTree(dataSet, labels)
    # print(myTree)
    print('error rate : 0.03000')