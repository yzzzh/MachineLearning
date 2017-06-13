# -*- coding: utf8 -*-
from numpy import *

def loadData(filename):
    dataMat = []
    f = open(filename)
    for line in f.readlines():
        line = line.strip().split('\t')
        dataMat.append([float(each) for each in line])
    return dataMat

#根据特征和对应的值分割样本
def splitDataSet(dataSet,feature,value):
    # nonzero:
    # 传入一个矩阵，返回一个行矩阵一个列矩阵
    # 行矩阵说明非零元素在哪一行，列矩阵说明这个非零元素在哪一列
    # 若传入布尔型矩阵，True代表非0，False代表0
    dataSet1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    dataSet2 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    return dataSet1,dataSet2

#回归树
#样本不能再分割，以样本的平均值作为预测值
def typeRegress(dataSet):
    return mean(dataSet[:,-1])

#样本的总方差
def regressError(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

#选择最好的(方差最少)分割，返回特征和相应的值
#leafType有回归节点和模型节点，计算方法不一样，默认回归节点，errorType同理
def chooseBestSplit(dataSet, leafType=typeRegress, errorType=regressError, minError=1, minSampleNum=4):
    #所有样本的预测都一样
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    #未分割时的方差
    origError = errorType(dataSet)
    bestError = inf
    bestFeatureIndex = 0
    bestSplitValue = 0
    for featureIndex in range(n-1):
        vals = set(dataSet[:,featureIndex].T.tolist()[0])
        for val in vals:
            splitSet1,splitSet2 = splitDataSet(dataSet,featureIndex,val)
            #分的数量太少不要
            if (shape(splitSet1)[0] < minSampleNum) or (shape(splitSet2)[0] < minSampleNum):
                continue
            newError = errorType(splitSet1) + errorType(splitSet2)
            if newError < bestError:
                bestError = newError
                bestSplitValue = val
                bestFeatureIndex = featureIndex
    #分割前后相差不大
    if (origError - bestError) < minError:
        return None,leafType(dataSet)
    #数量太少
    set1, set2 = splitDataSet(dataSet, bestFeatureIndex, bestSplitValue)
    if (shape(set1)[0] < minSampleNum) or (shape(set2)[0] < minSampleNum):
        return None,leafType(dataSet)
    return bestFeatureIndex,bestSplitValue

def createTree(dataSet, leafType=typeRegress, errorType=regressError, minError=0, minSampleNum=1):
    feature,val = chooseBestSplit(dataSet,leafType,errorType,minError,minSampleNum)
    if feature is None:
        return val
    myTree = {}
    myTree['splitFeature'] = feature
    myTree['splitValue'] = val
    left,right = splitDataSet(dataSet,feature,val)
    myTree['left'] = createTree(left,leafType,errorType,minError,minSampleNum)
    myTree['right'] = createTree(right,leafType,errorType,minError,minSampleNum)
    return myTree

def isTree(obj):
    return isinstance(obj,dict)

#坍塌处理，将树的所有子树合并为一个预测节点
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

#剪枝函数
#原理：若左右节点合并后总方差比原来的方差小，那么就合并左右节点
#用测试集来剪枝，剪枝的效果取决于测试集
#若测试集为空，对树进行坍塌处理
#若左右子树有一个或都为树，根据树节点将测试集分割为两部分，对左右子树进行递归操作
#若左右子树都不为树，即都是预测值
#那么先计算不合并的情况，根据树节点对测试集进行分割，然后计算出左右子树与左右分割后的测试集的总方差
#合并的情况：计算左右节点的平均值，再计算出与测试集的总方差
#两者比较，若后者方差小，则合并，否则不合并
def prune(tree,testData):
    #说明并没有测试集进入这个判断节点，即过拟合，那么就可以对树剪枝
    if shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree['left']) or isTree(tree['right']):
        leftSet,rightSet = splitDataSet(testData,tree['splitFeature'],tree['splitValue'])
    #对左子树剪枝
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],leftSet)
    #对右子树剪枝
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rightSet)
    #递归终点，若均为预测节点，那么尝试合并，否则不合并直接返回
    if not isTree(tree['left']) and not isTree(tree['right']):
        leftSet,rightSet = splitDataSet(testData,tree['splitFeature'],tree['splitValue'])
        errorNoMerge = sum(power(leftSet[:,-1] - tree['left'],2)) + sum(power(rightSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right']) / 2.0
        errorMerge = sum(power((testData[:,-1] - treeMean),2))
        if errorMerge < errorNoMerge:
            return treeMean
        else:
            return tree
    else:
        return tree

#模型树
#叶子结点是一组权值
#不能用于剪枝
def getWeights(dataSet):
    m,n = shape(dataSet)
    xData = mat(ones((m,n)))
    #1~n列
    xData[:,1:n] = dataSet[:,0:n-1]
    yData = dataSet[:,-1]
    tempMat = xData.T * xData
    if linalg.det(tempMat) == 0:
        raise NameError('name error')
    w = tempMat.I * (xData.T * yData)
    return w,xData,yData

def typeModel(dataSet):
    w,x,y = getWeights(dataSet)
    return w

def modelError(dataSet):
    w,x,y = getWeights(dataSet)
    yExpected = x * w
    return sum(power(yExpected-y,2))

#回归树预测
def regressTreeEval(leafNode,testData):
    return float(leafNode)

#模型树预测
def modelTreeEval(leafNode,testData):
    n = shape(testData)[1]
    xData = mat(ones((1,n+1)))
    xData[:,1:n+1] = testData
    return float(sum(xData * leafNode))

#分类器
def treeClassify(tree,testData,treeEval=regressTreeEval):
    if not isTree(tree):
        return treeEval(tree,testData)
    if testData[tree['splitFeature']] < tree['splitValue']:
        return treeClassify(tree['left'],testData,treeEval)
    else:
        return treeClassify(tree['right'],testData,treeEval)

def testTree(tree,testData,treeEval=regressTreeEval):
    m = shape(testData)[0]
    yExpected = mat(zeros((m,1)))
    for i in range(m):
        yExpected[i,0] = treeClassify(tree,mat(testData)[i,:],treeEval)
    return yExpected

if __name__ == '__main__':
    trainMat = mat(loadData('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadData('bikeSpeedVsIq_test.txt'))

    #回归树
    regressTree = createTree(trainMat, typeRegress, regressError, 1, 20)
    regressTree = prune(regressTree,testMat)
    yExpected = testTree(regressTree,testMat[:,0],regressTreeEval)
    print(corrcoef(yExpected,testMat[:,1],rowvar=0)[0,1])
    #模型树
    modelTree = createTree(trainMat, typeModel, modelError, 1, 20)
    yExpected = testTree(modelTree,testMat[:,0],modelTreeEval)
    print(corrcoef(yExpected,testMat[:,1],rowvar=0)[0,1])
    #线性回归
    weights,x,y = getWeights(trainMat)
    for i in range(shape(testMat)[0]):
        yExpected[i,0] = weights[0,0]+weights[1,0] * testMat[i,0]
    print(corrcoef(yExpected,testMat[:,1],rowvar=0)[0,1])
    #结论：树回归拟合效果更好