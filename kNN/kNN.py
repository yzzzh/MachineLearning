# -*- coding: utf8 -*-
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

#k-邻近分类器
def classify0(inX,dataSet,labels,k):
    #dataSet是一个矩阵，shape[0]取第一维长度，也就是行数
    dataSetSize = dataSet.shape[0]
    #tile(A,(n,m)),将A第一维重复n次,第二维重复m次,返回一个矩阵
    #矩阵减法
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    #矩阵乘方
    sqDiffMat = diffMat**2
    #axis=1,对矩阵的每一行相加返回一个数组
    sqDistances = sqDiffMat.sum(axis=1)
    #开方
    distances = sqDistances**0.5
    #根据大小对索引排序
    sortedDistIndicies = distances.argsort()
    classCount = {}
    #统计
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回计数最多的标签
    return sortedClassCount[0][0]

#归一化函数
#newVal = (oldVal-min)/(max-min)
def autoNorm(dataSet):
    #0的作用是获取每列最小值，返回一个1*3的矩阵
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    size = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(size,1))
    normDataSet = normDataSet / tile(ranges,(size,1))
    return normDataSet,ranges,minVals

#数据可视化
def show(datingDataMat,datingLabels):
    #获取图表
    fig = plt.figure()
    #获取样式
    #111:将画布分成x行y列，并显示在第z格中，从左到右，从上到下
    ax = fig.add_subplot(111)
    #将第一第二列显示出来，第三个参数是大小，第四个是颜色，可以是一个数字，也可以是数组
    ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()

#练习1
#获取训练集，得到分类模型
def file2matrix(filename):
    f = open(filename)
    lines = f.readlines();
    numberOfLines = len(lines)
    #返回相应行列数的矩阵，并初始化为0
    matrix = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        matrix[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return matrix,classLabelVector

#练习1测试代码
def datingClassTest():
    ratio = 0.1
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    size = normMat.shape[0]
    numTest = int(size*ratio)
    errorCount = 0.0
    for i in range(numTest):
        res = classify0(normMat[i,:],normMat[numTest:size,:],datingLabels[numTest:size],3)
        if(res != datingLabels[i]):
            errorCount+=1.0
    print("error rate: %f"%(errorCount/numTest))

#练习2
#根据图片获取测试项
def img2vector(filename):
    res = zeros(1024)
    f = open(filename)
    for i in range(32):
        line = f.readline()
        for j in range(32):
            res[32*i+j] = int(line[j])
    return res

#构建分类模型
def createHandWritingDataSet(file):
    labels = []
    fileList = listdir(file)
    size = len(fileList)
    resMat = zeros((size,1024))
    for i in range(size):
        filename = fileList[i]
        label = int((filename.split('.')[0]).split('_')[0])
        labels.append(label)
        resMat[i,:] = img2vector(file+'\\'+filename)
    return resMat,labels

def handwringClassTest():
    #不知道为什么相对路径不行
    trainingMat,trainingLabels = createHandWritingDataSet(r'D:\Python_Project\MachineLearning\kNN\trainingDigits')
    testMat,testLabels = createHandWritingDataSet(r'D:\Python_Project\MachineLearning\kNN\testDigits')
    size = len(testLabels)
    errorCount = 0.0
    for i in range(size):
        res = classify0(testMat[i,:],trainingMat,trainingLabels,3)
        if(res != testLabels[i]):
            errorCount+=1.0
    print('error rate: %f'%(errorCount/size))

if __name__ == '__main__':
    #datingClassTest()
    handwringClassTest()