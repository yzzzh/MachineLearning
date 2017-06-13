# -*- coding: utf8 -*-
from numpy import *
import matplotlib.pyplot as plt

#获取数据集和类标号
def loadDataSet():
    dataMat = []
    labelMat = []
    f = open('testSet.txt')
    for line in f.readlines():
        vals = line.strip().split()
        dataMat.append([1.0,float(vals[0]),float(vals[1])])
        labelMat.append(int(vals[2]))
    return dataMat,labelMat

#预测函数
def sigmoid(x):
    return 1.0/(1.0+exp(-x))

#原理：z = sum(w*x)
#w是训练好的向量，x是预测向量，求他们点积z
#z>0.5则为1，z<0.5则为0

#训练w的方法：
#w = w + a * Δf
#a为步长，Δf为梯度
#梯度的求法:目标值-当前值
#梯度小于特定值时或进行一定的轮数后算法停止

#梯度上升算法
def gradAscent0(dataMat, classLabels):
    #转化为矩阵方便操作
    dataMat = mat(dataMat)
    classLabels = mat(classLabels).transpose()
    m,n = shape(dataMat)
    #步长
    alpha = 0.001
    weights = mat(ones((n,1)))
    for k in range(1000):
        #预测值z,得到一个m*1矩阵
        expected = sigmoid(dataMat*weights)
        #求得梯度,m*1矩阵
        gradient = classLabels - expected
        #转化为n*m矩阵和m*1矩阵相乘，得到n*1矩阵
        #相当于每一项都获得了一个梯度值，每一项每一个值都和这个梯度值相乘，
        #再对所有项的某个特征值求和乘以步数，得到每一个特征需要变化的值
        weights = weights + alpha * dataMat.transpose() * gradient
    return weights

#随机梯度上升
#每次只选取一个样本而不是全部
#拟合效果不是太好
def gradAscent1(dataMat,classLabels):
    dataMat = array(dataMat)
    classLabels = array(classLabels)
    m, n = shape(dataMat)
    alpha = 0.01
    weights = ones(n)
    for k in range(m):
        expected = sigmoid(sum(dataMat[k]*weights))
        gradient = classLabels[k] - expected
        weights = weights + alpha * gradient * dataMat[k]
    weights = mat(weights).transpose()
    return weights

#改进的随机梯度上升算法
#比改进前增加了
#1.迭代次数
#2.alpha需要重新计算
#3.随机选取样本而不是顺序选取
#2和3有利于减少波动和加快收敛速度
def gradAscent2(dataMat,classLabels,iterNum=200):
    dataMat = array(dataMat)
    classLabels = array(classLabels)
    m, n = shape(dataMat)
    weights = ones(n)
    for j in range(iterNum):
        dataIndex = list(range(m))
        for k in range(m):
            alpha = 4/(1.0+j+k) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            expected = sigmoid(sum(dataMat[randIndex]*weights))
            gradient = classLabels[randIndex] - expected
            weights = weights + alpha * gradient * dataMat[randIndex]
            del dataIndex[randIndex]
    weights = mat(weights).transpose()
    return weights

def show(weights):
    dataMat,labelMat = loadDataSet()
    dataMat = array(dataMat)
    #行数
    n = shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i][1])
            ycord1.append(dataMat[i][2])
        else:
            xcord2.append(dataMat[i][1])
            ycord2.append(dataMat[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #画点
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    #获取weights的第一列得到1*n的数组
    weights = weights.getA()
    #z = w0x0+w1x1+w2x2
    #取z=0,x1=1即可获得分界线,x2=-(w0+w1x1)/w2
    x = array(arange(-3.0,3.0,0.1))
    y = (-weights[0]-weights[1]*x)/weights[2]
    #画线
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

#分类器
def classify(testVec,weights):
    prob = sigmoid(sum(testVec*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

#
def test():
    fTrain = open('horseColicTraining.txt')
    fTest = open('horseColicTest.txt')
    trainingSet = []
    trainingClasses = []
    for line in fTrain.readlines():
        line = line.strip().split('\t')
        trainingSet.append([float(each) for each in line[:21]])
        trainingClasses.append(float(line[21]))
    trainWeights = gradAscent2(trainingSet,trainingClasses,1000)
    errorNum = 0
    testNum = 0
    for line in fTest.readlines():
        testNum += 1
        line = line.strip().split('\t')
        label = line[21]
        line = [float(each) for each in line[:21]]
        if int(classify(line,trainWeights)) != int(label):
            errorNum+=1
    print('error rate: %f'%(errorNum/float(testNum)))


if __name__ == '__main__':
    # dataSet,labelSet = loadDataSet()
    # weights = gradAscent2(dataSet, labelSet,500)
    # show(weights)
    test()