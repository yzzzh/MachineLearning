# -*- coding: utf8 -*-
#SMO算法的另一种版本，容易写成MapReduce的形式

from numpy import *

def predict(w,x):
    return w * x.T

#pegasos算法原理：
#从集合中随机选取k个样本，判断样本是否被正确分类
#若否，根据被错误分类的样本重新计算权值(先计算wDelta,在计算w)
#迭代多次直至结束
def pegasos(dataSet, labels, lam, iterNum, k):
    m,n = shape(dataSet)
    w = zeros(n)
    dataIndex = list(range(m))
    for i in range(1,iterNum):
        wDelta = mat(zeros(n))
        #根据迭代次数调整学习率
        eta = 1.0 / (lam * i)
        random.shuffle(dataIndex)
        #随机选取k个样本
        for i in range(k):
            randIndex = dataIndex[i]
            p = predict(w,dataSet[randIndex,:])
            #分类错误，调整wDelta
            if labels[randIndex] * p < 1.0:
                wDelta += labels[randIndex] * dataSet[randIndex,:].A
        #根据wDelta调整w
        w = (1.0 - 1/i) * w + (eta / k) * wDelta
    return w