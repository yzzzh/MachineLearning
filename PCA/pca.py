# -*- coding: utf8 -*-
#
# PCA算法
#
# 设有m条n维数据。
#
# 1）将原始数据按列组成n行m列矩阵X
#
# 2）将X的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值
#
# 3）求出协方差矩阵C=1/m * X * X.T
#
# 4）求出协方差矩阵的特征值及对应的特征向量,并将向量标准化（单位化）
#
# 5）将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P
#
# 6）Y=PX即为降维到k维后的数据(此时x为均值化后的值)
#
#参考：http://blog.codinglabs.org/articles/pca-tutorial.html

#个人理解：我们要找的结果是Y=PX(n维向量降维到K维向量),而1/m*Y*Y.T是一个协方差矩阵（对角线是方差，其他地方是协方差）
#我们希望方差最大，协方差为0，
#于是把它变成一个对角矩阵D: 1/m*Y*Y.T = 1/m*(PX*PX.T) = (P*1/m*X*X.T*P.T) = (P*C*P.T) = D
#由于1/m*X*X.T=C也是协方差矩阵,协方差矩阵有一个性质，就是能找到一个单位矩阵E=(e1,e2,...,en)
#使得E*C*E.T为对角矩阵，因此E就是我们要找的P,而P就是我们要找的基向量的集合，我们根据D中的特征值的下标选择效果最好的K个向量即可

from numpy import *
import matplotlib.pyplot as plt

def loadDataMat(filename,p='\t'):
    dataSet = [list(map(float,line.strip().split(p))) for line in open(filename).readlines()]
    return mat(dataSet)

def pca(dataMat,k=10):
    #平均值
    meanVals = mean(dataMat,axis=0)
    #零均值化
    newDataMat = dataMat - meanVals
    #协方差矩阵
    covMat = cov(newDataMat,rowvar=0)
    #求特征值和特征向量
    #这里似乎已经帮我们排好序了
    eigVals,eigVecs = linalg.eig(mat(covMat))
    #观察前k个特征值所占百分比，从而确定k的值
    # s = sum(eigVals)
    # temp = 0.0
    # j = 0
    # for i in eigVals:
    #     j+=1
    #     temp += i
    #     print(j)
    #     print(temp/s*100)
    #对特征值排序
    eigValIndex = sorted(argsort(eigVals),reverse=True)
    #选择方差最多的前k个特征
    eigValIndex = eigValIndex[:k]
    #选择对应的这k个特征向量,向量已经标准化了
    eigVecs = eigVecs[:,eigValIndex]
    #变换后的结果
    resDataMat = newDataMat * eigVecs
    #不知道是什么
    # reconMat = resDataMat * eigVecs.T + meanVals
    # return resDataMat,reconMat
    return resDataMat
    # dataMat = dataMat.T
    # meanVals = mean(dataMat,1)
    # newDataMat = dataMat - meanVals
    # covMat = cov(newDataMat,rowvar=1)
    # eigVals,eigVecs = linalg.eig(mat(covMat))
    # eigValIndex = sorted(argsort(eigVals),reverse=True)
    # eigValIndex = eigValIndex[:k]
    # eigVecs = eigVecs[eigValIndex,:]
    # resDataMat = eigVecs * newDataMat
    # print(resDataMat.T)
    # return resDataMat,1

def testPCA():
    dataMat = loadDataMat('secom.data',' ')
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):
        #求第i列非NaN的平均值
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i])
        #将平均值赋给NaN
        dataMat[nonzero(isnan(dataMat[:,i].A))[0],i] = meanVal
    resData = pca(dataMat,6)
    print(shape(resData))

if __name__ == '__main__':
    # dataMat = loadDataMat('testSet.txt')
    # resData = pca(dataMat, 1)
    # # print(resData.T)
    # # print(shape(resData))
    #k值的选取需要观察特征值大小，这里观察到前6个特征值占了总的97%的百分比，从而实现了560到6的降维
    testPCA()