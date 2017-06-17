# -*- coding: utf8 -*-
from numpy import *

def loadData():
    return [[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

def loadData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

#三种相似度计算方法

#欧拉距离
def oulaSimilar(vecA,vecB):
    return 1.0/(1.0 + linalg.norm(vecA - vecB))

#皮尔逊相关系数
def pearsonSimilar(vecA,vecB):
    if len(vecA) < 3:
        return 1.0
    return 0.5 + 0.5*corrcoef(vecA,vecB,rowvar=0)[0][1]

#余弦相似度
def cosSimilar(vecA,vecB):
    num = float(vecA.T * vecB)
    denom = linalg.norm(vecA)*linalg.norm(vecB)
    return 0.5 + 0.5*(num/denom)

#对于某一用户，求B的评分，根据A（已评分）与B的相似度，乘A的评分，即可得到B的预测
#最后求出加权平均
def standardEstimate(dataMat,user,item,similarMethod):
    n = shape(dataMat)[1]
    totalScore = 0.0
    totalSimilar = 0.0
    for i in range(n):
        #用户对某一物品的评分
        userScore = dataMat[user,i]
        if userScore == 0:
            continue
        #物品A和B的重合部分
        overLap = nonzero(logical_and(dataMat[:,item].A > 0,dataMat[:,i].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            #对重合部分求相似度
            similarity = similarMethod(dataMat[overLap,item],dataMat[overLap,i])
        totalSimilar += similarity
        totalScore += similarity * userScore
    if totalSimilar == 0:
        return 0
    else:
        return totalScore / totalSimilar
#svd降维
#一个矩阵可以被分解成matrix = u * s * v.T, matrix 是m * n矩阵
#其中s是一个对角矩阵，我们称为奇异矩阵，对角线上的值称为奇异值，和pca的特征值差不多
#选择奇异值占比最多的前K个，生成新的对角矩阵newS
# 即可以将m * n 的矩阵降维到k * n 的矩阵,减少了噪声和冗余信息
# 公式:newMat = (matrix.T * u[:,k] * newS.I).T
#在这里是在低维空间中计算相似度，提高了计算效率
def svdEstimate(dataMat,user,item,similarMethod):
    n = shape(dataMat)[1]
    totalSimilar = 0.0
    totalScore = 0.0
    u,sigma,vt = linalg.svd(dataMat)
    sig4 = mat(eye(4)*sigma[:4])
    lowerItems = (dataMat.T * u[:,:4] * sig4.I).T
    for i in range(n):
        userScore = dataMat[user,i]
        if userScore == 0 or i == item:
            continue
        similarity = similarMethod(lowerItems[:,item],lowerItems[:,i])
        totalSimilar += similarity
        totalScore += similarity * userScore
    if totalSimilar == 0:
        return 0
    else:
        return totalScore/totalSimilar

#返回评分最高的N个物品
def recommend(dataMat,user,N=3,getSimilarity=cosSimilar,estimateMethod=standardEstimate):
    noscoreItems = nonzero(dataMat[user,:].A==0)[1]
    if len(noscoreItems) == 0:
        return None
    itemScores = []
    for item in noscoreItems:
        estimateScore = estimateMethod(dataMat,user,item,getSimilarity)
        itemScores.append((item,estimateScore))
    return sorted(itemScores,key=lambda k:k[1],reverse=True)[:N]

def printMat(dataMat,thresh=0.8):
    for i in range(32):
        for j in range(32):
            if float(dataMat[i,j])  < thresh:
                print(1,end='')
            else:
                print(0,end='')
        print()

#SVD第二个应用就是将矩阵压缩
#例如原来是m*n的矩阵
#分解成m*m,m*n,n*n三个矩阵
#降维后将变成m*k,k*k,k*n三个矩阵
#大小为m*k+k*k+k*n=(m+n+k)*k,比m*n少了不少
def imgCompress(numSVD=3,thresh=0.8):
    dataMat = []
    for line in open('0_5.txt').readlines():
        temp = []
        for i in range(len(line)-1):
            temp.append(int(line[i]))
        dataMat.append(temp)
    dataMat = mat(dataMat)
    print('original')
    printMat(dataMat,thresh)
    u,sigma,vt = linalg.svd(dataMat)
    newSigma = mat(eye(numSVD)*sigma[:numSVD])
    newDataMat = u[:,:numSVD] * newSigma * vt[:numSVD,:]
    print('compress')
    printMat(newDataMat,thresh)

if __name__ == '__main__':
    # dataMat = mat(loadData2())
    # print(recommend(dataMat,4,estimateMethod=svdEstimate,getSimilarity=cosSimilar))
    # imgCompress(3)
    print(corrcoef([1,0,1], [0,0,1], rowvar=0)[0][1])

