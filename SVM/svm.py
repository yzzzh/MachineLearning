# -*- coding: utf8 -*-
from numpy import *

#计算核矩阵
#将一个在低维空间的非线性问题转化为高维空间下的线性问题
def kernelTrans(X,vec,kType):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    #线性核
    if kType[0] == 'linear':
        K = X * vec.T
    #高斯核
    elif kType[0] == 'goss':
        for j in range(m):
            deltaRow = X[j,:] - vec
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kType[1] ** 2))
    else:
        raise NameError('no such k type')
    return K

#把所有所需要的参数集成在一个数据结构，方便操作
class optStruct:
    def __init__(self,dataMat,labels,C,toler,kType):
        self.X = dataMat
        self.labelMat = labels
        self.C = C
        self.toler = toler
        self.m = shape(dataMat)[0]
        #w可以用alpha和x,y表示,w = ∑ai*xi*yi
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        #误差缓存
        #第一个值存这个缓存是否有效，第二个值存真实的Ek值
        self.eCache = mat(zeros((self.m,2)))
        #核矩阵，通过使用核矩阵来使用核函数,减少冗余计算
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
          self.K[:,i] = kernelTrans(self.X,self.X[i,:],kType)

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    for line in open(filename).readlines():
        line = line.strip().split('\t')
        dataMat.append([float(line[0]),float(line[1])])
        labelMat.append(float(line[2]))
    return mat(dataMat),mat(labelMat).transpose()

#在0-m-1中随机选取除i外的下标
def selectJrand(i,m):
    j = i
    while j == i:
        j = int(random.uniform(0,m))
    return j

#调整alpha的值在L和H之间
def clipAlpha(a,L,H):
    if a > H:
        a = H
    if a < L:
        a = L
    return a

#计算某个样本的Ek值
def calcEk(opt,k):
    fXk = float(multiply(opt.alphas,opt.labelMat).T * opt.K[:,k] + opt.b)
    Ek = fXk - float(opt.labelMat[k])
    return Ek

#选取|Ei - Ej|最大的j
def selectJ(i,opt,Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    opt.eCache[i] = [1,Ei]
    #有效的Ek
    validEcacheList = nonzero(opt.eCache[:,0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i :continue
            Ek = calcEk(opt,k)
            deltaE = abs(Ei-Ek)
            if deltaE > maxDeltaE:
                maxDeltaE = deltaE
                maxK = k
                Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i,opt.m)
        Ej = calcEk(opt,j)
        return j,Ej

#更新Ek,用于alpha变化之后
def updateEk(opt,k):
    Ek = calcEk(opt,k)
    opt.eCache[k] = [1,Ek]

#更新alpha对
def updateAlphaPairs(i,opt):
    Ei = calcEk(opt,i)
    #选出不符合KKT条件的i
    if (opt.labelMat[i]*Ei < -opt.toler and opt.alphas[i] < opt.C)\
        or\
        (opt.labelMat[i]*Ei > opt.toler and opt.alphas[i] > 0):
        #选出j
        j,Ej = selectJ(i,opt,Ei)
        #旧值
        alphaIold = opt.alphas[i].copy()
        alphaJold = opt.alphas[j].copy()
        #计算L和H
        if opt.labelMat[i] != opt.labelMat[j]:
            L = max(0,opt.alphas[j]-opt.alphas[i])
            H = min(opt.C,opt.C + opt.alphas[j] - opt.alphas[i])
        else:
            L = max(0,opt.alphas[j] + opt.alphas[i] - opt.C)
            H = min(opt.C,opt.alphas[j] + opt.alphas[i])
        if L == H :
            return 0
        #更新alphaj，alphai
        eta = 2.0 * opt.K[i,j] - opt.K[i,i] - opt.K[j,j]
        if eta >= 0 :
            return 0
        opt.alphas[j] -= opt.labelMat[j] * (Ei - Ej) / eta
        opt.alphas[j] = clipAlpha(opt.alphas[j],L,H)
        updateEk(opt,j)
        if abs(opt.alphas[j] - alphaJold) < 0.00001:
            return 0
        opt.alphas[i] += opt.labelMat[j] * opt.labelMat[i] * (alphaJold - opt.alphas[j])
        updateEk(opt,i)
        #更新b
        b1 = opt.b - Ei - opt.labelMat[i] * (opt.alphas[i] - alphaIold) *\
            opt.K[i,i] - opt.labelMat[j] *\
            (opt.alphas[j] - alphaJold) * opt.K[i,j]
        b2 = opt.b - Ej - opt.labelMat[i] * (opt.alphas[i] - alphaIold) *\
            opt.K[i,j] - opt.labelMat[j] *\
            (opt.alphas[j] - alphaJold)*opt.K[j,j]
        if 0 < opt.alphas[i] < opt.C:
            opt.b = b1
        elif 0 < opt.alphas[j] < opt.C:
            opt.b = b2
        else:
            opt.b = (b1+b2)/2.0
        return 1
    else:
        return 0

#主算法
def smo(opt,maxIter):
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    #结束条件是迭代次数到了，或者没有alpha对发生变化
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        #第一次先扫描整个集合
        #第二次之后只扫描0<α<c的部分，直到alpha没有变化，然后重新扫描整个集合
        alphaPairsChanged = 0
        if entireSet:
            for i in range(opt.m):
                alphaPairsChanged += updateAlphaPairs(i,opt)
            iter += 1
        else:
            nonBoundIs = nonzero((opt.alphas.A > 0) * (opt.alphas.A < opt.C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += updateAlphaPairs(i,opt)
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
    return opt.b,opt.alphas

def calcW(alphas,dataMat,labelMat):
    m,n = shape(dataMat)
    w = zeros((n,1))
    for i in range(m):
        if alphas[i] != 0:
            w += multiply(alphas[i]*labelMat[i],dataMat[i,:].T)
    return w

def test(trainSetFile, testSetFile,kType=['goss', 1.3]):
    dataMat,labelMat = loadDataSet(trainSetFile)
    opt = optStruct(dataMat,labelMat,200,0.0001,kType)
    #sigma参数改变了支持向量的个数，sigma越小，拟合程度越高，但保留的支持向量越多，趋近于knn
    #sigma越大，拟合程度低，但保留的支持向量少，计算更快
    b,alphas = smo(opt,10000)
    #由于非支持向量并不影响计算，因此只选出支持向量,从而减少计算量
    svIndex = nonzero(alphas.A > 0)[0]
    sv = dataMat[svIndex,:]
    svLabel = labelMat[svIndex,:]
    svAlphas = alphas[svIndex,:]

    #使用训练集测试
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        K = kernelTrans(sv,dataMat[i,:],kType)
        predict = K.T * multiply(svLabel,svAlphas) + b
        if sign(predict) != sign(labelMat[i]):
            errorCount += 1
    print('train set error rate: %f'%(float(errorCount)/m))

    #使用测试集测试
    dataMat,labelMat = loadDataSet(testSetFile)
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        K = kernelTrans(sv, dataMat[i, :], kType)
        predict = K.T * multiply(svLabel,svAlphas) + b
        if sign(predict) != sign(labelMat[i]):
            errorCount += 1
    print('test set error rate: %f' % (float(errorCount) / m))

def img2vector(filename):
    res = zeros(1024)
    f = open(filename)
    for i in range(32):
        line = f.readline()
        for j in range(32):
            res[32*i+j] = int(line[j])
    return res

def loadImages(dirname):
    from os import listdir
    trainingFileList = listdir(dirname)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    labelMat = []
    for i in range(m):
        filename = trainingFileList[i]
        label = int(filename.strip().split('.')[0].split('_')[0])
        if label == 9:
            labelMat.append(-1)
        else:
            labelMat.append(1)
        trainingMat[i,:] = img2vector('%s/%s'%(dirname,filename))
    return mat(trainingMat),mat(labelMat).transpose()

def testDigits(trainFile,testFile,kType=['goss',1.3]):
    dataMat,labelMat = loadImages(trainFile)
    opt = optStruct(dataMat,labelMat,200,0.0001,kType)
    b,alphas = smo(opt,1000)

    svIndex = nonzero(alphas.A > 0)[0]
    sv = dataMat[svIndex, :]
    print(shape(sv)[0])
    svLabel = labelMat[svIndex, :]
    svAlphas = alphas[svIndex, :]

    # 使用训练集测试
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        K = kernelTrans(sv, dataMat[i, :], kType)
        predict = K.T * multiply(svLabel, svAlphas) + b
        if sign(predict) != sign(labelMat[i]):
            errorCount += 1
    print('train set error rate: %f' % (float(errorCount) / m))

    # 使用测试集测试
    dataMat, labelMat = loadImages(testFile)
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        K = kernelTrans(sv, dataMat[i, :], kType)
        predict = K.T * multiply(svLabel, svAlphas) + b
        if sign(predict) != sign(labelMat[i]):
            errorCount += 1
    print('test set error rate: %f' % (float(errorCount) / m))

if __name__ == '__main__':
    #sigma和C的设置要看具体应用多次调试，并没有统一的估算方法
    testDigits('trainingDigits','testDigits',['goss',50])
