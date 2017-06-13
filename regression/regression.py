# -*- coding: utf8 -*-
from numpy import *
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

def loadDataSet(filename):
    f = open(filename)
    dataMat = []
    labelMat = []
    numFeat = len(f.readline().strip().split('\t'))-1
    for line in f.readlines():
        line = line.strip().split('\t')
        dataMat.append([float(each) for each in line[:numFeat]])
        labelMat.append(float(line[-1]))
    return dataMat,labelMat

#求最佳回归系数
#e = ∑(y-x*w)^2
#写成矩阵格式: (y-xw)*(y*wx)
#对w求导（不会求）,得到x(y-xw)
#令其为0(收敛)，得到w = (x.T * x)^(-1) * x.T * y

def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    tempMat = xMat.T * xMat
    #求行列式的值
    #没有逆矩阵，返回空
    if linalg.det(tempMat) == 0.0:
        return
    weights = tempMat.I * (xMat.T * yMat)
    #返回一个1*n的回归系数
    return weights

#局部加权线性回归

#增加一个W向量，用来计算预测点与周围点之间的权重，离得越近权重越大，对预测点的影响也就越大
#w = (x.T * W * x)^(-1) * x.T * W * y
#权重的计算 w(i) = exp(|xi-x|/(-2*k^2))
#由于每次计算都要遍历所有点，计算量较大
#每个样本都能得到属于自己的权重，即权重是局部的
#关于k：k越大，考虑到的点越多，k越小，只会考虑到周围的点
#k越大，欠拟合，k越小，过拟合
def lwlrRegres(testVec, xTrain, yTrain, k):
    xMat = mat(xTrain)
    yMat = mat(yTrain).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for i in range(m):
        diffMat = testVec - xMat[i,:]
        #1*n和n*1矩阵，得到1*1矩阵，结果是平方和
        #矩阵只能用[i,i]访问而不能用[i][i]
        weights[i,i] = exp(diffMat*diffMat.T/(-2.0*k**2))
    tempMat = xMat.T * (weights * xMat)
    if linalg.det(tempMat) == 0.0:
        return 0
    w = tempMat.I * (xMat.T * (weights * yMat))
    return testVec * w

def TestLwlr(testMat, xTrain, yTrain, k=0.5):
    m = shape(testMat)[0]
    yExpected = zeros(m)
    for i in range(m):
        yExpected[i] = lwlrRegres(testMat[i], xTrain, yTrain, k)
    return yExpected

def testNormalRegres():
    xTrain, yTrain = loadDataSet('ex0.txt')
    xTrain = mat(xTrain)
    yTrain = mat(yTrain)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xTrain[:, 1].flatten().A[0], yTrain.A[0])
    weights = standRegres(xTrain, yTrain)
    xCopy = xTrain.copy()
    xCopy.sort(0)
    yExpected = xCopy * weights
    ax.plot(xCopy[:, 1], yExpected)
    # 计算两两的相关系数
    # print(corrcoef(yExpected.T,yMat))
    plt.show()

#改变k，观察拟合程度
def testLocalRegres():
    xTrain, yTrain = loadDataSet('ex0.txt')
    xTrain = mat(xTrain)
    yTrain = mat(yTrain)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xTrain[:, 1].flatten().A[0], yTrain.A[0])
    xTest = xTrain.copy()
    xTest.sort(0)
    yExpected = TestLwlr(xTest, xTrain, yTrain, 0.03)
    ax.plot(xTest[:,1].flatten().A[0], yExpected,c='red')
    plt.show()

#预测和实际之间的误差
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum() / len(yArr)

#岭回归
#添加限制∑w^2<=λ,防止某些w较大
#原式变为w = (x.T * x + λI)^(-1) * x.T * y
#λ要根据实际数据取不同值
def ridgeRegres(xTrain, yTrain, lam = 0.5):
    tempMat = xTrain.T * xTrain
    tempMat = tempMat + eye(shape(xTrain)[1])*lam
    if linalg.det(tempMat) == 0.0:
        return
    return tempMat.I * (xTrain.T * yTrain)

#先把数据标准化
#x = (x - u)/δ^2
#y = y - u
def ridgeTest(xTrain,yTrain):
    xMat = mat(xTrain)
    yMat = mat(yTrain).T
    #平均值
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMean = mean(xMat,0)
    #方差
    xVar = var(xMat,0)
    #可以不用，这里是因为全为1，导致方差为0，使得(x-u)/δ^2计算错误
    xVar[0] = 1.0
    xMat = (xMat - xMean) / xVar
    numTest = 30
    w = zeros((numTest,shape(xMat)[1]))
    for i in range(numTest):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        w[i,:] = ws.T
    return w

def testRidgeRegres():
    xTrain,yTrain = loadDataSet('abalone.txt')
    weights = ridgeTest(xTrain,yTrain)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(weights)
    plt.show()

#标准化x = (x-u)/δ^2
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    #可以不用，这里是因为全为1，导致方差为0，使得(x-u)/δ^2计算错误
    inVar[0] = 1.0
    inMat = (inMat - inMeans)/inVar
    return inMat

#前向逐步线性回归
#将权重初始化为1,每次迭代单独对每个特征 + 和 - 一个步长，得到这时的偏差,选择偏差最小的情况，更新权重
#最终将会得到使误差最小的权值
def stageWise(xTrain,yTrain,eps=0.01,numIter=200):
    xMat = mat(xTrain)
    yMat = mat(yTrain).T
    yMean = mean(yMat,0)
    yMat = yMat-yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    resMat = zeros((numIter,n))
    ws = zeros((n,1))
    wsMax = zeros((n,1))
    for i in range(numIter):
        #无穷大
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                error = rssError(yMat.A,yTest.A)
                if error < lowestError:
                    lowestError = error
                    wsMax = wsTest
        ws = wsMax.copy()
        resMat[i,:] = ws.T
    #最后一个就是我们想要的结果
    return resMat

def testStageWise():
    xTrain, yTrain = loadDataSet('abalone.txt')
    weights = stageWise(xTrain,yTrain,eps=0.005,numIter=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(weights)
    #会发现最终权值将会收敛，结果和最小二乘法相似
    plt.show()

#从购物系统获取信息
#已经过时了不能用了
# def searchForData(xTrain,yTrain,num,year,numPiece,originalPrice):
#     sleep(1)
#     myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
#     searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, num)
#     searchRes = json.loads(urllib.request.urlopen(searchURL).read())
#     items = searchRes['items']
#     for i in range(len(items)):
#         try:
#             currentItem = items[i]
#             if currentItem['product']['condition'] == 'new':
#                 isNew = 1
#             else:
#                 isNew = 0
#             listOfInv = currentItem['product']['inventories']
#             #过滤得到大于原价格一半的价格
#             for each in listOfInv:
#                 sellingPrice = each['price']
#                 if sellingPrice > 0.5*originalPrice:
#                     xTrain.append([year,numPiece,isNew,originalPrice])
#                     yTrain.append(sellingPrice)
#         except:
#             pass
#
# def setTrainingData(xTrain,yTrain):
#     #不用管这些数据，知道就好
#     searchForData(xTrain, yTrain, 10030, 2002, 3096, 269.99)
#     searchForData(xTrain, yTrain, 10179, 2007, 5195, 499.99)
#     searchForData(xTrain, yTrain, 10181, 2007, 3428, 199.99)
#     searchForData(xTrain, yTrain, 10189, 2008, 5922, 299.99)
#     searchForData(xTrain, yTrain, 10196, 2009, 3263, 249.99)
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):

    # 打开并读取HTML文件
    fr = open(inFile);
    soup = BeautifulSoup(fr.read(),'html.parser')
    i=1

    # 根据HTML页面结构进行解析
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()

        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0

        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            pass
            # print("item #%d did not sell" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','') #strips out $
            priceStr = priceStr.replace(',','') #strips out ,
            if len(soldPrice)>1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)

            # 去掉不完整的套装价格
            if  sellingPrice > origPrc * 0.5:
                    # print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([1.0,yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)

# 依次读取六种乐高套装的数据，并生成数据矩阵
def setTrainingData(retX, retY):
    scrapePage(retX, retY, 'setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, 'setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, 'setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, 'setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, 'setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, 'setHtml/lego10196.html', 2009, 3263, 249.99)

def testLego():
    xTrain=[]
    yTrain=[]
    setTrainingData(xTrain,yTrain)
    # w_normal = standRegres(xTrain,yTrain)
    # w_ridge = ridgeRegres(xTrain,yTrain.T)
    # w_stage = stageWise(xTrain,yTrain,eps=0.001,numIter=10000)[-1]
    # print(w_normal)
    # print(w_ridge)
    # print(w_stage)
    crossValidation(xTrain,yTrain)

#交叉验证测试岭回归
#即获取多组回归参数，选取效果最好(平均误差最小)的一组
def crossValidation(xArr,yArr,iterNum=10):
    m = len(yArr)
    indexList = list(range(m))
    errorMat = zeros((iterNum,30))
    #迭代iterNum次
    for i in range(iterNum):
        xTrain = []
        yTrain = []
        xTest = []
        yTest = []
        #在数据集中随机选90%作为训练集，10%作为测试集
        random.shuffle(indexList)
        for j in range(m):
            if j < m*0.9:
                xTrain.append(xArr[indexList[j]])
                yTrain.append(yArr[indexList[j]])
            else:
                xTest.append(xArr[indexList[j]])
                yTest.append(yArr[indexList[j]])
        #一共30组系数
        wMat = ridgeTest(xTrain, yTrain)
        #分别对每组系数测试一次
        xTest = mat(xTest)
        xTrain = mat(xTrain)
        xMean = mean(xTrain, 0)
        xVar = var(xTrain, 0)
        xVar[0] = 1.0
        xTest = (xTest - xMean) / xVar
        for k in range(30):
            # 标准化后还原
            yExpected = xTest * mat(wMat[k,:]).T + mean(yTrain)
            #记录误差
            errorMat[i,k] = rssError(yExpected.T.A,array(yTest))
    #找到误差最小的一组权值
    meanError = mean(errorMat,0)
    minMean = float(min(meanError))
    bestWeights = wMat[nonzero(meanError==minMean)]
    xVar = var(mat(xArr),0)
    #第一个方差总为0，很麻烦
    xVar[0] = 1.0
    #数据标准化后还原
    bestWeights = bestWeights / xVar
    print(bestWeights)
    return bestWeights


if __name__ == '__main__':
    # testStageWise()
    testLego()