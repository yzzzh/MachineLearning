# -*- coding: utf8 -*-
from numpy import *
import matplotlib.pyplot as plt
import urllib
import json
from time import sleep

def loadData(filename):
    trainData = []
    for line in open(filename).readlines():
        line = line.strip().split()
        trainData.append([float(each) for each in line])
    return trainData

#欧拉距离
def olaDistance(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

#随机产生k个质心
def randCluster(dataSet,k):
    n = shape(dataSet)[1]
    clusters = mat(zeros((k,n)))
    for i in range(n):
        minVal = min(dataSet[:,i])
        rangeVal = float(max(dataSet[:,i]) - minVal)
        #生成k行1列的随机矩阵（0~1)
        clusters[:,i] = minVal + rangeVal * random.rand(k,1)
    return clusters

#k均值算法
#由于质心是随机生成的，会收敛到局部最优而非全局最优
def kMeans(dataSet,k,getDistance=olaDistance,createClusters=randCluster):
    m = shape(dataSet)[0]
    #记录每个点所对应的质心和相应的距离
    clustersDistances = mat(zeros((m,2)))
    clusters = createClusters(dataSet,k)
    clustersChanged = True
    while clustersChanged:
        clustersChanged = False
        #找到离每个点最近的质心,并更新
        for i in range(m):
            minDistance = inf
            minIndex = -1
            for j in range(k):
                distance = getDistance(dataSet[i,:],clusters[j,:])
                if distance < minDistance:
                    minDistance = distance
                    minIndex = j
            if clustersDistances[i,0] != minIndex:
                clustersChanged = True
            clustersDistances[i,:] = minIndex,minDistance
        #根据数据点更新质心位置
        for cluster in range(k):
            #获取属于这个簇的所有点
            pointsInCluster = dataSet[nonzero(clustersDistances[:,0].A == cluster)[0]]
            #求平均值并更新
            clusters[cluster,:] = mean(pointsInCluster,axis=0)
    return clusters,clustersDistances

#二分k均值算法
#一开始只有一个簇，选择误差减少最多的簇分割，直到分割成k个
def biKMeans(dataSet,k,getDistance=olaDistance):
    m = shape(dataSet)[0]
    clustersDistances = mat(zeros((m,2)))
    resClusters = []
    resClusters.append(mean(dataSet,axis=0).tolist()[0])
    for j in range(m):
        clustersDistances[j,1] = getDistance(mat(resClusters[0]),dataSet[j,:])**2
    while(len(resClusters) < k):
        minError = inf
        #找到最适合分割的簇
        for i in range(len(resClusters)):
            dataInThisCluster = dataSet[nonzero(clustersDistances[:,0].A == i)[0],:]
            #没有点在这个簇
            if len(dataInThisCluster) == 0:
                continue
            subClusters,subClustersDistances = kMeans(dataInThisCluster,2,getDistance)
            errorSplit = sum(subClustersDistances[:,1])
            errorNoSplit = sum(clustersDistances[nonzero(clustersDistances[:,0].A != i)[0],1])
            if (errorNoSplit + errorSplit) < minError:
                minError = errorNoSplit + errorSplit
                bestSubClusters = subClusters
                bestSubClustersDisntance = subClustersDistances.copy()
                bestClustersToSplit = i
        #更新簇和距离
        bestSubClustersDisntance[nonzero(bestSubClustersDisntance[:,0].A == 0)[0],0] = bestClustersToSplit
        bestSubClustersDisntance[nonzero(bestSubClustersDisntance[:,0].A == 1)[0],0] = len(resClusters)
        resClusters[bestClustersToSplit] = bestSubClusters[0,:].tolist()[0]
        resClusters.append(bestSubClusters[1,:].tolist()[0])
        clustersDistances[nonzero(clustersDistances[:,0].A == bestClustersToSplit)[0],:] = bestSubClustersDisntance
    return mat(resClusters),clustersDistances

def show(dataSet,clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:,0].A,dataSet[:,1].A,c='green')
    ax.scatter(clusters[:,0].A,clusters[:,1].A,c='red',marker='+',s=500)
    plt.show()

def getGeoInfo(city,street):
    #这个网址已经不能用了
    apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (city,street)
    urlParams = urllib.parse.urlencode(params)
    url = apiStem + urlParams
    return json.loads(urllib.request.urlopen(url).read())

def writeLatLng(filename):
    fw = open('places.txt','w')
    for line in open(filename).readlines():
        l = line.strip().split('\t')
        info = getGeoInfo(l[1],l[2])
        if info['ResultSet']['Error'] == 0:
            lat = float(info['ResultSet']['Results'][0]['latitude'])
            lng = float(info['ResultSet']['Results'][0]['longitude'])
            fw.write('%s\t%f\t%f\n'%(line,lat,lng))
        sleep(1)
    fw.close()

#计算两个经纬度之间的距离
def latlngDistance(vecA,vecB):
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0

def testMap():
    dataSet = []
    for line in open('places.txt').readlines():
        line = line.strip().split('\t')
        dataSet.append([float(line[4]), float(line[3])])
    dataSet = mat(dataSet)
    clusters, clustersDistances = biKMeans(dataSet, 4, latlngDistance)
    show(dataSet, clusters)

def test():
    dataSet = []
    for line in open('testSet2.txt').readlines():
        line = line.strip().split('\t')
        dataSet.append([float(line[0]),float(line[1])])
    dataSet = mat(dataSet)
    clusters, clustersDistances = biKMeans(dataSet, 3)
    show(dataSet, clusters)

if __name__ == '__main__':
    test()