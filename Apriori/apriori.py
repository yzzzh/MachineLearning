# -*- coding: utf8 -*-
from numpy import *
from time import sleep

def loadData():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

#返回{{1},{2},{3},{4},{5}}
def createC1(dataSet):
    c1 = []
    for item in dataSet:
        for each in item:
            if not [each] in c1:
                c1.append([each])
    c1.sort()
    #frozenset,即不可改变的集合
    return list(map(frozenset,c1))

#Ck根据支持度过滤成为Lk
def CkToLk(dataSet,Ck,minSupport):
    #统计ck每项出现的次数
    ckCount = {}
    for data in dataSet:
        for each in Ck:
            if each.issubset(data):
                ckCount[each] = ckCount.get(each,0) + 1
    numItems = float(len(dataSet))
    Lk = []
    supportData = {}
    #过滤支持度不足的,返回lk
    for each in ckCount:
        support = ckCount[each] / numItems
        if support >= minSupport:
            Lk.append(each)
        supportData[each] = support
    return Lk,supportData

#Lk生成CK+1
def LkToCk_p1(Lk,kp1):
    Ck_p1 = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            #若Lk[i]和Lk[j]前k-1项相同，则合并成一个k+1项的Ck[i]
            item1 = list(Lk[i])[:kp1-2]
            item2 = list(Lk[j])[:kp1-2]
            # item1.sort()
            # item2.sort()
            if item1 == item2:
                #已经排好序了
                Ck_p1.append(Lk[i]|Lk[j])
    return Ck_p1

#主算法，得到平凡项集和支持度计数
def apriori(dataSet,minSupport=0.5):
    C1 = createC1(dataSet)
    dataSet = list(map(set,dataSet))
    L1,supportData = CkToLk(dataSet,C1,minSupport)
    L = []
    L.append(L1)
    k = 1
    #L最后一项的长度大于0
    while len(L[k-1]) > 0:
        Ck_p1 = LkToCk_p1(L[k-1],k+1)
        Lk_p1,support_p1 = CkToLk(dataSet,Ck_p1,minSupport)
        supportData.update(support_p1)
        L.append(Lk_p1)
        k += 1
    return L,supportData

#生成关联规则
#具体原理：
#对于频繁项集[1,2,3,4]
#先生成123->4,124->3,134->2,234->1
#若123->4,134->2,234->1符合要求
#进一步将124组合成12,14,24
#计算34->12,23->14,13->24
#若34->12,23->14符合要求
#将12,14组合成124
#计算3->124
#124不能再组合，算法结束
def generateRules(L,supportData,minConf=0.7):
    rules = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            #对每一个频繁项集，如[1,2,3],则生成[[1],[2],[3]]
            right = [frozenset([item]) for item in freqSet]
            if i == 1:
                calcConf(freqSet,right,supportData,rules,minConf)
            else:
                getRulesFromFreqSet(freqSet,right,supportData,rules,minConf)
    return rules

#right是关联规则的右边,即A->B中的B，计算出每个B的支持度，并筛选出符合最小支持度的B，更新关联规则并返回这些B
def calcConf(freqSet,right,supportData,rules,minConf=0.7):
    resRights = []
    for each in right:
        conf = supportData[freqSet] / supportData[freqSet - each]
        if conf >= minConf:
            print('%r --> %r,conf: %r'%(freqSet-each,each,conf))
            rules.append((freqSet - each,each,conf))
            resRights.append(each)
    return resRights

#先计算B长度为1的情况，计算完获取返回的符合要求的B，若B长度大于1，说明可以继续组合，将这些B组合成长度为2的B，继续递归
def getRulesFromFreqSet(freqSet,right,supportData,rules,minConf=0.7):
    m = len(right[0])
    if len(freqSet) > (m+1):
        # 这里没有算B长度为1的情况，不知道为什么
        ck_p1 = LkToCk_p1(right,m+1)
        lk_p1 = calcConf(freqSet,ck_p1,supportData,rules,minConf)
        if len(lk_p1) > 1:
            getRulesFromFreqSet(freqSet,lk_p1,supportData,rules,minConf)

# def getActionIds():
#     actionIdList = []
#     billTitleList = []
#     for line in open('recent20bills.txt').readlines():
#         billId = int(line.split('\t')[0])
#         try:
#             billDetail = votesmart.votes.getBill(billId)
#             for action in billDetail.actions:
#                 if action.level == ' House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
#                     actionId = int(action.actionId)
#                     print('actionID:%d billtitle:%d'%(actionId,billId))
#                     actionIdList.append(actionId)
#                     billTitleList.append(line.strip().split('\t')[1])
#         except:
#             pass
#         sleep(0.1)
#     return actionIdList,billTitleList

def testMushroom():
    mushDataSet = [line.split() for line in open('mushroom.dat')]
    L,supportData = apriori(mushDataSet,0.3)
    rules = generateRules(L,supportData,0.9)
    print(L)
    print(rules)

if __name__ == '__main__':
    # dataSet = loadData()
    # L,supportData = apriori(dataSet,0.5)
    # rules = generateRules(L,supportData,0.5)
    # print(rules)
    testMushroom()