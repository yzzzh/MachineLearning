# -*- coding: utf8 -*-
import twitter
import re
from time import sleep

class TreeNode:
    def __init__(self,name,count,parentNode):
        self.name = name
        self.count = count
        self.parentNode = parentNode
        self.next = None
        self.children = {}

    def inc(self,count):
        self.count += count

    def show(self,index=1):
        print('    '*index,self.name,' : ',self.count)
        for child in self.children.values():
            child.show(index+1)

def loadData():
    data = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return data

def initDataSet(data):
    res = {}
    for each in data:
        res[frozenset(each)] = 1
    return res

#FP树的构建：
#第一遍扫描整个数据库，计算所有数据的出现次数
#第二遍根据最小支持度筛选
#对数据库每一样本的集合根据出现的次数排序
#将排好序的样本插入到树中
def createTree(dataSet,minSupport=1):
    headerTable = {}
    #统计每个字母出现的次数
    for each in dataSet:
        for item in each:
            headerTable[item] = headerTable.get(item,0) + dataSet[each]
    #过滤
    for k in headerTable.copy().keys():
        if headerTable[k] < minSupport:
            del headerTable[k]

    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None,None
    #改变表结构，加入一个指针
    for k in headerTable:
        headerTable[k] = [headerTable[k],None]
    root = TreeNode('root',1,None)
    #对训练集每一项进行排序,并插入树中
    for key,value in dataSet.items():
        #这里value应该默认为1了
        counter = {}
        #记录每个字母出现的次数
        for item in key:
            if item in freqItemSet:
                counter[item] = headerTable[item][0]
        if len(counter) > 0:
            orderedItems = [each[0] for each in sorted(counter.items(),key=lambda p:p[1],reverse=True)]
            updateTree(orderedItems,root,headerTable,value)

    return root,headerTable

def updateTree(items, root, headerTable, count):
    #有这个节点就更新，没有就加入
    if items[0] in root.children:
        root.children[items[0]].inc(count)
    else:
        root.children[items[0]] = TreeNode(items[0], count, root)
        #更新链表
        if headerTable[items[0]][1] is None:
            headerTable[items[0]][1] = root.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],root.children[items[0]])
    #继续迭代下一个节点
    if len(items) > 1:
        updateTree(items[1:],root.children[items[0]],headerTable,count)

#将节点插入到链表最后一项
def updateHeader(root,targetNode):
    while root.next != None:
        root = root.next
    root.next = targetNode

#找到节点到根节点的路径
def ascendTree(treeNode,prefixPath):
    if treeNode.parentNode != None:
        prefixPath.append(treeNode.name)
        ascendTree(treeNode.parentNode,prefixPath)

#找到某个条件下的所有条件模式基
def findPrefixPath(treeNode):
    conditionPatterns = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode,prefixPath)
        if len(prefixPath) > 1:
            conditionPatterns[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.next
    return conditionPatterns

#prefix相当于前置条件，一开始为空
#从表中取一个条件与前置条件合并作为新的条件，那么这个新的条件肯定是频繁项集
#找到这个新条件的所有条件模式基，并作为数据集生成一棵新的树，这棵树以新条件为前置条件，不断重复递归，直到生成的树为空为止
#原理就是树的前置条件合并树中任意一个节点都是频繁项集
#例：一开始前置条件为空，生成一棵树，也就是最初的树
#B在树里面，B并空成为新的条件B，B就是频繁项集，那么找出B的所有条件模式基，以这些条件模式基为数据集生成一棵树
#那么这棵新的树的前置条件就是B
#继续，在这棵树中找到E，那么新的前置条件就是BE，BE是频繁项集，找出E的所有条件模式基并生成一棵树
#不断递归，直到不能生成树为止
def generateFreqSet(headerTable,minSupport,prefix,freqItemList):
    #对所有条件按从小到大排序
    basePatterns = [each[0] for each in sorted(headerTable.items(),key=lambda p:p[0])]
    for pattern in basePatterns:
        #与前置条件合并得到新的频繁项集
        freqSet = prefix.copy()
        freqSet.add(pattern)
        freqItemList.append(freqSet)
        #找到条件模式基
        conditionPatterns = findPrefixPath(headerTable[pattern][1])
        #根据条件模式基生成新的树
        newTree,newHeaderTable = createTree(conditionPatterns,minSupport)
        #新的树不为空，继续递归
        if newTree != None:
            generateFreqSet(newHeaderTable,minSupport,freqSet,freqItemList)

#根据搜索内容获取推特推文
def getTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY,consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,access_token_secret=ACCESS_TOKEN_SECRET)
    res = []
    for i in range(1,15):
        print('searching : ',i)
        res.append(api.GetSearch(searchStr,per_page=100,page=i))
        sleep(1)
    return res

#将推文转化为字符串列表
def testParse(text):
    #替换，除去url
    url = re.sub('(http[s]?:[/][/]|www.)([a-z][A-Z][0-9]|[/.]|[~])*','',text)
    tokens = re.split(r'\W*',url)
    return [each.tolower() for each in tokens if len(each) > 2]

#获取推文的频繁项集
def getTweetsFreqSet(tweetSet,minSupport):
    textList = []
    for i in range(4):
        for j in range(100):
            textList.append(testParse(tweetSet[i][j].text))
    dataSet = initDataSet(textList)
    tree,headerTable = createTree(dataSet,minSupport)
    freqSet = []
    generateFreqSet(headerTable,minSupport,set([]),freqSet)
    return freqSet

def testTweet():
    tweetSet = getTweets('')
    tweetFreq = getTweetsFreqSet(tweetSet,100)
    print(tweetFreq)

def testNews():
    newsData = [line.split() for line in open('kosarak.dat').readlines()]
    newsSet = initDataSet(newsData)
    tree,headerTable = createTree(newsSet,100000)
    freqSet = []
    generateFreqSet(headerTable,100000,set([]),freqSet)
    print(freqSet)

if __name__ == '__main__':
    # dataSet = loadData()
    # root,headerTable = createTree(dataSet,3)
    # freqItemList = []
    # generateFreqSet(headerTable,3,set([]),freqItemList)
    # print(len(freqItemList))
    # getTweets('123')
    testNews()