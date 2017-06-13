# -*- coding: utf8 -*-
from mrjob.job import MRJob
from mrjob.step import MRStep
from numpy import *
import pickle
#Pegasos算法的MapRecude版本,使用mrjob包实现
#将查找分类错误的样本下标的任务分配给mapper
#将根据分类错误的样本下标重新计算权值的任务分配给reducer
class MRsvm(MRJob):
    DEFAULT_INPUT_PROTOCOL = 'json_value'

    def __init__(self,*args,**kwargs):
        super(MRsvm,self).__init__(*args,**kwargs)
        #训练集,只有py2.7版本，其他打不开
        self.data = pickle.load(open('svmDat27','rb'))
        #权重
        self.weights = 0
        #学习率
        self.eta = 0.69
        #记录分类错误的样本的下标
        self.dataList = []
        #每一批次处理的样本数目
        self.batchSize = self.options.batchsize
        #mapper的数量
        self.numMappers = 1
        #迭代次数
        self.iterNum = 1

    #添加属性,默认的迭代次数和每批次样本数目
    def configure_options(self):
        super(MRsvm,self).configure_options()
        self.add_passthrough_option('--iterations',dest='iterations',default=2,type='int',
                                    help='iterNum: number of iterations to run')
        self.add_passthrough_option('--batchsize',dest='batchsize',default=100,type='int',
                                    help='batchsize: number of data points in a batch')

    #接收由reducer传回来的数据,即新的权重，迭代次数和随机选取的样本下标
    def map(self,mapperId,values):
        if False:yield
        if values[0] == 'weight':
            self.weights = values[1]
        elif values[0] == 'errorIndex':
            self.dataList.append(values[1])
        elif values[0] == 'iterNum':
            self.iterNum = values[1]

    #处理数据，这里只用于找到分类错误的下标,其他不做处理
    def map_final(self):
        labels = self.data[:,-1]
        x = self.data[:,:-1]
        if self.weights == 0:
            self.weights = zeros(shape(x)[1])
        for index in self.dataList:
            #判别公式，知道就好
            p = mat(self.weights) * x[index,:].T
            if labels[index] * p < 1.0:
                yield (1,['errorIndex',index])
            yield (1,['weight',self.weights])
            yield (1,['iterNum',self.iterNum])

    #根据分类错误的样本重新调整权重，并将新的权重和随机选取的样本下标重新发回给mapper
    def reduce(self,reducerId,values):
        for value in values:
            if value[0] == 'errorIndex':
                self.dataList.append(value[1])
            elif value[0] == 'weight':
                self.weights = value[1]
            elif value[0] == 'iterNum':
                self.iterNum = value[1]
        labels = self.data[:,-1]
        x = self.data[:,:-1]
        w = mat(self.weights)
        wDelta = mat(zeros(len(self.weights)))
        #重新调整权重，知道就好
        for index in self.dataList:
            wDelta += float(labels[index]) * x[index,:]
        eta = 1.0/(2.0 * self.iterNum)
        w = (1.0-1.0/self.iterNum)*w + (eta/self.batchSize)*wDelta
        for mapperNum in range(1,self.numMappers+1):
            #新的权重
            yield (mapperNum,['weight',w.tolist()[0]])
            if self.iterNum < self.options.iterations:
                #新的迭代次数
                yield [mapperNum,['iterNum',self.iterNum+1]]
                #随机选择样本发送给mapper
                for j in range(self.batchSize/self.numMappers):
                    yield (mapperNum,['errorIndex',random.randint(shape(self.data)[0])])

    #执行多次迭代
    def steps(self):
        return ([MRStep(mapper=self.map,mapper_final=self.map_final,reducer=self.reduce)]*self.options.iterations)

if __name__ == '__main__':
    MRsvm.run()
