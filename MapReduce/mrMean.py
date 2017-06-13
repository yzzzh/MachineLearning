# -*- coding: utf8 -*-
from mrjob.job import MRJob
from mrjob.step import MRStep

class MRMean(MRJob):
    #初始化
    def __init__(self,*args,**kwargs):
        super(MRMean,self).__init__(*args,**kwargs)
        self.inCount = 0
        self.inSum = 0
        self.inSqSum = 0

    #map的输入，相同的key会输入到同一个mapper
    def map(self,key,val):
        #不知道干嘛的
        if False:
            yield
        inVal = float(val)
        self.inCount += 1
        self.inSum += inVal
        self.inSqSum += inVal * inVal

    #map的输出,以迭代器的形式传递
    def map_final(self):
        inSum = self.inSum / self.inCount
        inSqSum = self.inSqSum / self.inCount
        yield (1,[self.inCount,inSum,inSqSum])

    #reducer处理输入并输出
    def reduce(self,key,values):
        inCount = 0
        inSum = 0
        inSqSum = 0
        for val in values:
            count = float(val[0])
            inCount += count
            inSum += count * float(val[1])
            inSqSum += count * float(val[2])
        inMean = inSum / inCount
        inSqMean = inSqSum / inCount
        yield (inMean,inSqMean)

    #设置处理的步骤
    def steps(self):
        return ([MRStep(mapper=self.map,reducer=self.reduce,mapper_final=self.map_final)])

if __name__ == '__main__':
    MRMean.run()