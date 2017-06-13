# -*- coding: utf8 -*-
import sys
from numpy import *

def read_input(file):
    for line in file:
        yield line.rstrip()

#根据各个mapper输出的局部mean和sqmean得到全局的mean和sqmean
_input = read_input(sys.stdin)
_input = [line.split('\t') for line in _input]
totalNum = 0
totalMean = 0.0
totalSqMean = 0.0
for each in _input:
    num = float(each[0])
    totalNum += num
    totalMean += num * float(each[1])
    totalSqMean += num * float(each[2])

globalMean = totalMean/totalNum
globalSqMean = totalSqMean/totalNum
print('%d\t%f\t%f'%(totalNum,globalMean,globalSqMean))
print('report : alive',file=sys.stderr)