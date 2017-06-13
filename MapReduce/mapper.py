# -*- coding: utf8 -*-
import sys
from numpy import *

#mapper的任务是处理好数据后传递给reducer

#一个迭代器
def read_input(file):
    for line in file:
        yield line.rstrip()

input = read_input(sys.stdin)
#将迭代器转化为一个列表
_input = [float(line) for line in input]
numInput = len(_input)
_input = mat(_input)
sqInput = power(_input,2)
print('%d\t%f\t%f'%(numInput,mean(_input),mean(sqInput)))
#file参数用于重定向
print('repore:still alive',file=sys.stderr)
