# -*- coding: utf8 -*-
from numpy import *
from tkinter import *
from RegressTree.regTree import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(minError,minNum):
    reDraw.figure.clf()
    reDraw.ax = reDraw.figure.add_subplot(111)
    #是否选中
    if checkBtnVal.get():
        if minNum < 2:
            minNum = 2
        myTree = createTree(reDraw.trainData, typeModel, modelError, minError, minNum)
        yExpected = testTree(myTree,reDraw.testData,modelTreeEval)
    else:
        myTree = createTree(reDraw.trainData, typeRegress, regressError, minError, minNum)
        yExpected = testTree(myTree,reDraw.testData,regressTreeEval)
    reDraw.ax.scatter(reDraw.trainData[:,0].A,reDraw.trainData[:,1].A,s=5)
    reDraw.ax.plot(reDraw.testData,yExpected)
    reDraw.canvas.show()

def getInput():
    #错误检测
    try:
        minNum = int(minNumText.get())
    except:
        minNum = 10
        minNumText.delete(0,END)
        minNumText.insert(0,'10')
    try:
        minError = float(minErrorText.get())
    except:
        minError = 1.0
        minErrorText.delete(0,END)
        minNumText.insert(0,'10')
    return minNum,minError

def drawNewTree():
    minNum,minError = getInput()
    reDraw(minNum,minError)

if __name__ == '__main__':
    #根对象
    root = Tk()

    #标签
    Label(root,text='Plot Place Holder').grid(row=1,columnspan=3)

    #figure利用TkAgg与canvas绑定，原来figure显示的内容将通过canvas显示出来
    reDraw.figure = Figure(figsize=(5,4),dpi=100)
    reDraw.canvas = FigureCanvasTkAgg(reDraw.figure,master=root)
    reDraw.canvas.show()
    reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)

    Label(root,text='Min Num').grid(row=1,column=0)
    #输入框
    minNumText = Entry(root)
    minNumText.grid(row=1,column=1)
    #默认值
    minNumText.insert(0,'10')
    Label(root,text='Min Error').grid(row=2,column=0)
    minErrorText = Entry(root)
    minErrorText.grid(row=2,column=1)
    minErrorText.insert(0,'1.0')

    Button(root,text='Draw',command=drawNewTree).grid(row=1,column=2,rowspan=3)

    #获取复选框的值
    checkBtnVal = IntVar()
    Checkbutton(root,text='Model Tree',variable=checkBtnVal).grid(row=3,column=0,columnspan=2)

    reDraw.trainData = mat(loadData('sine.txt'))
    reDraw.testData = mat(arange(min(reDraw.trainData[:,0]),max(reDraw.trainData[:,0]),0.1)).T

    reDraw(1.0,10)

    #运行
    root.mainloop()



