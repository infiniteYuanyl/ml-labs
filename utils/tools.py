from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import numpy as np

def visualize_2D(x,y,x_title='',y_title='',title=None):
       plt.ion()  # 打开交互模式
       fig = plt.figure()
       ax = fig.add_subplot(111)
       ax.set( title=title,
              ylabel=y_title, xlabel=x_title)
       plt.plot(x,y)
       
       plt.pause(1)
       plt.ioff()  # 关闭交互模式
       plt.show()
def visualize_lda(scores,predict):
       fig = plt.figure(figsize=(10,10)) 
       ax1 = fig.add_subplot(1,1,1)
       
       ax1.scatter(scores[predict==1],scores[predict==1],c='b',marker='o')
       ax1.scatter(scores[predict==0],scores[predict==0],c='r',marker='s')
       
       ax1.set_title("LDA")
       plt.show()

def visualize_perceptron(input,labels,w,theta):
       fig = plt.figure(figsize=(10,10)) 
       ax1 = fig.add_subplot(1,1,1)
       x = np.linspace(-6, 6, 5000, endpoint=True)
       y = (theta - w[0] * x) / w[1]
       
       ax1.scatter(input[labels==1,0],input[labels==1,1],c='b',marker='o')
       ax1.scatter(input[labels==-1,0],input[labels==-1,1],c='r',marker='s')
       ax1.scatter(x,y,c='c')
       ax1.set_title("perceptron")
       plt.show()
def visualize_svm(input,labels,w,bias,masks):
       fig = plt.figure(figsize=(10,10)) 
       ax1 = fig.add_subplot(1,1,1)
       x = np.linspace(-1, 8, 5000, endpoint=True)
       y = (bias + w[0] * x) / -w[1]
       ax1.scatter(x,y,c='c',linewidth=0.5)
       print()
       print('weight:',w.flatten())
       print('bias:',bias)
       print('mask_idx',np.where(masks == True))
       n = input.shape[0]
       for i in range(n):
            if masks[i]:
                x, y = input[i,:]
                plt.scatter(x, y, s=100, c = 'g', alpha=0.5, linewidth=0.5, edgecolor='purple')
       ax1.scatter(input[labels==1,0],input[labels==1,1],c='b',marker='o')
       ax1.scatter(input[labels==-1,0],input[labels==-1,1],c='r',marker='s')
       
       ax1.set_title("svm")
       plt.show()

def visualize_decision_tree(inTree):
       
    fig = plt.figure(1, facecolor='white')
    											
    fig.clf()																				
    axprops = dict(xticks=[], yticks=[])
    visualize_decision_tree.ax1 = plt.subplot(111, frameon=False, **axprops)    							
    plotTree.totalW = float(getNumLeafs(inTree))											
    plotTree.totalD = float(getTreeDepth(inTree))											
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;							
    plotTree(inTree, (0.5,1.0), '')															
    plt.show()
def visualize_naive_bayes(inputs, labels, predict):
    num2char = {1:' S', 2: 'M', 3: 'L'}
    labels = labels[:,np.newaxis]
    predict = predict[:,np.newaxis]
    data = np.hstack([inputs,labels,predict]).tolist()
    for i in range(inputs.shape[0]):
        data[i][1] = num2char[data[i][1]]

    # matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 展示中文字体
    matplotlib.rcParams["axes.unicode_minus"] = False  # 处理负刻度值
    kinds = ["A1", "A2", "label", "predict"]
    colors = ["#e41a1c", "#377eb8", "#00ccff", "#984ea3"]
    

    # 饼图下添加表格
    cellTexts = data
    rowLabels = [i for i in range(1,inputs.shape[0]+1)]
    plt.table(cellText=cellTexts,  # 简单理解为表示表格里的数据
              colWidths=[0.1]*4,  # 每个小格子的宽度 * 个数，要对应相应个数
              colLabels=kinds,  # 每列的名称
              colColours=colors,  # 每列名称颜色
              rowLabels=rowLabels,  # 每行的名称（从列名称的下一行开始）
              rowLoc="center",  # 行名称的对齐方式
              loc="center"  # 表格所在位置
              )
    plt.title("简单图形")
    # plt.figure(dpi=80)
    # plt.show()
def visualize_adaboost(inputs, labels, predict):
    
    labels = labels[:,np.newaxis]
    predict = predict[:,np.newaxis]
    
    data = np.hstack([inputs,labels,predict])
    data = np.around(data,decimals=1)
    
    print(data)
    # matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 展示中文字体
    matplotlib.rcParams["axes.unicode_minus"] = False  # 处理负刻度值
    kinds = ["feat1", "feat2", "label", "predict"]
    colors = ["#e41a1c", "#377eb8", "#00ccff", "#984ea3"]
    

    # 饼图下添加表格
    cellTexts = data
    rowLabels = [i for i in range(1,inputs.shape[0]+1)]
    plt.table(cellText=cellTexts,  # 简单理解为表示表格里的数据
              colWidths=[0.1]*4,  # 每个小格子的宽度 * 个数，要对应相应个数
              colLabels=kinds,  # 每列的名称
              colColours=colors,  # 每列名称颜色
              rowLabels=rowLabels,  # 每行的名称（从列名称的下一行开始）
              rowLoc="center",  # 行名称的对齐方式
              loc="center"  # 表格所在位置
              )
    plt.title("简单图形")
    plt.figure(dpi=80)
    plt.show()
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs



def getTreeDepth(myTree):
    maxDepth = 0												
    firstStr = next(iter(myTree))								
    secondDict = myTree[firstStr]								
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':				
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth			
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	arrow_args = dict(arrowstyle="<-")											
	# font = fm.FontProperties(fname=r"C:\\windows\\fonts\\simsunb.ttf", size=14)	
       
	visualize_decision_tree.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',	
		xytext=centerPt, textcoords='axes fraction',
		va="center", ha="center", bbox=nodeType, arrowprops=arrow_args\
                     # , FontProperties=font
                     )


def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]																
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	visualize_decision_tree.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
	decisionNode = dict(boxstyle="sawtooth", fc="0.8")										
	leafNode = dict(boxstyle="round4", fc="0.8")											
	numLeafs = getNumLeafs(myTree)  														
	depth = getTreeDepth(myTree)															
	firstStr = next(iter(myTree))																								
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)	
	plotMidText(cntrPt, parentPt, nodeTxt)													
	plotNode(firstStr, cntrPt, parentPt, decisionNode)										
	secondDict = myTree[firstStr]															
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD										
	for key in secondDict.keys():								
		if type(secondDict[key]).__name__=='dict':											
			plotTree(secondDict[key],cntrPt,str(key))        								
		else:																														
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


       


