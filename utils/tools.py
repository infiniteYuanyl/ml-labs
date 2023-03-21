from matplotlib import pyplot as plt
import matplotlib.font_manager as fm


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


       


