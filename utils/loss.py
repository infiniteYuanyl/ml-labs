
import numpy as np


def MSEloss(predict,labels):
        loss = np.sum((predict - labels)**2)
        return loss
def Errloss(predict,labels):
        '''
        错误分类的loss       
        '''
        loss = np.sum(predict != labels)
        return loss
def CEloss(predict,labels):
    pass