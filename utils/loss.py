
import numpy as np


def MSEloss(predict,labels):
        loss = np.sum((predict - labels)**2)
        return loss
def L1loss(predict,labels):
        return np.sum(predict - labels)

def Errloss(predict,labels):
        '''
        错误分类的loss       
        '''
        labels = labels.astype(np.int32)
        
        loss = np.sum(predict != labels)
        return loss
def CEloss(predict,labels):
    pass
