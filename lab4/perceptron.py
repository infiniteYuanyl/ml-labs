import copy
import sys
import numpy as np
from datetime import *
import tqdm
import time
import copy
import os
# sys.path.append('../')
# print(sys.path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0,UTILS_DIR)

from tools import visualize_perceptron
from base_model import BaseModel
from loss import MSEloss,Errloss,L1loss
class Perceptron(BaseModel):
    def __init__(self,dataloader=None,load_checkpoint=None,features_num=2,output_num=1,lr=0.01,epochs=1):
        super().__init__(dataloader,epochs=epochs)
        
        self.save_model = self.save_checkpoint
        self.load_checkpoint =  load_checkpoint
        self.features_num = features_num
        self.output_num = output_num
        self.lr = lr
        self.initialize_params()

    def initialize_params(self):
        if self.load_checkpoint is not None:
            with np.load(self.load_checkpoint,allow_pickle=True) as f:
                self.W = f['W']
                self.theta = f['Theta']
            print('load checkpoint from {}'.format(self.load_checkpoint))
        else:
            self.W = np.ones((self.features_num,self.output_num))
            self.theta = -1.0
            print('initializing weight！')
        print('W:',self.W)
        print('theta',self.theta)
        


    def update_params(self):
        self.W = self.W - self.lr *self.dW
        self.theta = self.theta - self.lr *self.dtheta
        
    def compute_loss(self,predict,label):
        
        # print(Errloss(predict,label))
        return Errloss(predict,label)

    def forward(self,input): 
        out = np.dot(input,self.W)
        self.input =  input
        out = out > self.theta
        out = 2*out -1
        
        return out.flatten()
    def backward(self, diff):  
        self.dW = np.dot(self.input.T,diff)
        self.dtheta = -np.mean(diff)
        
    def predict(self,input,labels):
        
        pred = self.forward(input)

    def evaluate(self,input,labels):
        test_data,test_labels= self.dataloader.get_data(mode='test')
        visualize_perceptron(test_data,test_labels,self.W,self.theta)
    def save_checkpoint(self,**kwargs):
        print('W:',self.W)
        print('theta',self.theta)
        d= datetime.today().strftime('%Y%m%d_%H_%M_%S')
        d.split(' ')
        np.savez('checkpoints/'+d+'_perceptron.npz',W=self.W,Theta=self.theta,allow_pickle=True)
        print('checkpoint has been saved in %s!'%('checkpoints/'+d+'_perceptron.npz'))

        



	




   



