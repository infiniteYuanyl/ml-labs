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

from tools import visualize_decision_tree
from base_model import BaseModel
from loss import MSEloss,Errloss
class Model(BaseModel):
    def __init__(self,dataloader=None,load_checkpoint=None):
        super().__init__(dataloader)
        
        self.save_model = self.save_checkpoint
        self.load_checkpoint =  load_checkpoint
        self.initialize_params()

    def initialize_params(self):
        # +1 为b的那一项，同时label也需要经过处理
        if self.load_checkpoint is not None:
            with np.load(self.load_checkpoint,allow_pickle=True) as f:
                self.tree_root = f['Tree'].tolist()
            print('load checkpoint from {}'.format(self.load_checkpoint))
        else:
            self.tree_root = {}
            self.d_tree = {}
            print('initializing weight！')
        print('c45 decision tree:',self.tree_root)


    def update_params(self):
        pass
    def compute_loss(self,predict,label):
        pass

    def forward(self,input):     
        
        pass
    def backward(self, diff):
        input,labels = self.dataloader.get_data()
        pass
        
    def predict(self,input,labels):
        
        pass

    def evaluate(self,input,labels):
        
        pass
    def save_checkpoint(self,**kwargs):
        # d= datetime.today().strftime('%Y%m%d_%H_%M_%S')
        # d.split(' ')
        # np.savez('checkpoints/'+d+'_tree_root.npz',Tree=dict(self.tree_root),allow_pickle=True)
        # print('checkpoint has been saved in %s!'%('checkpoints/'+d+'_tree_root.npz'))
        pass



	




   



