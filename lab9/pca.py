from tools import visualize_svm
from loss import MSEloss, Errloss
from base_model import BaseModel
import copy
import sys
import numpy as np
from numpy import *
from datetime import *
import tqdm
import time
import copy
import os
import math
from copy import deepcopy

from tools import visualize_pca
# sys.path.append('../')
# print(sys.path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0, UTILS_DIR)
           
        
class PCA(BaseModel):
    def __init__(self, dataloader=None, load_checkpoint=None, epochs=1, features_num=2,\
                  output_feature_num=2,is_test=False):
        super().__init__(dataloader, epochs)

        self.save_model = self.save_checkpoint
        self.load_checkpoint = load_checkpoint
        self.features_num = features_num
        self.output_feature_num = output_feature_num
        self.is_test = is_test
        
        self.initialize_params()

    def initialize_params(self):
        # +1 为b的那一项，同时label也需要经过处理

        if self.load_checkpoint is not None:
            with np.load(self.load_checkpoint, allow_pickle=True) as f:
                self.mean_vector = f['mean']
                
                
            print('load checkpoint from {}'.format(self.load_checkpoint))
        else:
            # self.mean_vector = np.zeros((self.features_num,self.output_categories))
            print('initializing weight！')
        self.input, self.labels = self.dataloader.get_data(mode='test' if self.is_test else 'train')
        #self.classifier = self.classifier(data=self.input,labels=self.labels,output_num=self.output_num)
        
        # print('weights:', self.alpha)

    def update_params(self):
        # self.W = self.W -lr * self.dW
        # self.bias = self.B -lr * self.dbias
        pass

    def compute_loss(self, predict, label):

        return Errloss(predict, label)
    
    def forward(self, input):
        
        return None


            

    def backward(self, diff):
        # 1.样本中心化
    
        x = self.input - np.mean(self.input,axis=0)
        
        covariance_matrix = np.cov(x, rowvar=False)

        # 2.进行特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # 3.根据特征值排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # 4.取前d个最大特征值对应的特征向量，这里取最大output_feature个
        
        selected_eigenvectors = sorted_eigenvectors[:, :self.output_feature_num]
        self.w = selected_eigenvectors
        #print('W',selected_eigenvectors)
        #可视化
        visualize_pca(x,self.w,sorted_eigenvectors[:,:self.features_num])
        
        
        
            

    def predict(self, input, labels):

        pred = self.forward(input)
        

    def evaluate(self, input, labels):
        #只对同样的样本进行评估
        #test_data, test_labels = self.dataloader.get_data(mode='test')
        
        
        pass
        

    def save_checkpoint(self, **kwargs):
        d = datetime.today().strftime('%Y%m%d_%H_%M_%S')
        d.split(' ')
        
        print()
        print('W',self.w)
        np.savez('checkpoints/'+d+'_pca.npz', W=self.w,allow_pickle=True)
        print('checkpoint has been saved in %s!' %
              ('checkpoints/'+d+'_pca.npz'))

    