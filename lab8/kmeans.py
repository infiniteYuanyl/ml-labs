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

from tools import visualize_kmeans
# sys.path.append('../')
# print(sys.path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0, UTILS_DIR)
           
        
class Kmeans(BaseModel):
    def __init__(self, dataloader=None, load_checkpoint=None, epochs=1, features_num=2,\
                  output_categories=2,is_test=False):
        super().__init__(dataloader, epochs)

        self.save_model = self.save_checkpoint
        self.load_checkpoint = load_checkpoint
        self.features_num = features_num
        self.output_categories = output_categories
        self.mean_vector = np.zeros((features_num,output_categories))
        self.is_test = is_test
        
        self.initialize_params()

    def initialize_params(self):
        # +1 为b的那一项，同时label也需要经过处理

        if self.load_checkpoint is not None:
            with np.load(self.load_checkpoint, allow_pickle=True) as f:
                self.mean_vector = f['mean']
                
                
            print('load checkpoint from {}'.format(self.load_checkpoint))
        else:
            self.mean_vector = np.zeros((self.features_num,self.output_categories))
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
    def cluster_generate(self,input):
        input = np.repeat(input,self.output_categories,axis=1).reshape(-1,\
                        self.features_num,self.output_categories)#N * feature -> N * feature * cate
        
        l2_distance = np.sum((input - self.mean_vector) ** 2,axis = 1) #N * feature -> N * cate
        pred = np.argmin(l2_distance,axis = 1)
        return pred
    def forward(self, input):
        
        if np.sum(self.mean_vector) < 1e-6:
            return None
        pred = self.cluster_generate(input)
        return pred


            

    def backward(self, diff):
        if np.sum(self.mean_vector) > 1e-6:
            return None
        """
        1.随机初始化均值向量
        """
        samples_num = self.input.shape[0]
        random_idx = np.random.permutation(samples_num)[:self.output_categories]
        self.mean_vector = self.input[random_idx,:].T

        terminate_cnt = 0
        while terminate_cnt < self.output_categories:
            
            """
            2.计算各个样本到均值向量的距离，划分为簇
            """
            cluster = self.cluster_generate(self.input)

            """
            3.更新均值向量
            """
            terminate_cnt = 0
            cluster_idx = np.unique(cluster)
            for idx in cluster_idx:
                if abs((self.mean_vector[:,idx] - np.mean(self.input[np.argwhere(cluster == idx),:],axis=0)).all()) > 1e-4:
                    self.mean_vector[:,idx] = np.mean(self.input[np.argwhere(cluster == idx),:],axis=0)
                else :
                    terminate_cnt += 1

            #输出一下聚类结果
        
        self.evaluate(self.input,self.forward(self.input))
        
            

    def predict(self, input, labels):

        pred = self.forward(input)
        

    def evaluate(self, input, labels):
        #只对同样的样本进行评估
        #test_data, test_labels = self.dataloader.get_data(mode='test')
        
        visualize_kmeans(self.input,self.mean_vector.T,labels)
        

    def save_checkpoint(self, **kwargs):
        d = datetime.today().strftime('%Y%m%d_%H_%M_%S')
        d.split(' ')
        print("k-means-vector")
        print(self.mean_vector)
        
        np.savez('checkpoints/'+d+'_kmeans.npz', means=self.mean_vector,allow_pickle=True)
        print('checkpoint has been saved in %s!' %
              ('checkpoints/'+d+'_kmeans.npz'))

    