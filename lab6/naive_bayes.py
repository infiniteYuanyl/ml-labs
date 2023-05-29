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
from copy import deepcopy
from collections import OrderedDict

from tools import visualize_naive_bayes
# sys.path.append('../')
# print(sys.path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0, UTILS_DIR)


class NaiveBayes(BaseModel):
    def __init__(self, dataloader=None, load_checkpoint=None, epochs=1, features_num=2, output_num=1):
        super().__init__(dataloader, epochs)

        self.save_model = self.save_checkpoint
        self.load_checkpoint = load_checkpoint
        self.features_num = features_num
        self.output_num = output_num
        
        self.initialize_params()

    def initialize_params(self):
        # +1 为b的那一项，同时label也需要经过处理

        if self.load_checkpoint is not None:
            with np.load(self.load_checkpoint, allow_pickle=True) as f:
                self.cond_prob_dict = f['prob'].reshape((-1))[0]
                self.categories = f['cate']
                print(type(self.cond_prob_dict))
                
            print('load checkpoint from {}'.format(self.load_checkpoint))
        else:
            self.cond_prob_dict = {}
            print('initializing weight！')
        self.input, self.labels = self.dataloader.get_data()
        
        
        # print('weights:', self.alpha)

    def update_params(self):
        # self.W = self.W -lr * self.dW
        # self.bias = self.B -lr * self.dbias
        pass

    def compute_loss(self, predict, label):

        return Errloss(predict, label)

    def forward(self, input):
        if self.cond_prob_dict =={}:
            return None
        predict = np.zeros(input.shape[0])
        
        #print('dict',self.cond_prob_dict)
        for i in range(input.shape[0]):
            input_predict = -2
            max_category_prob = -float('inf')
            x = input[i]
            for category in self.categories:
                category_prob = 0.0
                for feature_id in range(input.shape[1]):
                    feature_dict = self.cond_prob_dict[feature_id]
                    prob = feature_dict[(x[feature_id],category)]
                    if prob == 0:
                        category_prob = -float('inf')
                        break
                    log_prob = np.log(prob)
                    category_prob += log_prob
                if category_prob > max_category_prob:
                    max_category_prob = category_prob
                    input_predict = category
            predict[i] = input_predict
        
        return predict
            

            

    def backward(self, diff):
        
        categories = list(np.unique(self.labels))
        input_featues_num = self.input.shape[1]
        #print('categories',categories)
        self.cond_prob_dict = {}
        
        for feature_id in range(input_featues_num):
            feature_values = np.unique(self.input[:,feature_id])
            prob_list = []
            for category in categories:
                for feature_value in feature_values:
                    category_mask = self.labels == category
                    #print('category mask',category_mask)
                    feature_mask = self.input[:,feature_id] == feature_value
                    #print('feature mask',feature_mask)
                    # 统计在该类别下，各个特征值出现的概率
                    both_mask = category_mask & feature_mask
                    #print('all mask',both_mask)
                    prob = np.sum(both_mask) / np.sum(category_mask)
                    prob_list.append(((feature_value,category),prob))
            self.cond_prob_dict[feature_id] = OrderedDict(prob_list)

        
        self.categories = categories
        

    def predict(self, input, labels):

        pred = self.forward(input)
        

    def evaluate(self,**kwargs):
        #只对同样的样本进行评估
        test_data, test_labels = self.dataloader.get_data(mode='test')
        
        predict = self.forward(test_data)
        
        predict = predict.astype(np.int32)
        self.err = np.sum(predict!=test_labels) / test_data.shape[0]
        # print('predict',predict)
        # print('err',np.sum(predict!=labels))
        #visualize_naive_bayes(test_data,test_labels,predict)
    
        
    def save_checkpoint(self, **kwargs):
        d = datetime.today().strftime('%Y%m%d_%H_%M_%S')
        d.split(' ')
        print("prob_dict",self.cond_prob_dict)
        np.savez('checkpoints/'+d+'_nv_bayes.npz', prob=dict(self.cond_prob_dict),cate = self.categories\
                  ,allow_pickle=True)
        print('checkpoint has been saved in %s!' %
              ('checkpoints/'+d+'_nv_bayes.npz'))
        return 'checkpoints/'+d+'_nv_bayes.npz'

    