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
from collections import OrderedDict
from neural_network_layers import SoftmaxLossLayer
from tools import visualize_adaboost
# sys.path.append('../')
# print(sys.path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0, UTILS_DIR)
class LinearRegression:
    def __init__(self,data,labels,output_num=2,load_params=None):

        self.feature_num = data.shape[1]
        self.inputs = data = np.c_[data, np.ones(data.shape[0])]
        self.labels = labels
        self.output_num = output_num
        self.lr = 0.01
        self.params = load_params
        self.softmax = SoftmaxLossLayer()
        self.loss_weights = np.ones(data.shape[0]) / data.shape[0]
        self.err_rate = 0
        self.initialize_weight()

    def initialize_weight(self):
        # +1 为b的那一项，同时label也需要经过处理
        
        self.params = 1.0 * np.ones((self.feature_num+1,self.output_num),dtype=np.float64)
        print('initializing weight！')
        print('weight:',self.params.shape)
        print('features_num:',self.feature_num)
    def load_weights(self,weights):
        self.params = weights

    def update_params(self,dloss):
        
        self.params = self.params - self.lr * np.dot(self.inputs.T,dloss)

    def forward(self):
        
        predict = np.dot(self.inputs,self.params)
        prob = self.softmax.forward(predict)
        self.pred = np.argmax(prob,axis=1)
        loss = self.softmax.get_loss(self.labels)
        dloss = self.softmax.backward()
        dloss = dloss * self.labels.shape
        dloss = dloss * self.loss_weights.reshape(-1,1)
        self.update_params(dloss)
        return prob

    def train(self,epochs,show=False):
        
        epochs=int(epochs)
             
        for epoch in range(epochs):
            start = time.perf_counter()
            prob = self.forward()
            end = time.perf_counter()
           
        

class AdaBoost(BaseModel):
    def __init__(self, dataloader=None, load_checkpoint=None, epochs=1, features_num=2,\
                  output_num=1,classifiers_num=10,classifiers_train_epochs=100,classifier=LinearRegression,is_test=False):
        super().__init__(dataloader, epochs)

        self.save_model = self.save_checkpoint
        self.load_checkpoint = load_checkpoint
        self.features_num = features_num
        self.output_num = output_num
        self.classifier = classifier
        self.classifiers_num = classifiers_num
        self.classifiers_epochs = classifiers_train_epochs
        self.classifiers_params = []
        self.alphas = []
        self.is_test = is_test
        
        self.initialize_params()

    def initialize_params(self):
        # +1 为b的那一项，同时label也需要经过处理

        if self.load_checkpoint is not None:
            with np.load(self.load_checkpoint, allow_pickle=True) as f:
                self.classifiers_params = f['classifiers']
                self.alphas = f['alpha']
                
                
            print('load checkpoint from {}'.format(self.load_checkpoint))
        else:
            self.cond_prob_dict = {}
            print('initializing weight！')
        self.input, self.labels = self.dataloader.get_data(mode='test' if self.is_test else 'train')
        self.classifier = self.classifier(data=self.input,labels=self.labels,output_num=self.output_num)
        
        # print('weights:', self.alpha)

    def update_params(self):
        # self.W = self.W -lr * self.dW
        # self.bias = self.B -lr * self.dbias
        pass

    def compute_loss(self, predict, label):

        return Errloss(predict, label)

    def forward(self, input):
        if self.classifiers_params == []:
            return None
        sum_prob = np.zeros((self.input.shape[0],self.output_num))

        for i,classifer_params in enumerate(self.classifiers_params):
            self.classifier.load_weights(classifer_params)
            prob = self.classifier.forward()
            sum_prob += self.alphas[i] * prob
        pred = np.argmax(sum_prob,axis=1)
        return pred 

            

    def backward(self, diff):
        
        iters = self.classifiers_num
        self.classifiers_params = []
        for iter in range(iters):
            
            self.classifier.train(self.classifiers_epochs)
            if iter == 0:
                weights = np.ones(self.labels.shape[0])/self.labels.shape[0]
            mask = self.classifier.pred != self.labels
            mask_weights = weights
            mask_weights[np.argwhere(mask==False)] = 0
            
            
            
            err = np.sum(mask_weights)
            
            err = float(err)
            print('\nerr',err)
            if err >= 0.5:
                print('err go out!')
                break
            if err == 0:
                print('train ok')
                self.alphas.append(5)
                self.classifiers_params.append(self.classifier.params)
                break
            alpha = 0.5 * math.log((1-err)/err)
            
            norm_factor = np.sum(weights * np.exp(-alpha * (2 * ~mask -1)))
            
            weights = weights/norm_factor * np.exp(-alpha * (2 * ~mask -1))
            self.classifier.loss_weights = weights
            self.alphas.append(alpha)
            self.classifiers_params.append(self.classifier.params)

        # sum_prob = np.zeros((self.input.shape[0],self.output_num))
        # for i,classifer_params in enumerate(self.classifiers_params):
        #     # print('alpha',alpha.shape)
        #     self.classifier.load_weights(classifer_params)
            
        #     prob = self.classifier.forward()
        #     sum_prob += self.alphas[i] * prob
        # pred = np.argmax(sum_prob,axis=1)
        # print('res',pred == self.labels)
            

    def predict(self, input, labels):

        pred = self.forward(input)
        

    def evaluate(self, input, labels):
        #只对同样的样本进行评估
        test_data, test_labels = self.dataloader.get_data(mode='test')
        
        predict = self.forward(input)
        print('predict:',end='')
        print(predict)
        print('labels: ',end='')
        print(test_labels)
        acc = np.sum(predict == test_labels) / test_data.shape[0]
        print("test accuracy is ",acc)
        visualize_adaboost(test_data,test_labels,predict)

    def save_checkpoint(self, **kwargs):
        d = datetime.today().strftime('%Y%m%d_%H_%M_%S')
        d.split(' ')
        print("alphas")
        print(self.alphas)
        print("classifiers params:")
        for i in range(len(self.classifiers_params)):
            print(self.classifiers_params[i])
        np.savez('checkpoints/'+d+'_adaboost.npz', classifiers=self.classifiers_params,alpha=self.alphas,allow_pickle=True)
        print('checkpoint has been saved in %s!' %
              ('checkpoints/'+d+'_adaboost.npz'))

    