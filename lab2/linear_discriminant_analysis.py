import copy
import sys
import numpy as np
from datetime import *
import tqdm
import time
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0,UTILS_DIR)

from base_model import BaseModel
from loss import MSEloss,Errloss
from tools import visualize_lda
class LDA(BaseModel):
    def __init__(self,dataloader,epochs,load_checkpoint=None,is_test=False):
        super().__init__(dataloader,epochs)
        
        self.save_model = self.save_checkpoint
        self.load_checkpoint = load_checkpoint
        self.lr = 0.01 
        self.is_test = is_test
        self.initialize_params()

    def initialize_params(self):
        # +1 为b的那一项，同时label也需要经过处理
        data,labels = self.dataloader.get_data(mode='test' if self.is_test else 'train')
        self.classes_, yidx = np.unique(labels, return_inverse=True)
        self.samples_num ,self.features_num = data.shape
        self.priors_ = np.bincount(labels.astype(np.int32)) / self.samples_num
        
        means = np.zeros((len(self.classes_), self.features_num)) 
        np.add.at(means, yidx, data)
        self.input_means = means/ np.expand_dims(np.bincount(labels.astype(np.int32)), 1)

        self.cov_matrix = [np.cov(data[labels == group].T) \
                          for idx, group in enumerate(self.classes_)]
        self.covariance_ = sum(self.cov_matrix) / len(self.cov_matrix)
        #5、计算总体均值向量
        self.input_bar = np.dot(np.expand_dims(self.priors_, axis=0), self.input_means)

        if self.load_checkpoint is not None:
            with np.load(self.load_checkpoint) as f:
                self.params = f['W']
                self.lr =f['LR']
                self.covariance_ = f['CONV']
                self.input_means = f['MEANS']
                self.priors_ = f['PRIORS']
            print('load checkpoint from {}'.format(self.load_checkpoint))
        else:
            self.params = 1.0 * np.ones(self.features_num,dtype=np.float64)
            self.dparams = np.zeros(self.features_num,dtype=np.float64)
            print('initializing weight！')
        print('weight:',self.params.shape)
        print('features_num:',self.features_num)


    def update_params(self):
        self.params = self.dparams
    def compute_loss(self,predict,labels):
        return Errloss(predict,labels)

    def forward(self,input):
        
        scores = np.dot(input,self.params) 
        self.scores = scores
        # mask = np.argwhere(scores > self.sshresh )
        # predict = np.zeros(scores.shape)
        # predict[mask] = 1
        return scores
    def backward(self, diff):
        input,labels = self.dataloader.get_data()
        
        Sw = self.covariance_
        #求类间平均散度
        Sb = sum([sum(labels == item)*np.dot((self.input_means[idx,None] - self.input_bar).T, (self.input_means[idx,None] - self.input_bar)) \
                  for idx, item in enumerate(self.classes_)]) / (self.samples_num - 1)
        #SVD求Sw的逆矩阵
        U,S,V = np.linalg.svd(Sw)
        Sw = np.dot(np.dot(np.dot(V.T, np.linalg.inv(np.diag(S))), U.T), Sb)
        #求特征值和特征向量，并取实数部分
        la,vectors = np.linalg.eig(Sw)
        la = np.real(la)
        vectors = np.real(vectors)
        #特征值的下标从大到小排列
        la_idx = np.argsort(-la)
        #默认选取(N-1)个特征值的下标
        
        n_components = len(self.classes_)-1
        #选取特征值和向量
        lambda_index = la_idx[:n_components]
        w = vectors[:,lambda_index]
        
        self.dparams = w
        self.n_components = n_components
       
        
        # #分为0,1两类
        # # print('ooo  ',np.argwhere(raw_x[:,-1]==1).shape)
        # pos_x = raw_x[np.argwhere(raw_x[:,-1]==1).flatten(),:]
        # neg_x = raw_x[np.argwhere(raw_x[:,-1]==0).flatten(),:]
        # # print('net ',neg_x.shape)
        # # 剔除标签
        # pos_x = pos_x[:,:-1]
        # neg_x = neg_x[:,:-1]
        # # pos_x[:, :-1] = (pos_x[:, :-1] - np.mean(pos_x[:, :-1],axis=1)) / np.std(pos_x[:, :-1],axis=1)
        # # neg_x[:, :-1] = (neg_x[:, :-1] - np.mean(neg_x[:, :-1],axis=1)) / np.std(neg_x[:, :-1],axis=1)
        # miu_pos = np.mean(pos_x,axis=0)
        # miu_neg = np.mean(neg_x,axis=0)
        # # print(miu_neg.shape)
        # # print(neg_x.shape)
        # Sw = np.mean(np.dot((pos_x - miu_pos).T,(pos_x - miu_pos)),axis=0) + np.mean(np.dot((neg_x - miu_neg).T,(neg_x - miu_neg)),axis=0)
        
        # Sb = np.mean(np.dot((miu_pos-miu_neg).T,(miu_pos-miu_neg)),axis=0)
        # eps = 1e-8
        # # 增加正则向，防止无法求解逆
        # Sw += eps
        # U, S, V = np.linalg.svd(Sw)
        # Sw_  = np.dot(np.dot(V.T, np.linalg.pinv(np.diag(S))), U.T)
        # self.dparams = np.dot(Sw_,(miu_pos - miu_neg).T)
    def predict(self,input,labels):
        self.scores = np.dot(input,self.params) 
        
        # mean_ += np.log(np.expand_dims(self.priors_, axis=0))
        Sigma = self.covariance_
        print(Sigma.shape)
        print(self.input_means.shape)
        U,S,V = np.linalg.svd(Sigma)
        Sn = np.linalg.inv(np.diag(S))
        Sigman = np.dot(np.dot(V.T,Sn),U.T)
        #线性判别函数值
        value = np.log(np.expand_dims(self.priors_, axis=0)) - \
        0.5*np.multiply(np.dot(self.input_means, Sigman).T, self.input_means.T).sum(axis=0).reshape(1,-1) + \
        np.dot(np.dot(input, Sigman), self.input_means.T)
        return np.argmax(value/np.expand_dims(value.sum(axis=1),1),axis=1)
    def save_checkpoint(self,**kwargs):
        d= datetime.today().strftime('%Y%m%d_%H_%M_%S')
        d.split(' ')
        print('w: ',self.params)
        np.savez('checkpoints/'+d+'_paramsW.npz',W=self.params,LR=self.lr,CONV=self.covariance_,MEANS=self.input_means,PRIORS=self.priors_)
        print('checkpoint has been saved in %s!'%('checkpoints/'+d+'_paramsW.npz'))
    def evaluate(self,predict,labels):
        print('w:',self.params)
        # print('',labels)
        # print('predict',predict)
        err = np.sum(predict!=labels)
        print('samples_num',self.samples_num+1)
        
        print('err_num',err)
        print('mAcc',100 - 100* err/self.samples_num)
        print('predict',predict)
        visualize_lda(self.scores,predict)

    # def train(self,epochs,show=False):
    #     num = self.train_data.shape[0]
    #     epochs=int(epochs)
    #     print('train_num :', num)
    #     loss_plt = []
    #     pbar = tqdm.tqdm(range(epochs), ncols=150)       
    #     for epoch,_ in enumerate(pbar):

            
    #         data,labels = self.split_data(self.train_data)
    #         data = np.c_[data, np.ones(data.shape[0])]
    #         start = time.perf_counter()
    #         self.forward(data,labels,eval=False)
    #         end = time.perf_counter()
    #         # if epoch % 100 ==1:print('epoch: {} MSEloss: {}'.format(epoch+1, self.loss/self.train_data.shape[0]))
    #         pbar.set_description(f" Train Epoch {epoch+1}/{epochs}")
    #         pbar.set_postfix( loss=self.loss/self.train_data.shape[0], lr=str(round(self.lr,4)),cost_time = str(round((end-start)*1000,3)) +'ms')
    #         loss_plt.append(copy.deepcopy(self.loss/self.train_data.shape[0]))
    #         if self.lr >=0.0001:self.lr = self.lr * self.beta
    #         np.random.shuffle(self.train_data)
    #         # if epoch % 100 ==1:print(self.params)
    #     if show:
    #         visualize_2D(np.linspace(1,epochs+1,epochs),np.array(loss_plt),'epochs','loss','loss graph')
    #     print('Train over')
    #     d= datetime.today().strftime('%Y%m%d_%H_%M_%S')
    #     d.split(' ')
    #     np.savez('checkpoints/'+d+'_paramsW.npz',W=self.params,LR=self.lr)
    #     print('checkpoint has been saved in %s!'%('checkpoints/'+d+'_paramsW.npz'))

    # def test(self):
    #     print('test start')
    #     print('test cases:',self.test_data.shape[0])
    #     if self.test_data is None:
    #         raise ValueError('test_data not loaded!')
    #     self.eval=True
    #     data, labels = self.split_data(self.test_data,shuffled=True)
    #     data = np.c_[data, np.ones(data.shape[0])]
    #     self.forward(data, labels, eval=True)

    #     print('Test over! MSEloss:{},mAcc: {}'.format(self.loss/self.test_data.shape[0],self.Acc))
   





