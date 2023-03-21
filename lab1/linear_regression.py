import copy

import numpy as np
from datetime import *
import tqdm
import time
from utils import visualize_2D
class LinearRegression:
    def __init__(self,train_data,test_data=None,update_type='gradient',load_checkpoint=None):

        self.feature_num = train_data.shape[1] -1 
        self.train_data = train_data
        self.test_data = test_data
        self.update_type = update_type
        self.lr = 0.3
        self.beta = 0.9997
        self.loss = 0.0
        self.eval = False
        self.Acc = 0.0
        self.load_checkpoint = load_checkpoint
        self.initialize_weight()

    def initialize_weight(self):
        # +1 为b的那一项，同时label也需要经过处理
        if self.load_checkpoint is not None:
            with np.load(self.load_checkpoint) as f:
                self.params = f['W']
                self.lr =f['LR']

            print('load checkpoint from {}'.format(self.load_checkpoint))
        else:
            self.params = 1.0 * np.ones(self.feature_num+1,dtype=np.float64)
            print('initializing weight！')
        print('weight:',self.params.shape)
        print('features_num:',self.feature_num)
    def split_data(self,origin_data,shuffled=False):
        if shuffled:
            np.random.shuffle(origin_data)
        data = origin_data[:,:-1]
        labels = origin_data[:,-1]
        return data,labels


    def MSEloss(self,predict,label):
        loss = np.sum((predict - label)**2)
        # print('loss:',loss)
        self.loss = loss

        dloss = (predict - label)
        if self.eval:
            print('Acc:',(1-abs(np.mean(dloss))/label)*100)
            self.Acc+=(1-abs(np.mean(dloss))/label)*100
        
        # print('dloss: ',dloss.shape)
        return dloss
    def update_params(self,dloss,x):
        if self.update_type == 'min_2squares':
            self.step_by_leastsquare(dloss)
        elif self.update_type == 'gradient':
            self.step_by_gradient(dloss,x)
        else:
            raise NotImplementedError('methods of update_params are not implemented!')

    def step_by_leastsquare(self,loss):
        pass
    def step_by_gradient(self,dloss,x):
        #print(self.lr * np.dot(x,dloss))
        #print(dloss.shape)

        self.params = self.params - self.lr * np.dot(x.T,dloss)/self.train_data.shape[0]

    def forward_single(self,x,idx,labels,eval=False):
        #predict 为一个1 * 1 的数

        predict = np.dot(x.T,self.params)
        
        _,dloss = self.MSEloss(predict,labels[idx])
        if not eval :
            self.update_params(dloss,x)
        return 0
    def forward(self,x,labels,eval=False):
       
        predict = np.dot(x,self.params)
        
        dloss = self.MSEloss(predict,labels)
        if not eval :
            self.update_params(dloss,x)
        return 0

    def train(self,epochs,show=False):
        
        epochs=int(epochs)
        loss_plt = []
        pbar = tqdm.tqdm(range(epochs), ncols=150)       
        for epoch,_ in enumerate(pbar):

            
            data,labels = self.split_data(self.train_data)
            data = np.c_[data, np.ones(data.shape[0])]
            start = time.perf_counter()
            self.forward(data,labels,eval=False)
            end = time.perf_counter()
            # if epoch % 100 ==1:print('epoch: {} MSEloss: {}'.format(epoch+1, self.loss/self.train_data.shape[0]))
            pbar.set_description(f" Train Epoch {epoch+1}/{epochs}")
            pbar.set_postfix( loss=self.loss/self.train_data.shape[0], lr=str(round(self.lr,4)),cost_time = str(round((end-start)*1000,3)) +'ms')
            loss_plt.append(copy.deepcopy(self.loss/self.train_data.shape[0]))
            if self.lr >=0.0001:self.lr = self.lr * self.beta
            np.random.shuffle(self.train_data)
            # if epoch % 100 ==1:print(self.params)
        if show:
            visualize_2D(np.linspace(1,epochs+1,epochs),np.array(loss_plt),'epochs','loss','loss graph')
        print('Train over')
        d= datetime.today().strftime('%Y%m%d_%H_%M_%S')
        d.split(' ')
        np.savez('checkpoints/'+d+'_paramsW.npz',W=self.params,LR=self.lr)
        print('checkpoint has been saved in %s!'%('checkpoints/'+d+'_paramsW.npz'))

    def test(self):
        print('test start')
        print('test cases:',self.test_data.shape[0])
        if self.test_data is None:
            raise ValueError('test_data not loaded!')
        self.eval=True
        data, labels = self.split_data(self.test_data,shuffled=True)
        data = np.c_[data, np.ones(data.shape[0])]
        self.forward(data, labels, eval=True)

        print('Test over! MSEloss:{},mAcc: {}'.format(self.loss/self.test_data.shape[0],self.Acc))
   



