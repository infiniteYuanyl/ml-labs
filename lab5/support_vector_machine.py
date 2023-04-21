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
import random
import scipy.spatial.distance as ssd
# sys.path.append('../')
# print(sys.path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
sys.path.insert(0, UTILS_DIR)
def kernel(x, y,mode='linear',sigma=1.0,times=2):
        '''
        :param x: N x d
        :param y: M x d
        :return: M x M
        '''
        if mode == 'linear':
            return np.dot(x ,y.T)
        elif mode == 'rbf':
            dist = ssd.cdist(x, y, 'euclidean')
            return np.exp(-np.square(dist) * sigma)
        elif mode == 'pkf':
            return (np.dot(x,y.T) +  1) ** times

class SVM(BaseModel):
    def __init__(self, dataloader=None, load_checkpoint=None, epochs=1, features_num=2, output_num=1,kernel_mode='linear',
                 penalty=0.8, samples=0, tolerance=0.001):
        super().__init__(dataloader, epochs)

        self.save_model = self.save_checkpoint
        self.load_checkpoint = load_checkpoint
        self.features_num = features_num
        self.output_num = output_num
        self.penalty = penalty
        self.kernel_mode = kernel_mode
        self.tolerance = tolerance
        self.samples = samples
        self.initialize_params()

    def initialize_params(self):
        # +1 为b的那一项，同时label也需要经过处理

        if self.load_checkpoint is not None:
            with np.load(self.load_checkpoint, allow_pickle=True) as f:
                self.alpha = f['alpha']
                self.bias = f['bias']
                self.mask = f['mask']
            print('load checkpoint from {}'.format(self.load_checkpoint))
        else:
            self.weights = np.zeros((self.features_num, self.output_num))
            self.bias = 0.0
            self.alpha = np.zeros(self.samples)
            print('initializing weight！')
        input, labels = self.dataloader.get_data()
        self.input = input
        self.labels = labels
        self.kernel = kernel(input,input,mode=self.kernel_mode)
        # print('weights:', self.alpha)

    def update_params(self):
        # self.W = self.W -lr * self.dW
        # self.bias = self.B -lr * self.dbias
        pass

    def compute_loss(self, predict, label):

        return Errloss(predict, label)

    def forward(self, input):
        if not hasattr(self,'x_sup'):
            return None
        k = kernel(input, self.x_sup)
        wx = self.y_sup[np.newaxis, :] * self.alpha_sup[np.newaxis, :] * k
        y_pred = np.sum(wx, axis=1) + self.bias
        return np.sign(y_pred)

    def backward(self, diff):
        input, labels = self.dataloader.get_data()
        self.simple_smo()
        self.weights = np.dot(input.T, (self.alpha * labels))

    def predict(self, input, labels):

        pred = self.forward(input)

    def evaluate(self, input, labels):
        #只对同样的样本进行评估
        test_data, test_labels = self.dataloader.get_data(mode='test')
        
        weight = np.dot(test_data.T,(self.alpha * test_labels).reshape(-1,1))
        visualize_svm(test_data, test_labels, weight,
                      self.bias, self.mask)

    def save_checkpoint(self, **kwargs):
        d = datetime.today().strftime('%Y%m%d_%H_%M_%S')
        d.split(' ')
        np.savez('checkpoints/'+d+'_svm.npz', mask=list(self.mask),
                 alpha=self.alpha, bias=self.bias, allow_pickle=True)
        print('checkpoint has been saved in %s!' %
              ('checkpoints/'+d+'_svm.npz'))

    def emerge(self, **kwargs):
        self.backward(None)

    def get_rand_j_by_i(self, i):
        '''
        通过i，随机获取一个和i不相同的j
        '''
        j = i
        while (j == i):
            j = int(random.uniform(0, self.samples))
        return j
    def error(self, index):
        '''
        计算差值Ei,计算差值Ei,i=index
        '''
        pred = np.dot((self.alpha * self.labels).T , self.kernel[:, index]) + self.bias
        return pred - self.labels[index]

    def not_kkt(self, select_i,ei):
        '''
        返回给定样本是否不满足KKT条件，如果不满足返回True
        '''
        cond1 = (self.labels[select_i] * ei) < - \
            self.tolerance and (self.alpha[select_i] < self.penalty)
        cond2 = (self.labels[select_i] * ei
                 ) > self.tolerance and (self.alpha[select_i] > 0)
        return cond1 or cond2

    def simple_smo(self):
        update = 0
        for i in range(self.samples):
            #   1.ei =  alphai*yi*kernel(xj*xi) +b - yi
            error_i = self.error(i)
            alpha_i_old = self.alpha[i].copy()
            #   2.选择一个违反kkt条件的样本，容忍度为tolerance
            if self.not_kkt( i,error_i):
                #   3.随机选取j,计算器gj和ej。
                j = self.get_rand_j_by_i(i)
                error_j = self.error(j)
                #   4.为alpha_i_old和alpha_j_old保存下来用于更新
                
                alpha_j_old = self.alpha[j].copy()
                #   5.计算对于类别不同的两个样本i，j，如果他们对应alpha的上下届L,H一样，则不继续优化
                if (self.labels[i] != self.labels[j]):  # alpha的上限与下限
                    L = max(0, self.alpha[j] - self.alpha[i])
                    H = min(self.penalty, self.penalty +
                            self.alpha[j] - self.alpha[i])
                else:
                    L = max(0, self.alpha[j] +
                            self.alpha[i] - self.penalty)
                    H = min(self.penalty, self.alpha[j] + self.alpha[i])
                if L == H:
                    continue
                #   6.计算η值，如果大于0也不继续优化
                eta = 2 * self.kernel[i,j] - self.kernel[i,j] - self.kernel[j,j]
                if eta > 0:
                    continue

                #   7.对i进行修改，修改量与j相同，但方向相反
                self.alpha[j] = alpha_j_old- (self.labels[j]*(error_i-error_j)/eta)
                # 通过L H的值优化
                self.alpha[j] = np.clip(self.alpha[j], L, H)

                #   8.判断alpha_j的优化程度，更新alpha_i，并根据新的alpha得到新的bias
                if (np.abs(self.alpha[j]-alpha_j_old) < self.tolerance):
                    continue
                self.alpha[i] = alpha_i_old + self.labels[j]*self.labels[i] * \
                    (alpha_j_old - self.alpha[j])

                b1 = self.bias - error_i - self.labels[i] * (self.alpha[i]  - alpha_i_old) * self.kernel[i,i]\
                    -self.labels[j] * (self.alpha[j]  - alpha_j_old) * self.kernel[i,j]
                b2 = self.bias - error_j - self.labels[i] * (self.alpha[i]  - alpha_i_old) * self.kernel[i,j]\
                    -self.labels[j] * (self.alpha[j]  - alpha_j_old) * self.kernel[j,j]

                if (0 < self.alpha[i] < self.penalty ):
                    self.bias = b1
                elif (0 < self.alpha[j] < self.penalty):
                    self.bias = b2
                else:
                    self.bias = (b1 + b2) / 2.0
                update = update + 1
        # 9.设置支持向量的索引TF为mask，那些alpha值比tolerance大的为支持向量（如果小于可能不是，而是需要忽略的向量），方便推理和可视化
        mask = self.alpha > self.tolerance
        self.mask = mask
        self.support_vectors = mask
        self.x_sup = self.input[mask]
        self.y_sup = self.labels[mask]
        self.alpha_sup = self.alpha[mask]
        #紧急指针，用于如果当前这一遍没有更新任何参数，调用紧急方法执行（调试时用的），可忽略
        if update != 0:
            self.ptr = True
