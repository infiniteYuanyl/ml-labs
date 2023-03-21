

from data_loader import DataLoader
import tqdm
from datetime import *
import time
import numpy as np
import copy
from tools import visualize_2D
class BaseModel(object):
    def __init__(self,dataloader,epochs=1) :
        if type(dataloader) != DataLoader:
            raise TypeError('DataLoader type error!')
        self.dataloader =  dataloader
        self.train_epochs = epochs
        self.loss =0.0
        self.lr = 0.02
        self.show = False
        self.save_model = self.save_checkpoint
        self.eval = self.evaluate
    def initialize_params(self):
        pass
    def forward(self,input):
        pass
    def backward(self,diff):
        pass
    def update_params(self):
        pass
    def compute_loss(self,predict,labels):
        pass
    def predict(self):
        pass
    def train(self):
        loss_plt = []
        epochs = self.train_epochs
        pbar = tqdm.tqdm(range(epochs), ncols=150)       
        for epoch,_ in enumerate(pbar):
            
            data,labels = self.dataloader.get_items(mode='train')
            start = time.perf_counter()
            predict = self.forward(data)
            self.output = predict
            self.loss = self.compute_loss(predict,labels)
            diff=None
            if predict is not None:
                diff = predict - labels
            self.backward(diff)
            end = time.perf_counter()
            self.update_params()
            pbar.set_description(f" Train Epoch {epoch+1}/{epochs}")
            pbar.set_postfix( loss=self.loss, lr=str(round(self.lr,4)),cost_time = str(round((end-start)*1000,3)) +'ms')
            loss_plt.append(copy.deepcopy(self.loss))
            
        if self.show:
            visualize_2D(np.linspace(1,epochs+1,epochs),np.array(loss_plt),'epochs','loss','loss graph')
        self.save_model()
        print('Train over')
    def val(self):
        pass
    def test(self):

        data,labels = self.dataloader.get_items(mode='test')
        test_num = data.shape[0]
            
        start = time.perf_counter()
        predict = self.predict(data,labels)
        self.output = predict
        end = time.perf_counter()
        #print('cost_time: %f ms'%(end-start)*1000)
        
        self.evaluate(predict,labels)
        print('Test over')
        
    def evaluate(self,predict,labels):
        pass

    def save_checkpoint(self,**kwargs):
        pass
