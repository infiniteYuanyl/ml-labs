
import numpy as np

class DataLoader:
    def __init__(self,data,shuffle=True,batch=0,k_fold=0,val=False,test=False,split_idx=[],data_type=np.int32):
        
        self.mode2idx = {'train':0,'val':1,'test':2}
        self.batch_size = batch
        self.shuffle = shuffle
        self.data_type = data_type
        self.has_val = val
        self.has_test = test
        if split_idx != []:
            self.normal_fixed_split(data,split_idx)
        elif k_fold !=0:
            self.k_fold_cross_validation_split(data,k_fold)
    def get_items(self,mode='train',shuffle=False,batch_norm=False,feature_norm=False,type=np.int32):
        data = self.data[self.mode2idx[mode]]
        if data == []:
            raise ValueError('%s data doesn\'t exist!\n Please check your input!',mode)
        if shuffle:
            np.random.shuffle(data) 
        
        if batch_norm:
            data[:, :-1] = (data[:, :-1] - np.mean(data[:, :-1],axis=0)) / np.std(data[:, :-1],axis=0)  # 非标签列列均值方差归一化
        if feature_norm:
            data = data.T
            data[:-1, :] = (data[:-1, :] - np.mean(data[:-1, :],axis=0)) / np.std(data[:-1, :],axis=0)  # 非标签列列均值方差归一化
            data = data.T
        self.input_data = data[:,:-1].astype(self.data_type)
        self.labels = data[:,-1].astype(np.int32)
        return self.input_data,self.labels
    def get_data(self,mode='train'):
        if  not hasattr(self,'input_data') :
            return self.get_items(mode=mode,shuffle=False)
        return self.input_data,self.labels


        
    def normal_fixed_split(self,data,split_idx):
        '''
        使用split_idx 进行分割，可传入两个数字，分别代表val和test的分界值。

        '''
        # assert(self.has_val + self.has_test == len(split_idx))
        if len(split_idx) == 1:
            if self.has_val:
                split_idx=[split_idx[0],data.shape[0]]
            if self.has_test:
                split_idx=[split_idx[0],split_idx[0]]         
        self.train_data = data[:split_idx[0],:]
        self.val_data = data[split_idx[0]:split_idx[1],:]
        self.test_data = data[split_idx[1]:,:]
        self.data = [self.train_data,self.val_data,self.test_data]

        

    def k_fold_cross_validation_split(self,data,k_fold):
        # divided_num = self.k_fold + (1 if self.has_val else 0) + 1 if self.has_test else 0
        divided_num = self.k_fold + self.has_val + self.has_test
        
        self.train_data = []
        self.val_data = []
        self.test_data = []
    



        