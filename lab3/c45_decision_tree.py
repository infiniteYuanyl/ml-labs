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
class C45DecisionTree(BaseModel):
    def __init__(self,dataloader=None,feature_names={},label_names={},load_checkpoint=None):
        super().__init__(dataloader)
        self.feature_names = feature_names
        self.label_names =  label_names
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
        return Errloss(predict,label)

    def forward(self,input):     
        
        return None
    def backward(self, diff):
        input,labels = self.dataloader.get_data()
        self.tree_root = self.create_decision_tree(input,labels,list(range(0,input.shape[1])))
        print('tree: ',self.tree_root)
        
    def predict(self,input,labels):
        
        return 

    def evaluate(self,input,labels):
        
        visualize_decision_tree(self.tree_root)
    def save_checkpoint(self,**kwargs):
        d= datetime.today().strftime('%Y%m%d_%H_%M_%S')
        d.split(' ')
        np.savez('checkpoints/'+d+'_tree_root.npz',Tree=dict(self.tree_root),allow_pickle=True)
        print('checkpoint has been saved in %s!'%('checkpoints/'+d+'_tree_root.npz'))

    def seach_decision_tree(self,input):
        pass
    def create_decision_tree(self,dataset,labels,feature_ids):
        cur_classes = np.unique(labels)
        cur_classes_num = cur_classes.shape[0]
        cur_node = {}
        #1.如果当前剩下的标签数量只有1个，那么返回这个标签
        if cur_classes_num == 1:
            cur_node=self.label_names[int(labels[0])]
            return cur_node    
        repeat = True
        item = copy.deepcopy(dataset[0,feature_ids])
        for item_ in dataset[:,feature_ids]:
            if not (item == item_).all():
                repeat=False
                break
        
        #2.如果当前划分的feature_id为空，或者feature_id下的数据集，每个备选特征取值都一样，选择剩余标签中类最多的那个划分。
        if len(feature_ids) == 0 or repeat:
            print('cut~          ')
            cur_node = self.label_names[np.argmax(np.bincount(labels.astype(np.int32)))]
            return cur_node
        
        #3.选出信息增益率最高的特征类别
        best_feature = self.get_best_feature(dataset,labels,feature_ids)
        cur_feature_dataset = dataset[:,best_feature]
        cur_feature = np.unique(cur_feature_dataset)     

        #4.当前节点特征确定，作为字典的key；再选取当前节点特征的取值，取值的key和下一棵树上的元素建立映射  
        cur_node_key =  list(self.feature_names[best_feature].keys())[0]    
        cur_node_value = {}    
        best_feature_sum = len(list(self.feature_names[best_feature].values())[0])
        
        for sub_feature in range(1,best_feature_sum+1):
            if sub_feature in cur_feature:
                #4.1如果对于每个特征值，存在于样本的特征值中
                select_item_id = np.argwhere(dataset[:,best_feature] == sub_feature).flatten()  
                next_dataset = dataset[select_item_id,:] 
                next_labels = labels[select_item_id]
                #去除这个特征
                next_feature_ids = copy.deepcopy(feature_ids)
                next_feature_ids.remove(best_feature)
               
                cur_node_value.update({list(self.feature_names[best_feature].values())[0][sub_feature]:\
                    self.create_decision_tree(next_dataset,next_labels,next_feature_ids)})
            else:
                #4.2如果为不存在，即为空，则标记当前样本数量最多类的为该作为该特征值的终结节点
                cur_node_value.update({list(self.feature_names[best_feature].values())[0][sub_feature]\
                     : self.label_names[np.argmax(np.bincount(labels.astype(np.int32)))]})

        cur_node = {cur_node_key:cur_node_value}
        return cur_node      
    def get_info_gain_rate(self,class_data,feature_ids):
        '''
        通过labels和特征值的索引集合，来计算该特征的信息增益率
        '''
        #1.首先计算当前样本的信息熵ent_class
        class_sum = class_data.shape[0]
        class_data_uni,class_data_count = np.unique(class_data,return_counts=True)
        class_data_count_p = class_data_count / class_sum
        ent_class = -np.sum((class_data_count_p * np.log2(class_data_count_p)))
        
        #2.遍历每个特征值的集合，设置iv=-1e-6是防止除0
        iv = 1e-8
        for feature_id in feature_ids:
            
            #2.0.feature_id_sum，一个特征值在特征出现的次数。
            #  class_feture_sum, 标签中，该特征值的索引集合。理论上=feature_id_sum
            feature_id_sum = feature_id.shape[0]
            class_by_feature = class_data[feature_id]
            class_by_feature_sum = class_by_feature.shape[0]

            #2.1.获取每个特征值下，不同类出现的概率class_by_feature_count_p 
            class_by_feature_uni,class_by_feature_count = np.unique(class_by_feature,return_counts=True)   
            class_by_feature_count_p = class_by_feature_count / class_by_feature_sum
            ent_class -= ( feature_id_sum / class_sum *np.sum( (- class_by_feature_count_p * np.log2(class_by_feature_count_p))))   

            #2.2.获取这个特征值出现的概率
            feature_id_p = feature_id_sum / class_sum
            iv +=   (- feature_id_p * np.log2(feature_id_p))
        return ent_class / iv
    def get_best_feature(self,dataset,labels,features):
        best_feature = 0
        best_info_gain_rate = 100
        for feature in features:
            #用当前特征值，把labels划分开来。
            #1.首先，先取出当前特征的列。
            dataset_cut_by_feature = dataset[:,feature]

            #2.取出后，排序并且去重，得到在样本中，该特征所有的出现特征值
            dataset_cut_by_feature_copy = copy.deepcopy(dataset_cut_by_feature)
            np.sort(dataset_cut_by_feature_copy)
            feature_classes = np.unique(dataset_cut_by_feature_copy)

            #3.对于每一个特征值，划分开来，返回每个特征值对应原label的索引集合。
            select_feature_idx = []
            for feature_cls in feature_classes:
                select_feature_idx.append(np.argwhere(dataset_cut_by_feature == feature_cls).flatten())
            
            #4.通过每个特征值的索引和labels，可以计算出该特征的信息增益率。
            info_gain_rate = self.get_info_gain_rate(labels,select_feature_idx)
        
            #5.寻找最小信息增益率的特征值
            if info_gain_rate < best_info_gain_rate:
                best_info_gain_rate = info_gain_rate
                best_feature = feature
        
        return best_feature



	




   



