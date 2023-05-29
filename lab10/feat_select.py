import numpy as np
import copy
def feature_select(data):
    features = data[:,:-1]
    labels = data[:,-1]
    return evalute_features(features,labels)

def evalute_features(dataset,labels):
        '''
        通过数据，标签和备选特征，通过计算选出最优的特征
        '''
        best_feature = 0
        best_info_gain_rate = -1
        info_gains = []#统计备选特征的信息增益
        
        
        eps = 1e-8 #浮点数比较精度损失
        features = range(dataset.shape[1])
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
            
            #4.通过每个特征值的索引和labels，可以计算出该特征的信息增益和信息增益率。
            info_gain = get_info_gain(labels,select_feature_idx)
            #5.根据启发式规则，需要从候选属性中挑选信息增益高于平均的才能作为最优属性返回，故统计相关信息
            info_gains.append((feature,info_gain))
        info_gains = sorted(info_gains,key=lambda x: x[1], reverse=True)
        sort_idx = [ x[0] for x in info_gains]
        

        return np.concatenate((dataset[:,sort_idx],labels.reshape(-1,1)),axis=1),sort_idx
def get_info_gain(class_data,feature_ids):
        '''
        通过labels和特征值的索引集合，来计算该特征的信息增益率
        '''
        #1.首先计算当前样本的信息熵ent_class
        class_sum = class_data.shape[0]
        class_data_uni,class_data_count = np.unique(class_data,return_counts=True)
        class_data_count_p = class_data_count / class_sum
        ent_class = -np.sum((class_data_count_p * np.log2(class_data_count_p)))
        
        #2.遍历每个特征值的集合，设置iv=1e-8是防止除0
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
            
        return ent_class