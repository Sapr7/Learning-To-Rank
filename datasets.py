import pickle
from copy import deepcopy

import numpy as np
import pandas as pd

from catboost import Pool

import torch

from torch.utils.data import Dataset, DataLoader

class Data_From_File():
    
    @staticmethod
    def open_file(file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def split(data):
        return pd.DataFrame(data[0]), pd.DataFrame(data[1]), pd.DataFrame(data[2])  
    
    @staticmethod
    def get_data_from_pd(data, need = True) :
    
        X = data['fl_features'].to_numpy()
        y = data['labels'].to_numpy() / 4
        q = []
        if need:
            for i in range(len(data)):
                q.append([data['query_id'][i] for j in range(data['labels'][i].shape[0])])
                
        return X, y, q

    @staticmethod
    def get_doc_query(X,y,q):
        
        qfull = np.concatenate(q)
        yfull = np.concatenate(y)
        Xfull = np.vstack(X)
        
        return Xfull, yfull, qfull

    def __init__(self, file, pool = False, full = True):
        
        self.data_full = self.open_file(file)
        self.data_train, self.data_test, self.data_vali = self.split(self.data_full)
        
        self.get_data_test()
        self.get_data_train()
        self.get_data_vali()
        
        if full:
            self.X_train, self.y_train, self.q_train = self.get_doc_query(self.X_train, self.y_train, self.q_train)
            self.X_test, self.y_test, self.q_test = self.get_doc_query(self.X_test, self.y_test, self.q_test)
            self.X_vali, self.y_vali, self.q_vali = self.get_doc_query(self.X_vali, self.y_vali, self.q_vali)
            
        if pool:
            self.train = Pool(data = self.X_train,
                             label= self.y_train,
                             group_id= self.q_train)
            
            # self.test = Pool(data = self.X_test,
            #                  label= self.y_test,
            #                  group_id= self.q_test)
            
            self.vali = Pool(data = self.X_vali,
                             label= self.y_vali,
                             group_id= self.q_vali)
            
    def get_data_train(self):
        
        self.X_train, self.y_train, self.q_train = self.get_data_from_pd(self.data_train)
        return self.X_train, self.y_train, self.q_train
    
    def get_data_test(self):
        
        self.X_test, self.y_test, self.q_test = self.get_data_from_pd(self.data_test)
        return self.X_test, self.y_test, self.q_test
    
    def get_data_vali(self):
        
        self.X_vali, self.y_vali, self.q_vali = self.get_data_from_pd(self.data_vali)
        return self.X_vali, self.y_vali, self.q_vali
        
        
        
class Data_for_torch_ListNet(Dataset, Data_From_File):
    
    def __init__(self, file, which = 0):
        
        self.data = pd.DataFrame(Data_From_File.open_file(file)[which]).drop('doc_id', axis = 1)
        self.dict_data = self.data.set_index('query_id').T.to_dict('list')
        self.keys = list(self.dict_data.keys())
        
    def __len__(self):
        return len(self.dict_data)
        
    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.dict_data[key]
    
    
class Data_for_transformer_cnn(Dataset, Data_From_File ):
    
    def __init__(self, file, num_docs = 256,  which = 0, is_preprocess = True, is_shuffle = True, for_cnn = False):
        
        self.for_cnn = for_cnn
        self.size = num_docs 
        self.is_shuffle = is_shuffle
        
        self.data = pd.DataFrame(Data_From_File.open_file(file)[which]).drop('doc_id', axis = 1)
        self.dict_data = self.data.set_index('query_id').T.to_dict('list')
        self.keys = list(self.dict_data.keys())
        
        if is_preprocess:
            self.preprocess()
        
    def __len__(self):
        return len(self.dict_data)
        
    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.dict_data[key]
    
    def preprocess(self):
        # print(len(self.dict_data))
        
        for key in self.keys:
            
            data, labels = self.dict_data[key] 
            num_docs, feats = data.shape
            # perm = torch.randperm(torch.arange(data_shape))
            if self.is_shuffle:
                perm = torch.randperm(num_docs)
                data = data[perm]
                labels = labels[perm]
            
            new_data = torch.zeros([self.size, feats], dtype = float)
            new_labels = torch.zeros([self.size], dtype=int)
            
            if num_docs <= self.size:
                new_data[:num_docs] = torch.tensor(data)
                new_labels[:num_docs] = torch.tensor(labels)
                
            else:
                new_data = torch.tensor(data)[:self.size]
                new_labels = torch.tensor(labels)[:self.size]
                
            if self.for_cnn:
                new_data.unsqueeze(1)
                new_labels.unsqueeze(1)
                
                
                
            self.dict_data[key] = [new_data, new_labels]
                
            
            
            
            
            
        
        
        