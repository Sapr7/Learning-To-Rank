import pickle
from copy import deepcopy

import numpy as np
import pandas as pd

from catboost import Pool

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
        