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

def preprocess_data(file, num_docs=256, which=0, is_shuffle=True, for_cnn=False, device='cuda'):
    """
    Preprocess data for a transformer or CNN model.

    Args:
        file (str): Path to the data file.
        num_docs (int): Number of documents to pad/truncate to per query.
        is_shuffle (bool): Whether to shuffle the documents within each query.
        for_cnn (bool): Whether to preprocess data specifically for CNN input.

    Returns:
        dict: Preprocessed data with query_id as keys and (data, labels) as values.
    """
    raw_data = pd.DataFrame(Data_From_File.open_file(file)[which]).drop('doc_id', axis=1)
    processed_data = {}

    for query_id, group in raw_data.groupby('query_id'):
        data = np.vstack(group['fl_features'].to_numpy())
        labels = np.array(group['labels'].tolist())[0]
        mean_data, std_data = np.mean(data, axis = 0), np.std(data, axis = 0)
        
        data = (data - mean_data)/(std_data + 1e-6) 
        if data.ndim != 2:
            raise ValueError(f"Inconsistent shape for query_id {query_id}: {data.shape}")

        num_documents, features = data.shape

        if is_shuffle:
            perm = np.random.permutation(num_documents)
            data, labels = data[perm], labels[perm]

        padded_data = np.zeros((num_docs, features), dtype=np.float32)
        padded_labels = np.zeros((num_docs,), dtype=np.int64)

        padded_data[:min(num_documents, num_docs)] = data[:num_docs]
        padded_labels[:min(num_documents, num_docs)] = labels[:num_docs]

        if for_cnn:
            padded_data = np.expand_dims(padded_data, axis=1)
            padded_labels = np.expand_dims(padded_labels, axis=1)

        processed_data[query_id] = (torch.tensor(padded_data).to(device), torch.tensor(padded_labels).to(device))

    return processed_data


class Data_for_transformer_cnn(Dataset):
    def __init__(self, preprocessed_data):
        self.data = preprocessed_data
        self.keys = list(self.data.keys())
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.data[key]
    

