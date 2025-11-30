import pickle

import numpy as np
import pandas as pd

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
    
class Dataset_for_transformer(Dataset):
    def __init__(self, preprocessed_data):
        self.data = preprocessed_data
        self.keys = list(self.data.keys())
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.data[key]

class Dataset_for_finetune(Dataset):
    def __init__(self, list_of_data):
        self.data = list_of_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def preprocess_data(file:str, num_docs:int=256, which:int=None, is_shuffle:bool=True, device:str='cuda') -> dict:
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
    if which != None:
        raw_data = pd.DataFrame(Data_From_File.open_file(file)[which]).drop('doc_id', axis=1)
    else:
        raw_data = pd.DataFrame(Data_From_File.open_file(file)).drop('doc_id', axis=1)

    processed_data = {}
        
    for query_id, group in raw_data.groupby('query_id'):
        data = np.clip(np.vstack(group['fl_features'].to_numpy()).astype(np.float64), -1e15,1e100)
        labels = np.array(group['labels'].tolist())[0]
        
        mean_data, std_data = np.clip(np.mean(data, axis = 0).astype(np.float64), -1e15, 1e100), np.clip(np.std(data, axis = 0).astype(np.float64), 0.0, 1e100)    
        data = (data.astype(np.float64) - mean_data)/(std_data + 1e-6) 
        
        if data.ndim != 2:
            raise ValueError(f"Inconsistent shape for query_id {query_id}: {data.shape}")

        num_documents, features = data.shape

        if is_shuffle:
            perm = np.random.permutation(len(labels))
            data, labels = data[perm], labels[perm]

        padded_data = np.zeros((num_docs, features), dtype=np.float32)
        padded_labels = np.zeros((num_docs,), dtype=np.int64)

        padded_data[:min(num_documents, num_docs)] = data[:num_docs]
        padded_labels[:min(num_documents, num_docs)] = labels[:num_docs]
        
        processed_data[query_id] = (torch.tensor(padded_data).to(device), torch.tensor(padded_labels).to(device))
    
    print('preprocess is done')
    return processed_data

def preprocess_for_finetune(train_loader:DataLoader, model, p:float=0.7) -> list:
    model.eval()
    q = 1-p
    
    new_datas = []
    
    for batch in train_loader:
        x,y = batch
        
        outs = model(x.float())
        y = y.squeeze(0)
        x = x.squeeze(0)
        
        mask1 = (x.sum(-1)!=0)        
        o = torch.argmax(outs.squeeze(0), dim = -1) * mask1
        
        y_dist = torch.zeros(y.shape[0], 5, device = 'cuda')
        idxs_1 = torch.arange(y.shape[0], device = 'cuda')
        
        y_dist[idxs_1, y] = p

        mask2 = (torch.abs(o - y) == 1)
        idxs = o * mask2 + ~mask2 * (y - torch.sign(y-o))

        y_dist[idxs_1, idxs] += q 
        y_dist[idxs_1,0] *= mask1
        
        new_datas.append([x.to('cuda'), y_dist.to('cuda')])
        
    return new_datas
        