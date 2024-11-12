from itertools import combinations, permutations

import numpy as np
import torch

import math

class Permutation:
    
    def __len__(self):
        return math.factorial(self.n)
    
    @staticmethod
    def get_p_for_perm(perm, scores, phi):
        
        device = scores.device
        
        k = perm.shape[0]
        all_indices = torch.arange(scores.size(0)).to(device)
        mask = ~torch.isin(all_indices, perm).to(device)
        
        permuted_scores = scores[perm]
        other_scores = scores[mask]
        
        all_scores = torch.cat((permuted_scores, other_scores)).to(device)
        
        phi_score = phi(all_scores).to(device)
        
        for i in range(phi_score.shape[0]):
                phi_score[i] /= torch.sum(phi_score[i:])
        
        return torch.prod(phi_score[:k])
    
    @staticmethod
    def all_perms(idxs, as_list=True):
        return list(permutations(idxs)) if as_list else permutations(idxs)

    
    def __init__(self, n, scores, phi, device):
        
        self.device = device
        self.n = n
        self.scores = scores
        self.phi = phi
    
    def random_permutation(self):
        return torch.randperm(self.n)
    
    def get_all_probas(self):
        perms = self.all_perms(self.idxs, as_list=False)
        return torch.tensor([self.get_p_for_perm(perm, self.scores, self.phi) for perm in perms]).to(self.device)
    
    def top_k_perms(self, fix_idxs=None, full=True):
        if fix_idxs is None:
            fix_idxs = []
        
        idxs = [i for i in self.idxs if i not in fix_idxs]
        top_k_permutations = permutations(idxs) if not full else list(permutations(idxs))

        return [fix_idxs + list(perm) for perm in top_k_permutations]
    
class Gk(Permutation):
    
    @property
    def Gk_probas(self):
        return self.top_k_probas
    
    @property
    def all_Gk_perms(self):
        return self.all_perms_of_first_docs
    
    def __len__(self):
        
        # if self.k == self.n:
        #     return super().__len__()
        
        return self.number_of_perms 

    def __init__(self, k, n, scores, phi, device):
        super().__init__(n, scores, phi, device)
        
        self.k = k
        
        self.get_proba_for_topk_docs()
        
    def get_all_topk_perms(self):
        self.number_of_perms = torch.prod(torch.tensor([self.n - i for i in range(self.k)])).to(self.device)
        
        all_first_docs = list(combinations(np.arange(self.n), self.k))
        all_perms_of_first_docs = torch.tensor([list(permutations(list(_))) for _ in all_first_docs]).to(self.device).reshape([self.number_of_perms, self.k]) 
        
        return all_perms_of_first_docs
    
    def get_proba_for_topk_docs(self):
        
        # if self.k == self.n:
        #     self.top_k_probas = super().get_all_probas ()
        #     return self.top_k_probas
        
        self.all_perms_of_first_docs = self.get_all_topk_perms()
        
        self.top_k_probas = torch.zeros(self.number_of_perms).to(self.device)
        self.perm_for_proba = torch.zeros([self.number_of_perms, self.k])
        
        for i, perm in enumerate(self.all_perms_of_first_docs):
            self.top_k_probas[i] = super().get_p_for_perm(perm, self.scores, self.phi)
            
    def get_probas_for_perm(self):
        return zip(self.Gk_probas, self.all_Gk_perms)
