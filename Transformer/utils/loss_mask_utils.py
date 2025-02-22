import torch.nn as nn
import torch

from typing import Optional

class ListNet_Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.distribution = kwargs.get('distribution', 'polynomial')
        if self.distribution == 'polynomial':
            self.n = kwargs.get('degree', 2)
            
    def forward(self, output : torch.Tensor, y : torch.Tensor, mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        if self.distribution == 'polynomial':
            distribution_of_y = y**self.n / (torch.sum(y**self.n, dim=-1, keepdim=True) + 1e-10)
        elif self.distribution == 'softmax' :
            distribution_of_y = y.float().softmax(dim=-1)
        else:
            raise NotImplementedError
            
        num_of_rates = output.shape[-1]
    
        if num_of_rates != 1:
            softmax_output = output.softmax(dim=-1)
            output = torch.sum(torch.arange(num_of_rates, device = y.device) * softmax_output, dim = -1).unsqueeze(-1)
            
        output = output.squeeze(-1)  # output: [batch_size, num_docs]
    
        if mask is not None:
            # Set logits for padded documents to -inf
            output = output.masked_fill(~mask, -1e9)
    
        # Compute log-softmax over the class dimension
        log_softmax_output = torch.log_softmax(output, dim=-1)
        
        # Calculate the per-document cross-entropy loss
        loss = -torch.sum(distribution_of_y * log_softmax_output, dim=-1)  # Summed loss over classes
        
        # Final batch loss
        batch_loss = torch.mean(loss)  # Mean over all batch elements
    
        return batch_loss

class Cross_Entropy_point(nn.Module):
    def __init__(self, num_of_label:int=5):
        super().__init__()
        self.num_of_labels = num_of_label
        
    def forward(self, output : torch.Tensor, y : torch.Tensor, mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        output = output.contiguous().view(-1,self.num_of_labels)
        y = y.long().view(-1)
        
        ce = nn.CrossEntropyLoss(reduction = 'none')
        
        loss = ce(output, y) * mask.view(-1) if mask != None else ce(output, y) 
        
        return loss.mean()
    
class Combined_Loss(nn.Module):
    def __init__(self, theta:float=0.01, num_of_labels:int=5, **kwargs):
        super().__init__()
        self.theta = theta
        self.pointwise_loss = Cross_Entropy_point(num_of_labels)
        
        dists = kwargs.get('distribution', 'polynomial')
        if dists == 'polynomial':
            n = kwargs.get('degree',2)
        
        self.listwise_loss = ListNet_Loss(distribution = dists, degree = n)
        
    def forward(self, output : torch.Tensor, y : torch.Tensor, mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        
        listwise_loss = self.listwise_loss(output, y, mask)
        pointwise_loss = self.pointwise_loss(output, y, mask)
        
        return pointwise_loss + self.theta * listwise_loss
     
def cross_entropy_for_finetune(output : torch.Tensor, targets : torch.Tensor, mask:Optional[torch.Tensor]=None) -> torch.Tensor:
    outs = output.contiguous().view(-1,5)
    y = targets.contiguous().view(-1,5)
    
    soft_outs = outs.log_softmax(dim = -1)
    mask = (y.sum(-1) != 0)
    
    loss = -torch.sum(soft_outs * y, dim = -1) * mask.view(-1) if mask != None else -torch.sum(soft_outs * y, dim = -1)

    return loss.mean()

def create_mask(inputs:torch.Tensor) -> torch.Tensor:
    """
    Create a mask for padded documents based on feature sums.

    Args:
    - inputs (Tensor): Input tensor of shape [batch_size, num_docs, num_features].

    Returns:
    - mask (Tensor): Boolean mask of shape [batch_size, num_docs].
    """
    return inputs.sum(dim=-1) != 0