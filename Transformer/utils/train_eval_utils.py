import numpy as np
from sklearn.metrics import ndcg_score

from IPython.display import clear_output
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Any

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from utils.loss_mask_utils import create_mask, Cross_Entropy_point

cross_entropy = Cross_Entropy_point()

def expectation_for_eval(output:torch.Tensor) -> torch.Tensor:
    num_of_rates = output.shape[-1] 
    output = output.softmax(dim=-1)
    
    return torch.sum(output * torch.arange(num_of_rates, device = output.device), dim = -1).unsqueeze(-1)

def train_eval(train_loader:DataLoader, model, optimizer:Optimizer, loss_fn:Callable[...,torch.Tensor] = cross_entropy, num_epochs:int = 100, create_mask:Callable[..., torch.Tensor]=create_mask, need_eval=True, **kwargs) -> Tuple[list[Any], dict[str, list[Any]]]:
    
    plot = kwargs.get('plot', True)
    save = kwargs.get('save', True)
    scheduler = kwargs.get('scheduler', None)
    initial_name = kwargs.get('name', 'best_model')
    criterion = kwargs.get('criterion', 'metric')
    
    name = f'{initial_name} with best {criterion}.pth'
    max_score = 0
    min_loss = 1e300
    
    ndcg_5, ndcg_10, ndcg_all = [], [], []
    losses = []
    
    for epoch in range(num_epochs):
        
        model.train()  
        total_loss_epoch = 0
        
        #train model
        for batch in train_loader:
            inputs, targets = batch 

            #forward pass
            inputs = inputs.float()
            mask = create_mask(inputs)
            
            outputs = model(inputs)
            targets = targets.float()
            
            loss = loss_fn(outputs, targets, mask)
            
            #backward pass
            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()      
            
            total_loss_epoch += loss.item()
        
        avg_loss_epoch = total_loss_epoch / len(train_loader)
        
        if need_eval:
            val_loader = kwargs.get('val_loader', None)
            if not isinstance(val_loader, DataLoader):
                raise TypeError('val_loader must be DataLoader!')
            
            score_fn = kwargs.get('score_fn', ndcg_score)
            if score_fn is not ndcg_score:
                raise NotImplemented
            
            avg_ndcg5_epoch, avg_ndcg10_epoch, avg_ndcg_epoch = evaluate(val_loader, model, ndcg_score, create_mask)
            
            ndcg_5.append(avg_ndcg5_epoch)
            ndcg_10.append(avg_ndcg10_epoch)
            ndcg_all.append(avg_ndcg_epoch)
            
        losses.append(avg_loss_epoch)
            
        if scheduler:
            if not isinstance(scheduler, _LRScheduler):
                raise TypeError('scheduler must be scheduler')
            scheduler.step()
        
        if plot:
            if epoch >= 1 :
                if need_eval:
                    plot_results(loss = losses, metrics = {'ndcg@5' : ndcg_5,
                                                            'ndcg@10' : ndcg_10, 
                                                            'ndcg full' : ndcg_all})
                        
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss_epoch:.4f}')
            print(f'NDCG@5 {avg_ndcg5_epoch:.4f} || NDCG@10 {avg_ndcg10_epoch:.4f} || Avg NDCG: {avg_ndcg_epoch:.4f} ')
        
        if save:
            # need save info about model
            
            if (criterion == 'metric') and (not need_eval):
                raise AttributeError('need evaluate to get metric criterion')
            
            if criterion == 'loss':
                if min_loss >= avg_loss_epoch:
                    min_loss = avg_loss_epoch
                    torch.save(model.state_dict(), name)
                    print(f'model saved on {epoch+1} epoch with best {criterion} = {min_loss:.4f} ')
            elif criterion == 'metric':
                if max_score <= avg_ndcg5_epoch:
                    max_score = avg_ndcg5_epoch
                    torch.save(model.state_dict(), name)
                    print(f'model saved on {epoch+1} epoch with best {criterion} = {max_score:.4f} ')
            
    return losses, {'ndcg@5' : ndcg_5,
                    'ndcg@10' : ndcg_10, 
                    'ndcg full' : ndcg_all}

def evaluate(val_loader:DataLoader, model, score_fn:Callable=ndcg_score, create_mask:Callable[..., torch.Tensor]=create_mask, **kwargs) -> list[list[Any]]:
    model.eval()
    
    k = kwargs.get('k', [5, 10, 'full'])
    ndcg_scores = {key : [] for key in k }
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch  
            inputs = inputs.float()
            targets = targets.float()
            
            outputs = model(inputs)
            
            if outputs.shape[-1] != 1:
                outputs = expectation_for_eval(outputs)
            outputs = outputs.squeeze(-1)
            
            #use to not add scores when all features equal to 0
            mask = create_mask(inputs)
              
            for i in range(inputs.size(0)):  
                query_mask = mask[i]  
                query_outputs = outputs[i][query_mask]
                query_targets = targets[i][query_mask]
                if query_targets.numel() > 1:
                    
                    for i,k in enumerate(ndcg_scores.keys()):
                        c = None if k == 'full' else k
                        if query_targets.sum() != 0:
                            ndcg = score_fn(
                                [query_targets.cpu().numpy()],
                                [query_outputs.cpu().numpy()],
                            k = c)
                        else :
                            ndcg = 1
                        ndcg_scores[k].append(ndcg)
                        
        avg_ndcg5_epoch = sum(ndcg_scores[5]) / len(ndcg_scores[5]) if ndcg_scores[5] else 0.0
        avg_ndcg10_epoch = sum(ndcg_scores[10]) / len(ndcg_scores[10]) if ndcg_scores[10] else 0.0
        avg_ndcg_epoch = sum(ndcg_scores['full']) / len(ndcg_scores['full']) if ndcg_scores['full'] else 0.0
        
            
    return [avg_ndcg5_epoch, avg_ndcg10_epoch, avg_ndcg_epoch]

def plot_results(loss:list[Any], metrics:dict[Any,Any],is_metric=True, clear=True) -> None:
    
    if is_metric:
        metrics['loss'] = loss
        
        num = len(metrics)
        ncols = int(np.ceil(np.sqrt(num)))  
        nrows = int(np.ceil(num / ncols))
         
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = np.array(axes).reshape(nrows, ncols)
        
        for idx, (title, values) in enumerate(metrics.items()):
            row, col = divmod(idx, ncols) 
            if title == 'loss':
                axes[row, col].set_yscale('log') 
            axes[row, col].plot(values)
            axes[row, col].set_title(f'${title.upper()}$')
            axes[row, col].set_xlabel('num of epoch')
            axes[row, col].set_ylabel(f'{title}')
            axes[row, col].grid()

        for idx in range(num, nrows * ncols):
            row, col = divmod(idx, ncols)
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        if clear:
            clear_output()  
            plt.show()
        else:
            plt.show()
    else:
        plt.plot(loss)
        plt.set_title('Loss during train')
        plt.yscale('log')
        plt.xlabel('num of epoch')
        plt.ylabel('loss')
        plt.grid()
        if clear:
            clear_output()
            plt.show()
        else:
            plt.show()
    
    # if is_metric:
        
    #     fig, ax = plt.subplots(1,4, figsize=(20,6))
        
    #     ax[0].set_yscale('log')
    #     ax[0].plot(np.arange(len(loss)), loss)
    #     ax[0].set_xlabel('number of epoch')
    #     ax[0].set_ylabel('train loss')
    #     ax[0].set_title('train loss')
    #     ax[0].grid(True)
        
    #     for i,j in enumerate(metrics.keys()):
    #         ax[i+1].plot(np.arange(len(metrics[j])), metrics[j])
    #         ax[i+1].set_xlabel('number of epoch')
    #         ax[i+1].set_ylabel(f'{j}')
    #         ax[i+1].set_title(f'{j}')
    #         ax[i+1].grid(True)
    #     if clear:
    #         clear_output()
    #         plt.show()
    #     else:
    #         plt.show()            
             
    # else:
    #     plt.set_yscale('log')
    #     plt.plot(loss)
    #     plt.title('train loss')
    #     plt.grid()
    #     plt.xlabel('number of epoch')
    #     plt.xlabel('train loss')
    #     plt.show()
     
        

    