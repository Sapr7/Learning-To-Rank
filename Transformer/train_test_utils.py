import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ndcg_score

from IPython.display import clear_output

from loss_mask_utils import *





def train_eval(train_loader, val_loader, model, optimizer, scheduler, loss_fn = cross_entropy,  num_epochs = 100, create_mask = create_mask, plot = True, save = True, name = 'encoder_model.pth'):
    
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

            loss = loss_fn(outputs, targets, mask)
            
            #backward pass
            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()      
            
            total_loss_epoch += loss.item()
        
        avg_loss_epoch = total_loss_epoch / len(train_loader)
        

        model.eval()
        
        #test model
        val_loss_epoch = 0
        ndcg_scores = {5 : [],
                       10 : [],
                       'full' : []}

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch  
                inputs = inputs.float()

                outputs = model(inputs) 

                #use to not add scores when all features equal to 0
                mask = inputs.sum(dim=2) != 0  

                valid_outputs = outputs[mask] 
                valid_targets = targets[mask] 
                
                loss = loss_fn(valid_outputs, valid_targets)
                
                val_loss_epoch += loss.item()

                for i in range(inputs.size(0)):  
                    query_mask = mask[i]  
                    query_outputs = outputs[i][query_mask].squeeze()
                    query_targets = targets[i][query_mask]

                    if query_targets.numel() > 1:
                        
                        for i,k in enumerate(ndcg_scores.keys()):
                            c = None if k == 'full' else k
                            ndcg = ndcg_score(
                                [query_targets.cpu().numpy()],
                                [query_outputs.cpu().numpy()],
                            k = c)
                            ndcg_scores[k].append(ndcg)
                        

            avg_val_loss = val_loss_epoch / len(val_loader)
            
            avg_ndcg_epoch = sum(ndcg_scores['full']) / len(ndcg_scores['full']) if ndcg_scores['full'] else 0.0
            avg_ndcg5_epoch = sum(ndcg_scores[5]) / len(ndcg_scores[5]) if ndcg_scores[5] else 0.0
            avg_ndcg10_epoch = sum(ndcg_scores[10]) / len(ndcg_scores[10]) if ndcg_scores[10] else 0.0
            
            #scheduler step???
            if epoch < 30:
                scheduler.step()
            elif epoch >= 30 and epoch < 100 and epoch % 2 == 0:
                scheduler.step()
            elif epoch % 4 == 0 :
                scheduler.step()
            
            losses.append(avg_loss_epoch)
            ndcg_all.append(avg_ndcg_epoch)
            ndcg_5.append(avg_ndcg5_epoch)
            ndcg_10.append(avg_ndcg10_epoch)
            
            if plot:
                if epoch >= 1 :
                    plot_results(loss = losses, metrics = {'ndcg@5' : ndcg_5,
                                                            'ndcg@10' : ndcg_10, 
                                                            'ndcg full' : ndcg_all})
                    
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss_epoch:.4f}')
                print(f'Validation Loss: {avg_val_loss:.4f} || NDCG@5 {avg_ndcg5_epoch:.4f} || NDCG@10 {avg_ndcg10_epoch:.4f} || Avg NDCG: {avg_ndcg_epoch:.4f} ')
    if save:
        torch.save(model.state_dict(), name)
                
            
    return losses, {'ndcg@5' : ndcg_5,
                    'ndcg@10' : ndcg_10, 
                    'ndcg full' : ndcg_all}

def plot_results(loss, metrics, is_metric = True, clear = True):
     
    if is_metric:
        fig, ax = plt.subplots(1,4, figsize = (20, 6),constrained_layout=True)
        
        ax[0].set_yscale('log')
        ax[0].plot(np.arange(len(loss)), loss)
        ax[0].set_xlabel('number of epoch')
        ax[0].set_ylabel('train loss')
        ax[0].set_title('train loss')
        ax[0].grid(True)
        
        for i,j in enumerate(metrics.keys()):
            ax[i+1].plot(np.arange(len(metrics[j])), metrics[j])
            ax[i+1].set_xlabel('number of epoch')
            ax[i+1].set_ylabel(f'{j}')
            ax[i+1].set_title(f'{j}')
            ax[i+1].grid(True)
        if clear:
            clear_output()
            plt.show()
        else:
            plt.show()            
             
    else:
        plt.set_yscale('log')
        plt.plot(loss)
        plt.title('train loss')
        plt.grid()
        plt.xlabel('number of epoch')
        plt.xlabel('train loss')
        plt.show()
        
    
    
    
    
    
    
    
        
    #     fig, ax = plt.subplots(1,4,figsize = (20,6), constrained_layout=True)
        
    #     ax[0].set_yscale('log')
    #     ax[0].plot(np.arange(epoch+1), losses)
    #     ax[0].set_xlabel('epochs')
    #     ax[0].set_ylabel('loss')
    #     ax[0].set_title('train loss')
    #     ax[0].grid(True)
        
    #     ax[1].plot(np.arange(epoch+1), n5)
    #     ax[1].set_xlabel('epochs')
    #     ax[1].set_ylabel('ndcg@5')
    #     ax[1].set_title('ndcg@5')
    #     ax[1].grid(True)
        
    #     ax[2].plot(np.arange(epoch+1), n10)
    #     ax[2].set_xlabel('epochs')
    #     ax[2].set_ylabel('ndcg@10')
    #     ax[2].set_title('ndcg@10')
    #     ax[2].grid(True)
        
    #     ax[3].plot(np.arange(epoch+1), nall)
    #     ax[3].set_xlabel('epochs')
    #     ax[3].set_ylabel('ndcg')
    #     ax[3].set_title('ndcg')
    #     ax[3].grid(True)
    #     clear_output()
    #     plt.show()
    #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    #     print(f'Validation Loss: {avg_val_loss:.4f}, Avg NDCG: {avg_ndcg:.4f} || NDCG@5 {top5ndcg} || NDCG@10 {top10ndcg}')

    # torch.save(model.state_dict(), 'encoder_model.pth')