import torch
import torch.nn as nn


class ListNet(nn.Module):
    def __init__(self,inp_size = 136, out_size = 1 ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inp_size, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,out_size),
            
            )
        
        # self.razdel = torch.tensor(['_' for i in range(inp_size)])
        
    def forward(self, X):
        output = self.model(X).squeeze()
        return output