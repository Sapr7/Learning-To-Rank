import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, size, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + self.dropout(sublayer_output))
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden, dropout_rate):
        super().__init__()
        
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)
        self.attention_residual = ResidualBlock(d_model, dropout_rate)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )
        self.ffn_residual = ResidualBlock(d_model, dropout_rate)
        
    def forward(self, x, attention_mask=None, padding_mask=None):
        # Multi-head attention
        attn_output, _ = self.multihead_attention(
            query=x, key=x, value=x, attn_mask=attention_mask, key_padding_mask=padding_mask
        )
        x = self.attention_residual(x, attn_output)

        # Feed-forward
        ffn_output = self.feed_forward(x)
        x = self.ffn_residual(x, ffn_output)
        return x
         
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, ffn_hidden, input_dim, dropout_rate, output_dim = 5):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model, bias=True)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_hidden, dropout_rate)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)

    def create_padding_matrix(self, input_tensor):
        batch_size, seq_length, _ = input_tensor.size()
        row_sums = input_tensor.sum(dim=-1).to(input_tensor.device)
        padding_mask = (row_sums != 0).float().to(input_tensor.device)
        return padding_mask

    def forward(self, x):
        # Create masks
        padding_mask = self.create_padding_matrix(x)
        
        # Input projection
        x = self.input_projection(x)
        
        # Permute for compatibility with nn.MultiheadAttention
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        
        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        
        # Compute scores for each item
        output = self.output_layer(x)  # (seq_len, batch_size, 1)
                                                                                                    
        return output.permute(1, 0, 2)  # (batch_size, seq_len, 1)

def make_Encoder_model(d_model:int=512, n_heads:int=4, n_layers:int=2, ffn_hidden:int=512, input_dim:int=136, output_dim:int=5, dropout_rate:float=0.3, device:str='cuda') -> Encoder:

    return Encoder(d_model=d_model, 
                n_heads=n_heads, 
                n_layers=n_layers, 
                ffn_hidden=ffn_hidden, 
                input_dim=input_dim, 
                dropout_rate = dropout_rate,
                output_dim=output_dim
                ).to(device)

