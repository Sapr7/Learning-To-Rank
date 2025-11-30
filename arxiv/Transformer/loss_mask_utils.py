import torch.nn as nn
import torch

def cross_entropy(output, y, mask=None):
    """
    Compute cross-entropy loss for each document, ignoring padded documents.

    Args:
    - output (Tensor): Model's predicted output logits, shape [batch_size, num_docs, num_classes].
    - y (Tensor): True target labels, shape [batch_size, num_docs].
    - mask (Tensor, optional): A boolean mask indicating valid documents (True for valid, False for padded), 
                               shape [batch_size, num_docs].

    Returns:
    - batch_loss (Tensor): Mean loss over the batch, ignoring padded documents if mask is provided.
    """
    output = output.squeeze(-1)  # output: [batch_size, num_docs]

    if mask is not None:
        # Set logits for padded documents to -inf
        output = output.masked_fill(~mask, -1e9)

    # Compute log-softmax over the class dimension
    log_softmax_output = torch.log_softmax(output, dim=-1)
    x_outputs = output / (torch.sum(output, dim = -1, keepdim = True) +1e-10)

    # Calculate the per-document cross-entropy loss
    loss = -torch.sum(y / (torch.sum(y, dim=-1, keepdim=True) + 1e-10) * log_softmax_output, dim=-1)  # Summed loss over classes
    # Final batch loss
    batch_loss = torch.mean(loss)  # Mean over all batch elements

    return batch_loss

def create_mask(inputs):
    """
    Create a mask for padded documents based on feature sums.

    Args:
    - inputs (Tensor): Input tensor of shape [batch_size, num_docs, num_features].

    Returns:
    - mask (Tensor): Boolean mask of shape [batch_size, num_docs].
    """
    return inputs.sum(dim=-1) != 0