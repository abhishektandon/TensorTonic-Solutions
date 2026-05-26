import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    
    dim = K.shape[-1]
    atten = (Q @ torch.transpose(K, 1, 2))/math.sqrt(dim)
    softmax_atten = F.softmax(atten, dim=-1) # across the rows 
    out = softmax_atten @ V

    return out
    