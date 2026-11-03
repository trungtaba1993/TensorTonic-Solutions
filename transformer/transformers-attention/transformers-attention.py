import torch
import math
import torch.nn.functional as F

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    scores= (Q @ K.transpose(-2,-1))/math.sqrt(Q.size(-1))
    weights = F.softmax(scores, dim=-1)
    return weights @ V