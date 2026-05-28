import numpy as np
import math 

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    
    d_model = Q.shape[-1]
    n = Q.shape[-2]
    d_k = d_model // num_heads    
    
    Qm = (Q @ W_q).reshape(-1, n, num_heads, d_k)
    Km = (K @ W_k).reshape(-1, n, num_heads, d_k)
    Vm = (V @ W_v).reshape(-1, n, num_heads, d_k)
    
    Qi = np.transpose(Qm, (0, 2, 1, 3))
    Ki = np.transpose(Km, (0, 2, 1, 3))
    Vi = np.transpose(Vm, (0, 2, 1, 3))

    heads = []
    for i in range(num_heads):
        atten = Qi[:, i, :, :] @ np.transpose(Ki[:, i, :, :], (0, 2, 1))
        head_i = softmax(atten/math.sqrt(d_k), axis=-1) @ Vi[:, i, :, :]
        heads.append(head_i)

    out = np.concatenate(heads, axis=-1)
    out = out @ W_o
    return out