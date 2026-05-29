import numpy as np
import math

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(inp: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """

    x = np.asarray(inp)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)

    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    lyr_norm = gamma * (x - mu) / np.sqrt(var + eps) + beta

    return lyr_norm


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """

    is_2d = (Q.ndim == 2)

    if is_2d:
        Q = Q[None, ...]
        K = K[None, ...]
        V = V[None, ...]

    Qm = Q @ W_q
    Km = K @ W_k
    Vm = V @ W_v

    d_model = Q.shape[-1]
    n = Q.shape[-2]
    d_k = d_model // num_heads

    Qi = np.transpose(Qm.reshape(-1, n, num_heads, d_k), (0, 2, 1, 3))
    Ki = np.transpose(Km.reshape(-1, n, num_heads, d_k), (0, 2, 1, 3))
    Vi = np.transpose(Vm.reshape(-1, n, num_heads, d_k), (0, 2, 1, 3))

    heads = []
    for i in range(num_heads):
        atten = (Qi[:, i, :, :] @ np.transpose(Ki[:, i, :, :], (0, 2, 1))) / math.sqrt(d_k)
        head_i = softmax(atten, axis=-1) @ Vi[:, i, :, :]
        heads.append(head_i)

    out = np.concatenate(heads, axis=-1)
    out = out @ W_o

    if is_2d:
        out = np.squeeze(out, axis=0)

    return out


def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    o1 = x @ W1 + b1
    o1[o1 < 0] = 0
    x = o1 @ W2 + b2
    return x


def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray,
                  W_v: np.ndarray, W_o: np.ndarray, W1: np.ndarray,
                  b1: np.ndarray, W2: np.ndarray, b2: np.ndarray,
                  gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """

    x_hat = layer_norm(x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)
    output = layer_norm(x_hat + feed_forward(x_hat, W1, b1, W2, b2), gamma2, beta2)

    return output