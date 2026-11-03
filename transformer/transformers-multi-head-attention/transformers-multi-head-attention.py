import numpy as np

def multi_head_attention(X_q: np.ndarray,
    X_k: np.ndarray,
    X_v: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
    num_heads: int
  ) -> np.ndarray:
  """Compute multi-head attention for (..., seq_len, d_model) inputs
  Inputs may be unbatched:
        X_q: (query_len, input_dim_q)
        X_k: (key_len, input_dim_k)
        X_v: (key_len, input_dim_v)
  Or batched:
        X_q: (batch_size, query_len, input_dim_q)
        X_k: (batch_size, key_len, input_dim_k)
        X_v: (batch_size, key_len, input_dim_v)
  Weight matrices use NumPy convention:
        W_q: (input_dim_q, num_heads * key_head_dim)
        W_k: (input_dim_k, num_heads * key_head_dim)
        W_v: (input_dim_v, num_heads * value_head_dim)
        W_o: (num_heads * value_head_dim, output_dim)
  """

  unbatched = X_q.ndim == 2
  if unbatched:
    X_q=X_q[None,...]
    X_k=X_k[None,...]
    X_v=X_v[None,...]

  batch_size, query_len, d_model = X_q.shape
  key_len=X_k.shape[1]
  head_dim = d_model//num_heads

  # Project inputs into Q, K, and V.
  Q = X_q @ W_q
  K = X_k @ W_k
  V = X_v @ W_v

  # Split into heads:
  # (batch_size, seq_len, d_model)
  # -> (batch_size, num_heads, seq_len, head_dim)
  Q=Q.reshape(batch_size, query_len, num_heads, head_dim).swapaxes(1,2)
  K=K.reshape(batch_size, query_len, num_heads, head_dim).swapaxes(1,2)
  V=V.reshape(batch_size, query_len, num_heads, head_dim).swapaxes(1,2)

  # Compute attention scores and normalize over key positions.
  scores = (Q @ K.swapaxes(2,3)) / np.sqrt(head_dim)
  scores=scores - np.max(scores, axis=-1, keepdims=True)
  exp_scores = np.exp(scores)
  weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

  # context shape: (batch_size, num_heads, seq_len, head_dim)
  context = weights @ V

  # Concatenate the attention heads
  # (batch_size, num_heads, seq_len, head_dim)
  # -> (batch_size, seq_len, num_heads, head_dim)
  # -> (batch_size, seq_len, embed_dim)
  context = context.swapaxes(1, 2)
  context = context.reshape(batch_size, query_len, d_model)

  # Project to the output dimension.
  output = context @ W_o

  # Remove the batch dimension if the input was unbatched.
  return output[0] if unbatched else output




