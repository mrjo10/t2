import torch
import torch.nn.functional as F

# Example sentence
sentence = ["What", "are", "the", "symptoms", "of", "diabetes", "?"]
vocab_size = 10000  # Assume a vocabulary size

# Simulate token embeddings (each word as a 64-dim vector)
embedding_dim = 64
tokens = torch.randint(0, vocab_size, (len(sentence),))  # Token IDs
embeddings = torch.randn(len(sentence), embedding_dim)  # Random embeddings

# Compute Query, Key, and Value matrices
W_q = torch.randn(embedding_dim, embedding_dim)  # Query weights
W_k = torch.randn(embedding_dim, embedding_dim)  # Key weights
W_v = torch.randn(embedding_dim, embedding_dim)  # Value weights

Q = embeddings @ W_q
K = embeddings @ W_k
V = embeddings @ W_v

# Compute attention scores
d_k = embedding_dim ** 0.5
attention_scores = (Q @ K.T) / d_k  # Scaled dot-product
attention_weights = F.softmax(attention_scores, dim=-1)  # Apply softmax

# Compute final attention output
output = attention_weights @ V

# Print attention weights
print("Attention Weights:\n", attention_weights)

