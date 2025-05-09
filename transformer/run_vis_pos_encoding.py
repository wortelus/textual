import math

import matplotlib.pyplot as plt
import torch


def visualize_positional_encoding(seq_len=50, embed_dim=128):
    pe = torch.zeros(seq_len, embed_dim)
    for pos in range(seq_len):
        for i in range(0, embed_dim, 2):
            pe[pos, i] = math.sin(pos / (10000 ** (2 * i / embed_dim)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / embed_dim)))
    plt.figure(figsize=(10, 6))
    plt.imshow(pe.numpy())
    plt.colorbar()
    plt.title("Positional Encoding Matrix")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.show()

visualize_positional_encoding()