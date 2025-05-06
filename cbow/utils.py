import numpy as np


def init_matrices(actual_vocab_size, embedding_dim):
    # embeddingy
    # (actual_vocab_size x embedding_dim)
    E = np.random.normal(0, 0.1, (actual_vocab_size, embedding_dim))

    # váhy skryté vrstvy
    # (embedding_dim x actual_vocab_size)
    W_out = np.random.normal(0, 0.1, (embedding_dim, actual_vocab_size))

    # bias vektor výstupní vrstvy
    #  (1 x vocab_size)
    b_out = np.zeros((1, actual_vocab_size))

    return E, W_out, b_out

def softmax(z_scores):
    # z_scores má tvar (1, vocab_size)
    # odečteme maxima pro numerickou stabilitu
    stable_z = z_scores - np.max(z_scores, axis=1, keepdims=True)

    exp_z = np.exp(stable_z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)