import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def main():
    embeddings_E = np.load("word_embeddings_E.npy")
    word_to_idx = np.load("word_to_idx.npy", allow_pickle=True).item()

    words = word_to_idx.keys()
    vectors = embeddings_E

    # PCA
    # pca = PCA(n_components=2)
    # vectors_pca = pca.fit_transform(vectors)
    #
    # x = vectors_pca[:, 0]
    # y = vectors_pca[:, 1]
    #
    # x_margin = (x.max() - x.min()) * 0.1
    # y_margin = (y.max() - y.min()) * 0.1
    #
    # plt.figure(figsize=(100, 80))
    # plt.scatter(x, y)
    #
    # for i, word in enumerate(words):
    #     plt.annotate(word, (x[i] + 0.01, y[i] + 0.01), fontsize=9)
    #
    # plt.xlim(x.min() - x_margin, x.max() + x_margin)
    # plt.ylim(y.min() - y_margin, y.max() + y_margin)
    #
    # plt.title("PCA vizualizace embeddingů")
    # plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=10)
    vectors_tsne = tsne.fit_transform(vectors)

    x = vectors_tsne[:, 0]
    y = vectors_tsne[:, 1]

    x_margin = (x.max() - x.min()) * 0.1
    y_margin = (y.max() - y.min()) * 0.1

    plt.figure(figsize=(120, 100))
    plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, (x[i] + 0.01, y[i] + 0.01), fontsize=9)

    plt.title("t-SNE vizualizace embeddingů")
    plt.show()


if __name__ == "__main__":
    main()