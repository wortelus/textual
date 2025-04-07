import numpy as np

from tf_idf import calculate_tf, calculate_tf_idf, load_docs


def doc_to_vector(tf_idf, vocab_index):
    vec = np.zeros(len(vocab_index))
    for token, value in tf_idf.items():
        idx = vocab_index[token]
        vec[idx] = value
    return vec

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 and norm2 else 0

def main():
    docs = load_docs()
    tf_documents = calculate_tf(docs)
    tf_idf_documents = calculate_tf_idf(tf_documents)
    # tf_idf_documents = tf_documents

    # vytvoření slovníku všech unikátních termů ze všech dokumentů
    # vocab - list unikátních termů
    # vocab_index - mapování term -> index v 'vocab'
    vocab = set()
    for doc in tf_idf_documents:
        vocab.update(doc.keys())
    vocab = list(vocab)
    vocab_index = {token: idx for idx, token in enumerate(vocab)}

    # převod tf-idf slovníků na vektory
    vectors = [doc_to_vector(doc, vocab_index) for doc in tf_idf_documents]

    # Výpočet kosinové podobnosti mezi všemi dvojicemi dokumentů
    n = len(vectors)
    similarities = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = cosine_similarity(vectors[i], vectors[j])
            similarities[i][j] = sim
            similarities[j][i] = sim

    print("Matice kosinových podobností:")
    print("  " + "    ".join(str(i).zfill(3) for i in range(n)))
    for i, row in enumerate(similarities):
        print(i, end=" ")
        print(" ".join(f"{val:.4f}" for val in row))

    # Najdeme dvojici dokumentů s nejvyšší podobností (mimo sám sebe)
    max_sim = -1
    pair = (None, None)
    for i in range(n):
        for j in range(i + 1, n):
            if similarities[i][j] > max_sim:
                max_sim = similarities[i][j]
                pair = (i, j)
    print(f"Nejpodobnější dokumenty jsou {pair[0]} a {pair[1]} s kosinovou podobností {max_sim:.4f}")


if __name__ == "__main__":
    main()