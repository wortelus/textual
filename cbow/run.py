import numpy as np

from cbow.load import load_data, build_vocabulary, generate_training_pairs

CORPUS_DIR = "/home/wortelus/cbow/czech_text_document_corpus_v20/"


def main():
    sentences, word_count = load_data(CORPUS_DIR)
    print(f"loaded {word_count} words from {len(sentences)} sentences")

    vocab = build_vocabulary(sentences, vocab_size=10000)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}
    print(f"vocabulary size: {len(vocab)}, first 10 words: {vocab[:10]}")

    # training pairs
    pairs = generate_training_pairs(sentences, vocab, context_window=2)
    print(f"generated {len(pairs)} training pairs")

    # convert to indices
    indexed_pairs = []
    for context, target in pairs:
        indexed_pairs.append(([word_to_idx[word] for word in context], word_to_idx[target]))
    print(f"first 10 indexed pairs: {indexed_pairs[:10]}")

    # fit
    from cbow.fit import fit
    E, w_out, b_out = fit(vocab, indexed_pairs, num_epochs=5, embedding_dim=100, learning_rate=0.01)

    np.save("word_embeddings_E.npy", E)
    np.save("word_to_idx.npy", word_to_idx)
    np.save("w_out.npy", w_out)
    np.save("b_out.npy", b_out)



if __name__ == "__main__":
    main()
