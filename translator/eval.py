import os.path
import time

import numpy as np
import pickle as pkl

from annoy import AnnoyIndex

ANNOY_PATH = "annoy.ann"
ANNOY_IDX_WORD_PATH = "annoy_idx_word.pkl"

DIMS = 300

# Normalizovat embeddingy ?
NORMALIZE = True

# n_trees určuje počet náhodných stromů
# Více stromů -> přesnější výsledky, delší čas sestavení, větší index
# Hodnota 10-100 je běžná
N_TREES = 50


def normalize(v):
    if not NORMALIZE:
        return v

    norm = np.linalg.norm(v)
    # zamezení dělení nulou
    if norm == 0:
        return v
    return v / norm


def load_annoy_index(path=ANNOY_PATH, annoy_idx_word_path=ANNOY_IDX_WORD_PATH, dimension=DIMS):
    annoy_idx = AnnoyIndex(dimension, 'angular')
    annoy_idx.load(path)

    # Načíst inverzní index
    with open(annoy_idx_word_path, "rb") as f:
        target_inv_idx = pkl.load(f)

    return annoy_idx, target_inv_idx


def annoy_index(target_emb, path=ANNOY_PATH, annoy_idx_word_path=ANNOY_IDX_WORD_PATH, dimension=DIMS, save=False):
    annoy_idx = AnnoyIndex(dimension, 'angular')
    annoy_idx_word = {}

    print(f"start tvorby ANNOY indexu")
    start_time = time.time()
    for i, (word, vec) in enumerate(target_emb.items()):
        annoy_idx_word[i] = word
        annoy_idx.add_item(i, normalize(vec))
    print(f"index stvořen za {time.time() - start_time:.2f}s")

    print(f"start buildu ANNOY indexu s {N_TREES} stromy...")
    start_time = time.time()
    annoy_idx.build(N_TREES)
    print(f"index build hotov za {time.time() - start_time:.2f}s")

    if save:
        print("Ukládám ANNOY index...")
        annoy_idx.save(path)
        with open(annoy_idx_word_path, "wb") as f:
            pkl.dump(annoy_idx_word, f)
        print("index uložen")

    return annoy_idx, annoy_idx_word


def translate_word(word, source_emb, annoy_idx, target_inv_idx, W, k=5):
    # neexistující slovo
    if word not in source_emb:
        return []

    # slovo -> embedding
    vec_src = source_emb[word]
    # překlad
    vec_tr = vec_src.dot(W.T)

    # normalizace embeddingu
    vec_tr_norm = normalize(vec_tr)

    # top k nejbližších sousedů
    neighbor_indices, distances = annoy_idx.get_nns_by_vector(vec_tr_norm, k, include_distances=True)

    scores = []
    for i in range(len(neighbor_indices)):
        idx = neighbor_indices[i]
        # annoy "angular" distance je ANNOY verze cosine distance
        dist = distances[i]

        # Annoy vzdálenosti na kosinovou podobnost
        cosine_sim = 1 - (dist ** 2 / 2)
        cosine_sim = max(-1.0, min(1.0, cosine_sim))  # Zajištění rozsahu [-1, 1]

        tgt_word = target_inv_idx[idx]
        scores.append((tgt_word, cosine_sim))

    top_k = sorted(scores, key=lambda x: x[1], reverse=True)
    return top_k


def evaluate_accuracy(pairs, source_emb, annoy_idx, annoy_idx_word, W, top_k=(1, 5)) -> dict[int, float]:
    """
    Spočítá přesnost překladu na testovacích datech.
    Vrátí dict: top_k -> accuracy
    """

    total = 0
    correct = {k_val: 0 for k_val in top_k}
    for cs, en in pairs:
        total += 1
        preds = translate_word(cs, source_emb, annoy_idx, annoy_idx_word, W, k=max(top_k))
        pred_words = [w for w, _ in preds]
        for k_val in top_k:
            if en in pred_words[:k_val]:
                correct[k_val] += 1
    return {k_val: correct[k_val] / total for k_val in top_k}


if __name__ == '__main__':
    print("Načítám testovací data...")
    test_pairs = pkl.load(open('test_pairs.pkl', 'rb'))
    src_emb = pkl.load(open('src_emb.pkl', 'rb'))
    tgt_emb = pkl.load(open('tgt_emb.pkl', 'rb'))

    print("Načítám W...")
    W = pkl.load(open('W.pkl', 'rb'))

    if os.path.exists(ANNOY_PATH) and os.path.exists(ANNOY_IDX_WORD_PATH):
        print("cesta k ANNOY indexu existuje, načteme data z disku")
        annoy_index, annoy_idx_word = load_annoy_index()
    else:
        print("cesta k ANNOY indexu neexistuje, vytvoříme nový")
        annoy_index, annoy_idx_word = annoy_index(tgt_emb, save=True)

    print("Vyhodnocuji...")
    acc = evaluate_accuracy(test_pairs, src_emb, annoy_index, annoy_idx_word, W, top_k=np.arange(1, 10))
    for k, k_acc in acc.items():
        print(f"Přesnost Top-{k}: {k_acc * 100:.2f} %")
