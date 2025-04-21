import numpy as np
import gzip
import pickle as pkl


# načtení dat
def load_embeddings(path, limit=None) -> dict[str, np.ndarray]:
    embeddings = {}
    open_fn = gzip.open if path.endswith('.gz') else open

    # read text
    with open_fn(path, 'rt', encoding='utf-8', errors='ignore') as f:
        first = f.readline().strip().split()

        # kontrola, pokud je header (počet slov + dim)
        if len(first) > 2 and all(tok.isdigit() for tok in first[:2]):
            # header, přeskočit
            pass
        else:
            # první řádek skutečný embedding
            word = first[0]
            vec = np.array(list(map(float, first[1:])), dtype=np.float32)
            embeddings[word] = vec
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            parts = line.strip().split()
            if len(parts) <= 2:
                continue
            word = parts[0]
            vec = np.array(list(map(float, parts[1:])), dtype=np.float32)
            embeddings[word] = vec
    return embeddings


# načte překladové páry do list[tuple[str, str]]
def load_bilingual_dict(path) -> list[tuple[str, str]]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            cs, en = line.strip().split()
            pairs.append((cs, en))
    return pairs


# vytvoří seznam [n_samples, dim] pro dva jazyky na základě překladových párů
def build_matrices(pairs: list[tuple[str, str]], src_emb, tgt_emb):
    X_list, Y_list = [], []
    for cs, en in pairs:
        if cs in src_emb and en in tgt_emb:
            X_list.append(src_emb[cs])
            Y_list.append(tgt_emb[en])
    X = np.stack(X_list)
    Y = np.stack(Y_list)
    return X, Y


def to_pickle(a_lang_path, b_lang_path, pairs_path, name="train", save_embeddings=False):
    print("Načítám embeddingy... první jazyk")
    src_emb = load_embeddings(a_lang_path)
    if save_embeddings:
        with open(f'src_emb.pkl', 'wb') as f:
            pkl.dump(src_emb, f)

    print("Načítám embeddingy... druhý jazyk")
    tgt_emb = load_embeddings(b_lang_path)
    if save_embeddings:
        with open(f'tgt_emb.pkl', 'wb') as f:
            pkl.dump(tgt_emb, f)

    print(f"A: {len(src_emb)} slov")
    print(f"B: {len(tgt_emb)} slov")

    print("Načítám překladové páry")
    train_pairs = load_bilingual_dict(pairs_path)
    print(f"Tréninkových párů: {len(train_pairs)}")
    X_train, Y_train = build_matrices(train_pairs, src_emb, tgt_emb)

    print(f"X_train: {X_train.shape}")
    print(f"Y_train: {Y_train.shape}")

    # to pickle
    import pickle
    with open(f'X_{name}.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open(f'Y_{name}.pkl', 'wb') as f:
        pickle.dump(Y_train, f)


if __name__ == "__main__":
    # Cesty k souborům
    cs_vec_path = '../cc.cs.300.vec.gz'
    en_vec_path = '../cc.en.300.vec.gz'
    train_dict_path = '../cs-en.0-5000.txt'
    test_dict_path = '../cs-en.5000-6500.txt'

    to_pickle(cs_vec_path, en_vec_path, train_dict_path, name="train", save_embeddings=True)
    with open(f"test_pairs.pkl", "wb") as f:
        pkl.dump(load_bilingual_dict(test_dict_path), f)
