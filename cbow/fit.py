import time

import numpy as np

from cbow.utils import init_matrices, softmax


def fit(vocab: list,
        indexed_pairs: list,
        num_epochs: int = 10,
        embedding_dim: int = 100,
        learning_rate: float = 0.01):
    E, W, b = init_matrices(len(vocab), embedding_dim)

    print(f"Trénink CBOW modelu s {len(vocab)} slovy a embedding dimenzí {embedding_dim}.")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_epoch_loss = 0

        # Náhodné promíchání tréninkových dat pro každou epochu
        np.random.shuffle(indexed_pairs)

        for i, (context_idxs, target_idx) in enumerate(indexed_pairs):
            if not context_idxs:
                continue

            # vypočteme průměr kontextových slov
            context_embeddings = E[context_idxs, :]
            v_context = np.mean(context_embeddings, axis=0, keepdims=True)

            # skóre pro všechna slova ve slovníku na základě průměrného kontextového vektoru 'h'
            # (1, vocab_size)
            z_scores = np.dot(v_context, W) + b
            # softmax
            # (1, vocab_size)
            y_predicted_probs = softmax(z_scores)

            # cross-entropy loss
            # epsilon pro numerickou stabilitu
            loss = -np.log(y_predicted_probs[0, target_idx] + 1e-9)
            total_epoch_loss += loss

            # one-hot encoding pro cílové slovo
            # (1, vocab_size)
            y_true_one_hot = np.zeros_like(y_predicted_probs)
            y_true_one_hot[0, target_idx] = 1

            #
            # update gradientu pro W a b
            #

            # základní gradientní signál
            # (1, vocab_size)
            grad_z = y_predicted_probs - y_true_one_hot

            # gradient ztráty vzhledem k matici W
            # h.T: (embedding_dim, 1) @ grad_z: (1, vocab_size)
            # (embedding_dim, vocab_size)
            grad_W_out = np.dot(v_context.T, grad_z)

            # gradient ztráty vzhledem k vektoru bias
            # (1, vocab_size)
            grad_b_out = grad_z

            # aktualizujeme matici W a bias
            W -= learning_rate * grad_W_out
            b -= learning_rate * grad_b_out

            #
            # update gradientu pro E
            #

            # gradient ztráty vzhledem k průměrnému kontextovému vektoru h (dh)
            # grad_z: (1, vocab_size) @ W_out.T: (vocab_size, embedding_dim)
            # (1, embedding_dim)
            grad_h = np.dot(grad_z, W.T)

            # distribuce gradientu dh na embeddingy kontextových slov (dE)
            # prakticky se provede jen normalizace grad_h vůči počtu kontextových slov
            # (1, embedding_dim)
            grad_E_for_context_word = grad_h / len(context_idxs)

            # postupně aktualizujeme embeddingy kontextových slov
            for word_idx_in_context in context_idxs:
                E[word_idx_in_context, :] -= (
                        learning_rate * grad_E_for_context_word.squeeze())  # .squeeze() odstraní (1,) dimenzi

            # verbose output každých cca 100 kroků (krok je 1 sentence)
            if (i + 1) % (max(1, len(indexed_pairs) // 100)) == 0:  # Cca 100 výpisů za epochu
                current_avg_loss = total_epoch_loss / (i + 1)
                print(
                    f"Epocha {epoch + 1}/{num_epochs}, Krok {i + 1}/{len(indexed_pairs)}, Prům. ztráta (batch): {current_avg_loss:.4f}")

        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = total_epoch_loss / len(indexed_pairs)
        print(f"Epocha {epoch + 1} dokončena za {epoch_duration:.2f}s. Průměrná ztráta: {avg_epoch_loss:.4f}")

    print("Trénink dokončen.")
    return E, W, b
