import os
import string
from collections import Counter

import stopwordsiso as stopwords

cz_stopwords = stopwords.stopwords("cs")
interpunctuation = set(string.punctuation)

UNKNOWN_TOKEN = "<UNK>"


def load_data(path: str, word_limit: int = 0) -> (list[list[str]], int):
    sentences = []
    word_count = 0
    # only .lemma files
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.lemma')]
    for file in files:
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            for line in f:
                # split by whitespace
                words = line.strip().split()

                # remove empty strings
                words = [w.lower() for w in words if
                         w.lower() not in cz_stopwords and
                         w not in interpunctuation
                         and w.isalpha()
                         ]
                sentences.append(words)
                word_count += len(words)

            if word_limit and word_count >= word_limit:
                break

    return sentences, word_count


def build_vocabulary(sentences, vocab_size):
    flat_tokens = [word for sentence in sentences for word in sentence]
    word_counts = Counter(flat_tokens)

    vocab = [word for word, _ in word_counts.most_common(vocab_size)]
    vocab.append("<UNK>")

    return vocab


def generate_training_pairs(sentences, vocab, context_window=2):
    training_pairs = []

    # set pro rychlejší vyhledávání
    vocab_set = set(vocab)
    for sentence in sentences:
        for i, target_word in enumerate(sentence):
            if target_word not in vocab_set:
                # pokud slovo není ve slovníku, přidáme
                target_word = UNKNOWN_TOKEN

            # okolní slova
            context = []

            # kontextová slova jsou vlevo a vpravo od cílového slova
            for j in range(i - context_window, i + context_window + 1):
                if j != i and 0 <= j < len(sentence):
                    context_word = sentence[j]
                    if context_word in vocab_set:
                        context.append(context_word)
                    else:
                        # pokud slovo není ve slovníku, přidáme
                        context.append(UNKNOWN_TOKEN)

            training_pairs.append((context, target_word))
    return training_pairs
