from collections import Counter

import nltk
from nltk import trigrams
from nltk.util import bigrams

nltk.download('punkt_tab')

cs_file = open("../opus.nlpl.eu/OpenSubtitles.cs-en.cs", "r")
cs = cs_file.read()
cs = cs.lower()
cs_file.close()

tokens_cs = nltk.word_tokenize(cs)
tokens_count = Counter(tokens_cs)

most_frequent = tokens_count.most_common(20)
for word, freq in most_frequent:
    print(f"{word}: {freq}")

# V - vocabulary size
V = len(set(tokens_count))

# Bigrams
bigrams_list = list(bigrams(tokens_cs))
bigrams_count = Counter(bigrams_list)
bigrams_top = bigrams_count.most_common(20)
for bigram, freq in bigrams_top:
    print(f"{bigram}: {freq}")

# Bigram probabilities
bigram_probabilities = {}
for (w1, w2), count in bigrams_count.items():
    N = tokens_count[w1]
    prob = (count + 1) / (N + V)
    bigram_probabilities[(w1, w2)] = prob

for bigram, prob in sorted(bigram_probabilities.items())[:20]:
    print(f"{bigram}: {prob}")

# Trigrams
trigrams_list = list(trigrams(tokens_cs))
trigrams_count = Counter(trigrams_list)
trigrams_top = trigrams_count.most_common(20)
for trigram, freq in trigrams_top:
    print(f"{trigram}: {freq}")

# Trigram probabilities (Laplace smoothing)
trigram_probabilities = {}
for (w1, w2, w3), count in trigrams_count.items():
    N = bigrams_count[(w1, w2)]
    prob = (count + 1) / (N + V)
    trigram_probabilities[(w1, w2, w3)] = prob

for trigram, prob in sorted(trigram_probabilities.items())[:20]:
    print(f"{trigram}: {prob}")
