import math
import nltk
from nltk import word_tokenize, ngrams
from collections import Counter

vocabs = {
    "czech": "opus.nlpl.eu/OpenSubtitles.cs-en.cs",
    "english": "opus.nlpl.eu/OpenSubtitles.cs-en.en"
}

vocab_choice = input("Select language (czech, english): ").lower()
if vocab_choice not in vocabs:
    print("Invalid language selected.")
    exit(1)

language = vocab_choice
vocab_file = vocabs[vocab_choice]

SN = input("Number of suggestions (default 5): ")
try:
    SN = int(SN)
except ValueError:
    SN = 5
    print(f"Value unspecified, using {SN}...")

nltk.download('punkt')
with open(vocab_file, "r") as cs_file:
    cs = cs_file.read().lower()

# Remove unwanted characters
blacklist_chars = ["'", "ˇ", "-", '"']
for char in blacklist_chars:
    cs = cs.replace(char, "")

# Tokenize the corpus
tokens = word_tokenize(cs, language)

# Train test split
split_point = int(0.8 * len(tokens))
train_tokens = tokens[:split_point]
test_tokens = tokens[split_point:]

# train token counts + train vocab size
train_token_counts = Counter(train_tokens)
V = len(train_token_counts)

#
# build bigram and trigram counts
#

# Unigrams: already in train_token_counts
pass

# Bigrams
train_bigrams_counts = Counter(ngrams(train_tokens, 2))

# Trigrams
train_trigrams_counts = Counter(ngrams(train_tokens, 3))


def compute_unigram_perplexity(test_tokens, train_token_counts, V):
    total_train_tokens = sum(train_token_counts.values())
    log_prob_sum = 0.0
    for token in test_tokens:
        # Laplace smoothing: (count + 1) / (total tokens + V)
        prob = (train_token_counts.get(token, 0) + 1) / (total_train_tokens + V)

        log_prob_sum += math.log(prob)
    avg_log_prob = log_prob_sum / len(test_tokens)
    return math.exp(-avg_log_prob)


def compute_bigram_perplexity(test_tokens, train_token_counts, train_bigrams_counts, V):
    log_prob_sum = 0.0
    count = 0
    for i in range(1, len(test_tokens)):
        w1 = test_tokens[i - 1]
        w2 = test_tokens[i]

        # Use Laplace smoothing: (bigram count + 1) / (unigram count of w1 + V)
        prob = (train_bigrams_counts.get((w1, w2), 0) + 1) / (train_token_counts.get(w1, 0) + V)

        log_prob_sum += math.log(prob)
        count += 1
    avg_log_prob = log_prob_sum / count
    return math.exp(-avg_log_prob)


def compute_trigram_perplexity(test_tokens, train_bigrams_counts, train_trigrams_counts, V):
    log_prob_sum = 0.0
    count = 0
    for i in range(2, len(test_tokens)):
        w1, w2, w3 = test_tokens[i - 2], test_tokens[i - 1], test_tokens[i]
        context_count = train_bigrams_counts.get((w1, w2), 0)

        # Laplace smoothing: (trigram count + 1) / (context count + V)
        prob = (train_trigrams_counts.get((w1, w2, w3), 0) + 1) / (context_count + V)

        log_prob_sum += math.log(prob)
        count += 1
    avg_log_prob = log_prob_sum / count
    return math.exp(-avg_log_prob)


uni_perp = compute_unigram_perplexity(test_tokens, train_token_counts, V)
bi_perp = compute_bigram_perplexity(test_tokens, train_token_counts, train_bigrams_counts, V)
tri_perp = compute_trigram_perplexity(test_tokens, train_bigrams_counts, train_trigrams_counts, V)

print(f"Test Unigram Perplexity: {uni_perp:.4f}")
print(f"Test Bigram Perplexity:  {bi_perp:.4f}")
print(f"Test Trigram Perplexity: {tri_perp:.4f}")
