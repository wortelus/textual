#
# Simple n-gram language model for text generation. Uses trigrams and Laplace smoothing to create the model.
#

import curses
import math
import pickle
from os import makedirs
from os.path import exists
from random import sample

import nltk
from nltk import word_tokenize, trigrams, bigrams
from collections import Counter

#
# User input
#

vocabs = {
    "czech": "opus.nlpl.eu/OpenSubtitles.cs-en.cs",
    "english": "opus.nlpl.eu/OpenSubtitles.cs-en.en"
}

vocab = input("Select language (czech, english): ").lower()
if vocab not in vocabs:
    print("Invalid language selected.")
    exit(1)

language = vocab
vocab = vocabs[vocab]

SN = input("Number of suggestions (default 5): ")
try:
    SN = int(SN)
except ValueError:
    SN = 5
    print(f"Value unspecified, using {SN}...")

#
# Load the vocabulary
#

# Download the tokenizer
nltk.download('punkt')
with open(vocab, "r") as cs_file:
    cs = cs_file.read().lower()

blacklist_chars = ["'", "ˇ", "-", '"']
for char in blacklist_chars:
    cs = cs.replace(char, "")

# Tokenization
makedirs("cache_textgen", exist_ok=True)
if exists("cache_textgen/tokenizer.pkl"):
    print("Loading tokenizer...")
    with open("cache_textgen/tokenizer.pkl", "rb") as pkl_file:
        tokens = pickle.load(pkl_file)
    print("Tokenizer loaded.")
else:
    print("Tokenizing...")
    tokens = word_tokenize(cs, language)
    # create and dump
    with open("cache_textgen/tokenizer.pkl", "wb") as pkl_file:
        pickle.dump(tokens, pkl_file)
    print("Tokenization done.")

# Vocab size
token_counts = Counter(tokens)
V = len(token_counts)

# Bigram counts
bigrams_list = list(bigrams(tokens))
bigrams_counts = Counter(bigrams_list)
bigrams_probs = {}
# Bigram probabilities (Laplace smoothing)
for (w1, w2), count in bigrams_counts.items():
    N = token_counts[w1]
    prob = (count + 1) / (N + V)
    bigrams_probs[(w1, w2)] = prob

bigrams_dict = {}
for (w1, w2), prob in bigrams_probs.items():
    if w1 not in bigrams_dict:
        bigrams_dict[w1] = []
    if len(bigrams_dict[w1]) < 50:
        bigrams_dict[w1].append((w2, prob))

# Trigrams
trigrams_list = list(trigrams(tokens))
trigrams_count = Counter(trigrams_list)

# Trigram probabilities (Laplace smoothing)
trigrams_probs = {}
for (w1, w2, w3), count in trigrams_count.items():
    N = bigrams_counts[(w1, w2)]
    prob = (count + 1) / (N + V)
    trigrams_probs[(w1, w2, w3)] = prob

trigrams_dict = {}
for (w1, w2, w3), prob in trigrams_probs.items():
    if (w1, w2) not in trigrams_dict:
        trigrams_dict[(w1, w2)] = []
    if len(trigrams_dict[(w1, w2)]) < 50:
        trigrams_dict[(w1, w2)].append((w3, prob))

# Sort by probability
for key, values in trigrams_dict.items():
    values.sort(key=lambda x: x[1], reverse=True)

# textgen helpers
n = 1000
common_starting_words = [word for word, freq in token_counts.most_common(n)]

new_word_pool_size = 50


def next_word_pool_size(last_word):
    if (len(last_word) == 1 or
            last_word.isdigit() or
            last_word in common_starting_words):
        return new_word_pool_size
    elif len(last_word) == 2 or not last_word.isalpha():
        return 30
    else:
        return SN


def bigram_suggestions(words):
    # Get the last word and try to generate suggestions
    if words:
        last_word = words[-1]
        suggestions = [suggestion[0] for suggestion in bigrams_dict.get(last_word, [])]
        local_sn = next_word_pool_size(last_word)
        suggestions = suggestions[:local_sn]
    else:
        suggestions = sample(common_starting_words, new_word_pool_size)

    return suggestions


def trigram_suggestions(words):
    # Get the last word and try to generate suggestions
    if len(words) > 1:
        _word = words[-1]
        __word = words[-2]
        last_words = (__word, _word)
        suggestions = [suggestion[0] for suggestion in trigrams_dict.get(last_words, [])]
        local_sn = next_word_pool_size(_word)
        suggestions = suggestions[:local_sn]
    elif words:
        raise ValueError("Not enough words to generate suggestions (trigram), use bigram().")
    else:
        suggestions = sample(common_starting_words, new_word_pool_size)

    return suggestions


def calculate_perplexity(input_text):
    # Tokenize the current input text
    words = word_tokenize(input_text, language)
    if not words:
        return float('inf')
    total_tokens = len(tokens)  # Total tokens in the corpus (for unigram smoothing)
    log_prob_sum = 0.0
    perp_count = 0

    # Calculate probability for the first word using unigram smoothing.
    first_word = words[0]
    p_prob = (token_counts.get(first_word, 0) + 1) / (total_tokens + V)
    log_prob_sum += math.log(p_prob)
    perp_count += 1

    # For subsequent words, use bigram probabilities with Laplace smoothing.
    for i in range(1, len(words)):
        prev = words[i - 1]
        curr = words[i]
        bigram_count = bigrams_counts.get((prev, curr), 0)
        unigram_count = token_counts.get(prev, 0)
        p_prob = (bigram_count + 1) / (unigram_count + V)
        log_prob_sum += math.log(p_prob)
        perp_count += 1

    # Compute average log probability and convert to perplexity.
    avg_log_prob = log_prob_sum / perp_count
    perplexity = math.exp(-avg_log_prob)
    return perplexity


def main():
    line = 0
    max_length = input("Maximum length of the generated text (q to quit): ").lower().strip()
    if max_length == "q":
        print("Exiting...")
    try:
        max_length = int(max_length)
    except ValueError:
        max_length = 50
        print(f"Invalid value, using {max_length}...")

    while True:
        # First starting word
        try:
            chosen_first_word = input("Choose the first word (q to quit): ").lower().strip()
            if not chosen_first_word:
                print("Random first word...")
                text = [sample(common_starting_words, 1)[0]]
            elif chosen_first_word == "q":
                print("Exiting...")
                break
            else:
                if chosen_first_word not in token_counts:
                    raise ValueError("Invalid first word.")
                text = [chosen_first_word]
        except ValueError:
            print("Invalid value, using random first word...")
            text = [sample(common_starting_words, 1)[0]]

        # Generate text
        for i in range(1, max_length):
            if len(text) == 1:
                suggestions = bigram_suggestions(text)
            else:
                suggestions = trigram_suggestions(text)

            if suggestions and len(suggestions) > 0:
                next_word = sample(suggestions, 1)[0]
            else:
                # Add '.' and start a new sentence
                text.append(".")
                next_word = sample(common_starting_words, 1)[0]

            text.append(next_word)

        line += 1
        full_text = " ".join(text)
        print(f"{line} perplexity {calculate_perplexity(full_text)}:")
        print(full_text)
        print("\n")


if __name__ == "__main__":
    main()
