#
# Simple n-gram language model for autocompletion. Uses bigrams and Laplace smoothing to create the model
# and curses to create a simple terminal interface for autocompletion.
#

import curses
import math
import pickle
from os import makedirs
from os.path import exists

import nltk
from nltk import word_tokenize, bigrams
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

# Bigram
token_counts = Counter(tokens)
# Vocab size
V = len(token_counts)

# Bigram counts
bigrams_list = list(bigrams(tokens))
bigram_counts = Counter(bigrams_list)
bigrams_probs = {}
for (w1, w2), count in bigram_counts.items():
    N = token_counts[w1]
    prob = (count + 1) / (N + V)
    bigrams_probs[(w1, w2)] = prob

bigrams_dict = {}
for (w1, w2), prob in bigrams_probs.items():
    if w1 not in bigrams_dict:
        bigrams_dict[w1] = []
    if len(bigrams_dict[w1]) < SN:
        bigrams_dict[w1].append((w2, prob))

# Sort by probability
for key, values in bigrams_dict.items():
    values.sort(key=lambda x: x[1], reverse=True)


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
        prev = words[i-1]
        curr = words[i]
        bigram_count = bigram_counts.get((prev, curr), 0)
        unigram_count = token_counts.get(prev, 0)
        p_prob = (bigram_count + 1) / (unigram_count + V)
        log_prob_sum += math.log(p_prob)
        perp_count += 1

    # Compute average log probability and convert to perplexity.
    avg_log_prob = log_prob_sum / perp_count
    perplexity = math.exp(-avg_log_prob)
    return perplexity

def main(stdscr):
    # Show the cursor
    curses.curs_set(1)
    # Enable keypad mode
    stdscr.keypad(True)
    # Disable auto-echoing of keys
    curses.noecho()

    input_line = ""
    while True:
        # Clear the screen
        stdscr.clear()

        # Get the last word and try to generate suggestions
        words = input_line.strip().split()
        if words:
            last_word = words[-1].lower()
            suggestions = [f"{last_word} {suggestion[0]} ({suggestion[1]:.4f})" for suggestion in
                           bigrams_dict.get(last_word, [])]
        else:
            suggestions = []

        # Limit the suggestion size to suggestions_n
        suggestions = suggestions[:SN] + [""] * (SN - len(suggestions))


        for i, suggestion in enumerate(suggestions):
            stdscr.addstr(i, 0, suggestion)

        # Display the input line on the sixth line.
        stdscr.addstr(SN, 0, f"Perplexity: {calculate_perplexity(input_line):.4f}")
        stdscr.addstr(SN + 1, 0, input_line)
        stdscr.refresh()

        # Wait for user input.
        c = stdscr.get_wch()
        if c == "\n": # Enter
            pass
        if c in ('\b', '\x7f', curses.KEY_BACKSPACE, 127): # backspace
            input_line = input_line[:-1]
        else:
            input_line += str(c)

        # If user types a space, the word has ended and the loop will continue to
        # be based on the new word.

    # After finishing, display the final input.
    stdscr.addstr(SN + 2, 0, "Final input: " + input_line)
    stdscr.getch()


curses.wrapper(main)