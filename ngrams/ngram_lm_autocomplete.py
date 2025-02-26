#
# Simple n-gram language model for autocompletion. Uses bigrams and Laplace smoothing to create the model
# and curses to create a simple terminal interface for autocompletion.
#

import curses

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

# Tokenization
tokens = word_tokenize(cs, language)

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
        stdscr.addstr(SN, 0, input_line)
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