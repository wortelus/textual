import random
import string
import matplotlib.pyplot as plt

from pattern_matching.bmh import bmh_search
from pattern_matching.bruteforce import brute_force_search
from pattern_matching.kmp import kmp_search

def generate_text(n, alphabet):
    return ''.join(random.choice(alphabet) for _ in range(n))

def generate_pattern(m, alphabet):
    return ''.join(random.choice(alphabet) for _ in range(m))

text_lengths = [100, 500, 1000, 5000, 10000]
def get_pattern_length(n):
    return max(5, n // 10)

alphabets = {
    'malá písmena': string.ascii_lowercase,
    'DNA': "ACGT"
}
algorithms = {
    'brute force': brute_force_search,
    'kmp': kmp_search,
    'bmh': bmh_search
}
num_runs = 5

# Výsledky budeme ukládat jako slovník:
# { alphabet: {
#   algoritmus: [průměrná porovnání pro každou délku textu]
#   }
# }
results = {alphabet_name: {algo: [] for algo in algorithms} for alphabet_name in alphabets}

for alphabet_name, alphabet in alphabets.items():
    for n in text_lengths:
        m = get_pattern_length(n)
        avg_comparisons = {algo: 0 for algo in algorithms}
        for _ in range(num_runs):
            text = generate_text(n, alphabet)
            pattern = generate_pattern(m, alphabet)
            for algo_name, algo_func in algorithms.items():
                _, comparisons = algo_func(text, pattern)
                avg_comparisons[algo_name] += comparisons

        # průměrovani přes num_runs
        for algo_name in algorithms:
            avg_comparisons[algo_name] /= num_runs
            results[alphabet_name][algo_name].append(avg_comparisons[algo_name])

# Vykreslení grafů
plt.figure(figsize=(12, 6))

for idx, (alphabet_name, data) in enumerate(results.items()):
    plt.subplot(1, 2, idx + 1)
    for algo_name, comp_values in data.items():
        plt.plot(text_lengths, comp_values, marker='o', label=algo_name)
    plt.title(f"Efektivita pro abecedu: {alphabet_name}")
    plt.xlabel("Délka textu")
    plt.ylabel("Průměrný počet porovnání")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
