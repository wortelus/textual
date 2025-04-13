import random
import string
import time
import statistics

from compress.encode.fibonacci import fibonacci_decode, fibonacci_encode
from compress.encode.gamma import gamma_decode, gamma_encode
from compress.encode.unary import unary_decode, unary_encode
from compress.search import search_plain, search_encoded

# velikost slovníku
NUM_WORDS = 1000
# počet dokumentů (docID od 1 do 10 000)
NUM_DOCS = 10_000
# počet unikátních (slovo, docID) dvojic
NUM_PAIRS = 1_000_000


def generate_random_word(min_len=3, max_len=10):
    # vygeneruje náhodné slovo složené z malých písmen
    return ''.join(random.choices(string.ascii_lowercase, k=random.randint(min_len, max_len)))


# slovník s NUM_WORDS náhodnými slovy
dictionary = [generate_random_word() for _ in range(NUM_WORDS)]

# invertovaný index – pro každé slovo prázdný seznam
# word: [docID1, docID2, ...]
inverted_index = {word: [] for word in dictionary}

# celkem NUM_WORDS * NUM_DOCS možných dvojic.
all_possible = NUM_WORDS * NUM_DOCS

# Náhodně vybereme NUM_PAIRS unikátních čísel, která budeme mapovat na (slovo, docID)
sample_indices = random.sample(range(all_possible), NUM_PAIRS)

# Rozložení: x % NUM_WORDS = index slova, x // NUM_WORDS + 1 = docID (protože docIDs jsou od 1)
for x in sample_indices:
    word_index = x % NUM_WORDS
    docID = (x // NUM_WORDS) + 1
    inverted_index[dictionary[word_index]].append(docID)

# Seřadíme seznamy docIDs pro každé slovo
for word in inverted_index:
    inverted_index[word] = sorted(set(inverted_index[word]))

def gap_encode(doc_ids: list) -> list:
    """
    Převede seznam docIDs na posloupnost gapů.
    První číslo zůstává beze změny, každé další číslo se nahradí rozdílem k předchozímu.
    """
    if not doc_ids:
        return []
    gaps = [doc_ids[0]]
    for i in range(1, len(doc_ids)):
        gaps.append(doc_ids[i] - doc_ids[i - 1])
    return gaps


# velikosti v bitech
total_plain_size = 0
total_unary_size = 0
total_gamma_size = 0
total_fib_size = 0

# uchováme reprezentace u každého slova pro testování vyhledávání
plain_index = {}
unary_index = {}
gamma_index = {}
fib_index = {}

for word, doc_ids in inverted_index.items():
    if not doc_ids:
        continue
    # Plain reprezentace: čísla oddělená mezerou
    plain_str = " ".join(map(str, doc_ids))
    # total_plain_size += len(plain_str)
    # počet bitů v plain reprezentaci
    total_plain_size += 32 * len(doc_ids) + (len(doc_ids) - 1) * 8

    # Získáme gap encoded verzi
    gaps = gap_encode(doc_ids)

    # Zakódujeme gapy pomocí tří metod:
    unary_str = "".join(unary_encode(gap) for gap in gaps)
    gamma_str = "".join(gamma_encode(gap) for gap in gaps)
    fib_str = "".join(fibonacci_encode(gap) for gap in gaps)

    # vzhledem k tomu že tyto kódy jsou binárně ale v str, size je tedy velikost v bitech
    total_unary_size += len(unary_str)
    total_gamma_size += len(gamma_str)
    total_fib_size += len(fib_str)

    # uložíme reprezentace do indexu
    plain_index[word] = plain_str
    unary_index[word] = unary_str
    gamma_index[word] = gamma_str
    fib_index[word] = fib_str

print("=== Velikosti reprezentací (celková délka v bitech) ===")
print(f"Plain:           {total_plain_size}")
print(f"Unární:          {total_unary_size}")
print(f"Eliasovo gamma:  {total_gamma_size}")
print(f"Fibonacci:       {total_fib_size}")

#
# tvorba dotazů pro time benchmark
#

# seznam dotazů: (word, target)
search_queries = []
for word, doc_ids in inverted_index.items():
    if doc_ids:
        # vybereme náhodně jeden docID z daného seznamu
        target = random.choice(doc_ids)
        search_queries.append((word, target))

# omezíme počet dotazů
N_SEARCHES = 1000
if len(search_queries) > N_SEARCHES:
    search_queries = random.sample(search_queries, N_SEARCHES)

#
# Měření času
#

plain_times = []
unary_times = []
gamma_times = []
fib_times = []

# pro plain variantu (použijeme přímo původní seznam z invertovaného indexu)
for word, target in search_queries:
    doc_ids = inverted_index[word]
    t0 = time.time()
    found = search_plain(doc_ids, target)
    t1 = time.time()
    plain_times.append(t1 - t0)

for word, target in search_queries:
    encoded = unary_index[word]
    t0 = time.time()
    found = search_encoded(encoded, unary_decode, target)
    t1 = time.time()
    unary_times.append(t1 - t0)

for word, target in search_queries:
    encoded = gamma_index[word]
    t0 = time.time()
    found = search_encoded(encoded, gamma_decode, target)
    t1 = time.time()
    gamma_times.append(t1 - t0)

for word, target in search_queries:
    encoded = fib_index[word]
    t0 = time.time()
    found = search_encoded(encoded, fibonacci_decode, target)
    t1 = time.time()
    fib_times.append(t1 - t0)

print("\n=== Průměrné doby vyhledávání (na 100 dotazů, v sekundách) ===")
print(f"Plain:           {statistics.mean(plain_times):.8f} s")
print(f"Unární:          {statistics.mean(unary_times):.8f} s")
print(f"Eliasovo gamma:  {statistics.mean(gamma_times):.8f} s")
print(f"Fibonacci:       {statistics.mean(fib_times):.8f} s")
