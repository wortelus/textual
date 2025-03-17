from pattern_matching.bruteforce import brute_force_search
from pattern_matching.kmp import kmp_search
from pattern_matching.bmh import bmh_search

import random
import string

def run_print(text, pattern):
    print(f"brute force:\t{brute_force_search(text, pattern)}")
    print(f"kmp:\t\t\t{kmp_search(text, pattern)}")
    print(f"bmh:\t\t\t{bmh_search(text, pattern)}")

n = 100
m = 5
print(f"Náhodný alpphabetický text n={n}, m={m}")
text = ''.join(random.choice(string.ascii_uppercase) for _ in range(n))
pattern = ''.join(random.choice(string.ascii_uppercase) for _ in range(m))
run_print(text, pattern)

n = 1000
m = 50
print(f"Náhodný alpphabetický text n={n}, m={m}")
text = ''.join(random.choice(string.ascii_uppercase) for _ in range(n))
pattern = ''.join(random.choice(string.ascii_uppercase) for _ in range(m))
run_print(text, pattern)

n = 1000
m = 50
print(f"Alphabetický text n={n}, m={m} * A")
text = ''.join(random.choice(string.ascii_uppercase) for _ in range(n))
pattern = 'A' * m
run_print(text, pattern)

n = 1000
m = "AGCT"
print(f"DNA test n={n}, m={m}")
text = ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(n))
pattern = m
run_print(text, pattern)

n = 1000
m = 3
print(f"AAAAA... text a AAAAA....B pattern n={n}, m={m}")
text = ''.join(random.choice(['a', 'b']) for _ in range(n))
pattern = "a" * (m - 1) + "b"

run_print(text, pattern)

n = 1000
m = 4
print(f"Náhodný alpphabetický text n={n}, a ABCD * m={m}")
text = ''.join(random.choice(string.ascii_lowercase) for _ in range(n))
pattern = "ABCD" * m

run_print(text, pattern)

n = 1000
m = 4
print(f"Náhodný a,b,c text n={n}, a ABCD * m={m}")
text = ''.join(random.choice(['a', 'b', 'c']) for _ in range(n))
pattern = "abc" * m

run_print(text, pattern)


if __name__ == '__main__':
    pass
