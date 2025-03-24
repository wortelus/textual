import string


def levenshtein_distance(s, t):
    # pokud jsou slova stejné, vzdálenost je 0
    if s == t:
        return 0

    # pokud je jedno ze slov prázdné, vzdálenost je délka druhého slova
    if len(s) == 0:
        return len(t)
    if len(t) == 0:
        return len(s)

    # vytvoření matice (len(s) + 1) x (len(t) + 1)
    rows = len(s) + 1
    cols = len(t) + 1
    d = [[0 for _ in range(cols)] for _ in range(rows)]

    # Inicializace prvního řádku a prvního sloupce
    for i in range(rows):
        d[i][0] = i
    for j in range(cols):
        d[0][j] = j

    # Dynamické programování
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # smazání
                d[i][j - 1] + 1,  # vložení
                d[i - 1][j - 1] + cost  # nahrazení
            )
    return d[-1][-1]

def edits1(word):
    # abeceda – kromě anglických písmen jsou zde i některé české znaky
    letters = string.ascii_lowercase + "áčďéěíňóřšťúůýž"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]

    return set(deletes + transposes + replaces + inserts)


# Funkce pro generování všech variant na editační vzdálenost 2.
def edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))


def main():
    # Testování funkce na několika příkladech:
    test_cases = [
        ("kitten", "sitting"),  # očekávaná 3
        ("flaw", "lawn"),  # očekávaná 2
        ("škola", "školy"),  # očekáváná 1
        ("jablko", "banán")  # očekávaná 5
    ]

    for s, t in test_cases:
        dist = levenshtein_distance(s, t)
        print(f"Levenshteinova vzdálenost mezi '{s}' a '{t}' je: {dist}")

if __name__ == "__main__":
    main()