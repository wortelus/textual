def compute_prefix_function(pattern):
    pattern_len = len(pattern)
    prefix = [0] * pattern_len

    # délka nejdelšího prefixu, který je zároveň sufixem
    k = 0

    for p in range(1, pattern_len):
        while k > 0 and pattern[k] != pattern[p]:
            k = prefix[k - 1]
        if pattern[k] == pattern[p]:
            k += 1
        prefix[p] = k
    return prefix


def kmp_search(text, pattern):
    """
    Vyhledávání vzoru pomocí algoritmu Knuth-Morris-Pratt (KMP).
    Vrací tuple: (seznam indexů shod, počet porovnání)
    """
    n = len(text)
    m = len(pattern)
    positions = []
    comparisons = 0

    # Vytvoření prefixové tabulky
    prefix = compute_prefix_function(pattern)

    # počet shodných znaků
    q = 0
    for i in range(n):
        comparisons += 1

        while q > 0 and pattern[q] != text[i]:
            # porovnání při neúspěchu
            comparisons += 1
            q = prefix[q - 1]

        if pattern[q] == text[i]:
            q += 1
        if q == m:
            positions.append(i - m + 1)
            q = prefix[q - 1]

        comparisons += 1
    return positions, comparisons
