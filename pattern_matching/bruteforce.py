def brute_force_search(text, pattern):
    """
    Vyhledávání vzoru v textu metodou hrubé síly.
    Vrací tuple: (seznam indexů shod, počet porovnání)
    """
    n = len(text)
    m = len(pattern)
    positions = []
    comparisons = 0

    for i in range(n - m + 1):
        comparisons += 1

        match = True
        for j in range(m):
            comparisons += 2
            if text[i + j] != pattern[j]:
                match = False
                break

        comparisons += 1
        if match:
            positions.append(i)

    return positions, comparisons
