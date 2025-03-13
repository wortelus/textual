def build_bmh_shift_table(pattern):
    m = len(pattern)
    shift_table = {}

    # Defaul posun je délka vzoru
    for char in set(pattern):
        shift_table[char] = m

    # tím že budujeme od konce, bude mít posun znaku vždy nejmenší hodnotu z těch možných
    for i in range(m - 1):
        current_shift = m - i - 1
        char = pattern[i]
        shift_table[char] = current_shift
    return shift_table


def bmh_search(text, pattern):
    n = len(text)
    m = len(pattern)
    positions = []
    comparisons = 0

    if m == 0:
        return positions, comparisons

    shift_table = build_bmh_shift_table(pattern)
    i = 0

    while i <= n - m:
        comparisons += 1

        j = m - 1
        # kontrolujeme od konce, zda sedí
        while j >= 0:
            comparisons += 2
            if pattern[j] != text[i + j]:
                break
            j -= 1

        # Vzor nalezen
        if j < 0:
            positions.append(i)
            i += 1
        else:
            # Vzor nenalezen, posun o shift table value
            shift = shift_table.get(text[i + m - 1], m)
            i += shift

        comparisons += 1
    return positions, comparisons