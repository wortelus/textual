import re
import collections
import string

from spell_checker.levenshtein import edits1, levenshtein_distance, edits2


def read_line(line):
    if "/" in line:
        return line.split("/")[0].lower()
    return line.lower()

def read_dictionary(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = text.split("\n")
    words = set([read_line(line) for line in lines])
    return words

def read_regular_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    words = re.findall(r'\b[^\W\d_]+\b', text, flags=re.UNICODE)
    # Normalizace – převedeme všechna slova na malá písmena.
    words = [word.lower() for word in words]
    return words

def build_frequency_dict(words):
    return collections.Counter(words)

def correct_word(dict_words: set[str], word):
    lw = word.lower()
    # Pokud slovo už ve slovníku je, vrátíme jej
    if lw in dict_words:
        return lw

    # Nejprve generujeme kandidáty ED1
    candidates1 = edits1(lw)
    valid1 = dict_words.intersection(candidates1)
    if valid1:
        best = min(valid1, key=lambda w: levenshtein_distance(lw, w))
        return best + " (ED1)"

    # Pokud není, generujeme kandidáty ED2
    candidates2 = edits2(lw)
    valid2 = dict_words.intersection(candidates2)
    if valid2:
        best = min(valid2, key=lambda w: levenshtein_distance(lw, w))
        return best + " (ED2)"
    return None

def correct_word_lookup(dict_words: set[str], word, counter):
    lw = word.lower()
    # Pokud slovo už ve slovníku je, vrátíme jej
    if lw in dict_words:
        return lw

    # Nejprve generujeme kandidáty ED1
    candidates1 = edits1(lw)
    valid1 = dict_words.intersection(candidates1)
    if valid1:
        best = max(valid1, key=lambda w: counter[w])
        return best + " (ED1)"

    # Pokud není, generujeme kandidáty ED2
    candidates2 = edits2(lw)
    valid2 = dict_words.intersection(candidates2)
    if valid2:
        best = max(valid2, key=lambda w: counter[w])
        return best + f" (ED2)"
    return None

def main():
    filename = "cs_CZ.dic"
    dictionary_set = read_dictionary(filename)
    print("Celkový počet unikátních slov:", len(dictionary_set))

    sentence = input("Zadejte větu: ")
    # Rozdělíme větu na slova
    words = re.findall(r'\b\w+\b', sentence, flags=re.UNICODE)
    result = []
    for w in words:
        # Zpracujeme pouze slova, která obsahují pouze písmena (včetně diakritiky)
        if re.fullmatch(r'[A-Za-zÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž]+', w):
            correction = correct_word(dictionary_set, w)
            if correction:
                result.append(correction)
            else:
                result.append(w)
        else:
            result.append(w)
    print("Opravená slova:")
    for original, corr in zip(words, result):
        print(f"{original} -> {corr}")

    print("regular dataset")
    filename = "cs_third.txt"

    regular = read_regular_dataset(filename)
    regular_set = set(regular)
    print("Počet slov v datasetu:", len(regular))
    freq_dict = build_frequency_dict(regular)
    result = []
    for w in words:
        # Zpracujeme pouze slova, která obsahují pouze písmena (včetně diakritiky)
        if re.fullmatch(r'[A-Za-zÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž]+', w):
            correction = correct_word_lookup(regular_set, w, freq_dict)
            if correction:
                result.append(correction)
            else:
                result.append(w)
        else:
            result.append(w)
    print("Opravená slova:")
    for original, corr in zip(words, result):
        print(f"{original} -> {corr}")


if __name__ == '__main__':
    main()
