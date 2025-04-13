def search_plain(doc_ids: list, target: int) -> bool:
    # Hledání v nezakódovaném (plain) setříděném seznamu (lineární pro ilustraci, mohl by být log n)
    for doc in doc_ids:
        if doc == target:
            return True
        if doc > target:
            return False
    return False


def search_encoded(encoded: str, decode_func, target: int) -> bool:
    # sekvenční hledání v zakódovaném seznamu (např. unární, gamma)
    current = 0
    remainder = encoded
    while remainder:
        gap, remainder = decode_func(remainder)
        current += gap
        if current == target:
            return True
        if current > target:
            return False
    return False
