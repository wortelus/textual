def evaluate_boolean_query(query: str, index):
    # rozdělení dotazu na tokeny (předpokládáme, že operátory jsou odděleny mezerami)
    tokens = query.split()

    # výskyt v dokumentech prvního tokenu
    result_set = set(index[tokens[0]].keys()) if tokens[0] in index else set()

    i = 1
    while i < len(tokens):
        operator = tokens[i]
        next_token = tokens[i + 1]
        next_set = set(index[next_token].keys()) if next_token in index else set()

        if operator.upper() == "AND":
            result_set = result_set.intersection(next_set)
        elif operator.upper() == "OR":
            result_set = result_set.union(next_set)
        elif operator.upper() == "NOT":
            result_set = result_set.difference(next_set)
        i += 2

    return result_set


# rozšířený boolean model s váhováním
def evaluate_weighted_query(query, index):
    tokens = query.split()
    result_docs = set(index[tokens[0]].keys()) if tokens[0] in index else set()

    for i in range(1, len(tokens), 2):
        operator = tokens[i]
        next_token = tokens[i + 1]
        next_docs = set(index[next_token].keys()) if next_token in index else set()
        if operator.upper() == "AND":
            result_docs = result_docs.intersection(next_docs)
        elif operator.upper() == "OR":
            result_docs = result_docs.union(next_docs)
        elif operator.upper() == "NOT":
            result_docs = result_docs.difference(next_docs)

    # váhování: součet četností dotazovaných tokenů v dokumentech
    scores = {}
    for token in tokens:
        token = token.lower()
        if token in index:
            for doc_id, freq in index[token].items():
                if doc_id in result_docs:
                    scores[doc_id] = scores.get(doc_id, 0) + freq

    # seřazení dokumentů podle skóre (sestupně)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs
