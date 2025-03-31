from os import listdir
from os.path import join
import re
from collections import defaultdict

import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def get_docs(target_dir: str) -> list[str]:
    docs = []
    for file in listdir(target_dir):
        if file.startswith("part_0"):
            with open(join(target_dir, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs


def tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def create_index(docs: list[str]):
    inverted_index = defaultdict(dict)
    for doc_id, text in enumerate(docs):
        tokens = tokenize(text)
        for token in tokens:
            inverted_index[token][doc_id] = inverted_index[token].get(doc_id, 0) + 1

    # for token, posting in inverted_index.items():
    #     print(f"Token: {token} -> {posting}")

    return inverted_index


