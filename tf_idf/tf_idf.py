import math
import os
import re
import string
from collections import defaultdict, Counter

import nltk

DIR_NAME = "../gutenberg"

custom_stopwords = {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'that', 'this', 'for', 'on', 'with', 'as', 'by',
                    'an', 'at', 'be', 'or', 'are'}
nltk_stopwords = nltk.corpus.stopwords.words("english")


def preprocess(text):
    # malá písmena
    text = text.lower()
    # odstranění interpunkce
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # tokenizace - dle mezer
    tokens = text.split()
    # odstranění custom stopwords
    tokens = [token for token in tokens if token not in custom_stopwords]
    # odstranění nltk stopwords
    tokens = [token for token in tokens if token not in nltk_stopwords]

    return tokens


def load_docs():
    docs = []
    files_in_dir = os.listdir(DIR_NAME)
    for file in files_in_dir:
        with open(os.path.join(DIR_NAME, file), "r") as f:
            content = f.read()
            docs.append(preprocess(content))

    return docs


def calculate_tf(docs):
    # tf dle relativní četnosti v jednom (1) dokumentu
    # tf = {token: count / total_tokens} - term frequency
    tf_documents = []
    for tokens in docs:
        total_tokens = len(tokens)

        # lokální tokeny
        tf = {}
        counts = Counter(tokens)
        for token, count in counts.items():
            # relativní četnost v rámci 1 dokumentu
            tf[token] = count / total_tokens
        tf_documents.append(tf)

    return tf_documents


def calculate_tf_idf(tf_documents):
    # df(t) je počet dokumentů, ve kterých se token vyskytuje
    df = defaultdict(int)
    docs_count = len(tf_documents)
    for tf in tf_documents:
        for token in tf.keys():
            df[token] += 1

    # výpočet idf pro každý term/token
    # log( N / df(t) )
    idf = {}
    for token, df_t in df.items():
        idf[token] = math.log(docs_count / df_t)

    # Výpočet tf-idf pro každý dokument
    tf_idf_documents = []
    for tf in tf_documents:
        tf_idf = {}
        for token, tf_val in tf.items():
            tf_idf[token] = tf_val * idf[token]
        tf_idf_documents.append(tf_idf)

    return tf_idf_documents


def score_document(query, tf_idf):
    score = 0
    for token in query:
        score += tf_idf.get(token, 0)
    return score


def main():
    # load docs
    docs = load_docs()
    tf_documents = calculate_tf(docs)
    tf_idf_documents = calculate_tf_idf(tf_documents)

    # dotaz
    query = input("Zadejte dotaz: ")
    query = preprocess(query)

    # seřazení dokumentů podle skóre
    scores = [score_document(query, doc) for doc in tf_idf_documents]
    sorted_docs = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
    print("Dokumenty seřazené podle skóre:")
    for i, score in sorted_docs:
        print(f"Dokument \t{i}:\t{score:.4f}")


if __name__ == "__main__":
    main()
