import json
from os.path import join

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


def load_files(path: str):
    with open(join(path, "train.json"), 'r', encoding='utf-8') as f:
        train = json.load(f)
    with open(join(path, "val.json"), 'r', encoding='utf-8') as f:
        val = json.load(f)
    with open(join(path, "test.json"), 'r', encoding='utf-8') as f:
        test = json.load(f)

    return train, val, test


def get_stats(dataset):
    max_dialogue_length = np.max([len(item["dialogue"]) for item in dataset])
    max_summary_length = np.max([len(item["summary"]) for item in dataset])

    avg_dialogue_length = np.mean([len(item["dialogue"]) for item in dataset])
    avg_summary_length = np.mean([len(item["summary"]) for item in dataset])

    print(f"Max dialogue length: {max_dialogue_length}")
    print(f"Avg dialogue length: {avg_dialogue_length:.2f}")

    print(f"Max summary length: {max_summary_length}")
    print(f"Avg summary length: {avg_summary_length:.2f}")
