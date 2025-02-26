import nltk
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize

# Příklad trénovacích dat – seznam vět
data = [
    "This is a sentence .",
    "This is another sentence .",
    "Yet another sentence ."
]

# Tokenizace vět (doporučuje se převést na malá písmena)
tokenized_text = [word_tokenize(sent.lower()) for sent in data]

# Definice n-gramu, zde bigramy (n=2) s přidáním paddingu
n = 2
train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)

# Vytvoření Laplaceova (add-one) jazykového modelu
model = Laplace(n)
model.fit(train_data, padded_vocab)

# Příklad: Výpočet pravděpodobnosti slova 'is' následující po 'this'
prob = model.score("is", ["this"])
print(f"P(Po 'this' se vyskytne 'is') = {prob}")
