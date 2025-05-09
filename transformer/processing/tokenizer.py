from transformers import AutoTokenizer
from transformer.const import TOKENIZER_NAME

def get_tokenizer(name: str = TOKENIZER_NAME):
    tokenizer = AutoTokenizer.from_pretrained(name)

    TOK_PAD_IDX = tokenizer.pad_token_id
    TOK_VOCAB_SIZE = tokenizer.vocab_size

    print(f"tokenizér {name} načten, vocab size: {TOK_VOCAB_SIZE}, pad token id: {TOK_PAD_IDX}")
    return tokenizer, TOK_PAD_IDX, TOK_VOCAB_SIZE