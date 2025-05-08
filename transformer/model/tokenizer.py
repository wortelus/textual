from transformers import AutoTokenizer

def load_tokenizer(model_name: str = "t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Velikost slovníku (vocab_size): {tokenizer.vocab_size}")

    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print("padding token id:", tokenizer.pad_token_id)

    return tokenizer

def test_tokenizer(tokenizer):
    sample_sentence = "This is a test sentence."
    encoded = tokenizer.encode(sample_sentence, return_tensors="pt")
    decoded = tokenizer.decode(encoded[0])

    print(f"Sentence:\t{sample_sentence}")
    print(f"Encoded:\t{encoded}")
    print(f"Decoded:\t{decoded}")