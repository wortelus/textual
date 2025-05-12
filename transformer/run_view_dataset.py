import torch
from torch.utils.data import DataLoader

from transformer.const import MAX_SRC_SEQ_LEN, MAX_TGT_SEQ_LEN
from transformer.processing.dataset import load_samsum
from transformer.processing.tokenizer import get_tokenizer


def main():
    seed = 10

    # tokenizer
    tokenizer, pad_idx, vocab_size = get_tokenizer()

    # available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch zařízení: {device}")
    tokenized_train, tokenized_val, tokenized_test = load_samsum(train_size=0, val_size=0, test_size=0,
                                                                 tokenizer=tokenizer,
                                                                 max_src_seq_len=MAX_SRC_SEQ_LEN,
                                                                 max_tgt_seq_len=MAX_TGT_SEQ_LEN,
                                                                 pad_idx=pad_idx,
                                                                 stats=True,
                                                                 seed=seed)

    BATCH_SIZE = 16
    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(tokenized_val, batch_size=BATCH_SIZE)

    example = "I am Sharol. I am Maria. I am Spencer"
    encoded = tokenizer.encode(example)
    decoded = tokenizer.decode(encoded)
    print(f"encoded: {encoded}")
    print(f"decoded: {decoded}")

    label = tokenized_train[0]["labels"]
    decoder_input = tokenized_train[0]["decoder_input_ids"]
    print(f"({len(label)}) label: {' '.join([str(i) for i in label])}")
    print(f"({len(decoder_input)}) decoder_input: {' '.join([str(i) for i in decoder_input])}")


if __name__ == "__main__":
    main()
