import torch
from torch import nn

from transformer.const import NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, FFN_HID_DIM, DROPOUT, \
    MAX_MODEL_SEQ_LEN
from transformer.load import load_files, get_stats
from transformer.model.seq2seq import Seq2SeqTransformer
from transformer.model.tokenizer import load_tokenizer, test_tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    train_json, val_json, test_json = load_files("corpus")

    print("train:", len(train_json))
    get_stats(train_json)
    print("val:", len(val_json))
    get_stats(val_json)
    print("test:", len(test_json))
    get_stats(test_json)

    tokenizer = load_tokenizer()
    test_tokenizer(tokenizer)

    # poznámka, summary má vždy délku 300
    SRC_VOCAB_SIZE = tokenizer.vocab_size
    TGT_VOCAB_SIZE = tokenizer.vocab_size

    print("DEVICE:", DEVICE)

    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        SRC_VOCAB_SIZE,
        TGT_VOCAB_SIZE,
        FFN_HID_DIM,
        DROPOUT,
        MAX_MODEL_SEQ_LEN,  # maxlen pro PositionalEncoding
        batch_first=True  # Důležité!
    ).to(DEVICE)

    # počet parametrů
    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'náš model má {count_parameters(model):,} trénovatelných parametrů')

    # dummy data pro testování
    BATCH_SIZE_DUMMY = 4
    MAX_SRC_LEN_DUMMY = 60
    MAX_TGT_LEN_DUMMY = 50
    src_dummy = torch.randint(0, SRC_VOCAB_SIZE, (BATCH_SIZE_DUMMY, MAX_SRC_LEN_DUMMY), device=DEVICE)
    tgt_dummy = torch.randint(0, TGT_VOCAB_SIZE, (BATCH_SIZE_DUMMY, MAX_TGT_LEN_DUMMY),
                              device=DEVICE)  # Toto jsou "posunuté" cíle

    # výstup by měl mít tvar [BATCH_SIZE_DUMMY, MAX_TGT_LEN_DUMMY, TGT_VOCAB_SIZE]
    # test průchodu bez gradientů
    with torch.no_grad():
        # model do evaluačního módu
        model.eval()
        # Dummy průchod
        logits_dummy = model(src_dummy, tgt_dummy)
        # zpět do trénovacího módu
        model.train()


    print("shape výstupních logitů:", logits_dummy.shape)
    print("done")


if __name__ == "__main__":
    main()
