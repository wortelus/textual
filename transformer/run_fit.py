import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer.const import MAX_SRC_SEQ_LEN, MAX_TGT_SEQ_LEN
from transformer.processing.dataset import load_samsum, dataset_stats
from transformer.processing.fit import train_transformer
from transformer.processing.tokenizer import get_tokenizer
from transformer.model.seq2seq import Seq2SeqTransformer

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
                                                    seed=seed)

    BATCH_SIZE = 16
    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(tokenized_val, batch_size=BATCH_SIZE)

    from transformer.const import NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, FFN_HID_DIM, DROPOUT, \
        MAX_MODEL_SEQ_LEN

    model = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        emb_size=EMB_SIZE,
        nhead=NHEAD,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        dim_feedforward=FFN_HID_DIM,
        dropout=DROPOUT,
        max_model_seq_len=MAX_MODEL_SEQ_LEN,
        pad_idx=pad_idx,
        batch_first=True
    ).to(device)

    print(f'Model má {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trénovatelných parametrů.')

    # loss funkce - křížová entropie
    # ignore_index=PAD_IDX ať padding tokeny v cílových sekvencích (labels) nejsou brány v úvahu
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # optimizér - AdamW
    learning_rate = 0.0001
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # learning rate scheduler
    # snížení learning rate, pokud se validace nezlepšuje
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)

    # TRÉNOVACÍ SMYČKA
    num_epochs = 20
    print("trénování...")
    train_transformer(model=model,
                      train_dataloader=train_dataloader,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      num_epochs=num_epochs,
                      val_dataloader=val_dataloader,
                      device=device)

    print("hotovo")

if __name__ == "__main__":
    main()