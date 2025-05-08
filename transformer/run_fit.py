import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import time  # Pro měření času epoch

from transformer.fit import train_epoch, evaluate_epoch
from transformer.load import load_files
from transformer.model.seq2seq import Seq2SeqTransformer

# tokenizer
MODEL_NAME_FOR_TOKENIZER = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FOR_TOKENIZER)
PAD_IDX = tokenizer.pad_token_id
SRC_VOCAB_SIZE = tokenizer.vocab_size
TGT_VOCAB_SIZE = tokenizer.vocab_size

# available device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Používám zařízení: {DEVICE}")

# hyperparametry modelu

# velikost embeddingu
EMB_SIZE = 256

# Musí dělit EMB_SIZE (256 % 8 == 0)
NHEAD = 8

FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1

# positional encoding
MAX_MODEL_SEQ_LEN = 512

# maximální délky pro tokenizaci
MAX_SRC_SEQ_LEN = 512
MAX_TGT_SEQ_LEN = 300

try:
    samsum_dataset_full = load_dataset("samsum")
except Exception as e:
    print(f"Chyba při načítání SAMSum: {e}. Ukončuji.")
    exit()


def preprocess_function(data):
    dialogues = [dialogue for dialogue in data["dialogue"]]
    summaries = [summary for summary in data["summary"]]

    model_inputs = tokenizer(
        dialogues,
        max_length=MAX_SRC_SEQ_LEN,
        padding="max_length",
        truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels_output = tokenizer(
            summaries,
            max_length=MAX_TGT_SEQ_LEN,
            padding="max_length",
            truncation=True
        )

    model_inputs["labels"] = labels_output["input_ids"]
    decoder_input_ids_batch = []

    for label_ids in model_inputs["labels"]:
        # Vytvoříme sekvenci začínající PAD_IDX (BOS pro T5)
        # a pokračující label_ids bez posledního tokenu.
        # Musí mít stejnou délku jako label_ids (MAX_TGT_SEQ_LEN)
        shifted_labels = [PAD_IDX] + label_ids[:-1]
        decoder_input_ids_batch.append(shifted_labels)

    model_inputs["decoder_input_ids"] = decoder_input_ids_batch
    return model_inputs


print("Tokenizace datasetu...")
small_train_dataset = samsum_dataset_full["train"].shuffle(seed=42).select(range(1000))
small_val_dataset = samsum_dataset_full["validation"].shuffle(seed=42).select(range(200))

tokenized_train = small_train_dataset.map(preprocess_function, batched=True,
                                          remove_columns=["dialogue", "summary", "id"])
tokenized_val = small_val_dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])

# Nastavení formátu na PyTorch tenzory
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "labels"])

BATCH_SIZE = 4  # Zmenšete, pokud máte málo VRAM (např. 4 nebo 8)

train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(tokenized_val, batch_size=BATCH_SIZE)

model = Seq2SeqTransformer(
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    emb_size=EMB_SIZE,
    nhead=NHEAD,
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    dim_feedforward=FFN_HID_DIM,
    dropout=DROPOUT,
    max_model_seq_len=MAX_MODEL_SEQ_LEN,
    batch_first=True
).to(DEVICE)

print(f'Model má {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trénovatelných parametrů.')

# loss funkce
# ignore_index=PAD_IDX ať padding tokeny v cílových sekvencích (labels) nebudou brány v úvahu
loss_func = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Optimizer
LEARNING_RATE = 0.0001
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# learning rate scheduler
# snížení learning rate, pokud se validace nezlepšuje
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

NUM_EPOCHS = 10
torch.autograd.set_detect_anomaly(True)

#
# TRÉNOVACÍ SMYČKA
#
best_val_loss = float("inf")
print("Začínám trénink...")
for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()
    train_loss = train_epoch(model, train_dataloader, optimizer, loss_func, DEVICE)
    val_loss = evaluate_epoch(model, val_dataloader, loss_func, DEVICE)
    epoch_time = time.time() - start_time

    print(f"Epoch {epoch}/{NUM_EPOCHS}, "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Time: {epoch_time:.2f}s")

    # aktualizace scheduleru
    scheduler.step(val_loss)

    # Ukládání modelu
    best_val_loss = val_loss
    torch.save(model.state_dict(), f"epoch_{epoch}_val_loss_{val_loss:.4f}.pt")

print("Trénink dokončen.")
