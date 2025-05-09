import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, FlaxT5Model
import math
import time

from transformer.fit import train_epoch, evaluate_epoch, validation_loop
from transformer.model.seq2seq import Seq2SeqTransformer

# tokenizer
MODEL_NAME_FOR_TOKENIZER = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FOR_TOKENIZER)
t5_model = FlaxT5Model.from_pretrained(f"google-t5/t5-small")

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
    pad_idx=PAD_IDX,
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
print("Začínám trénink...")
def train_transformer(model, data_loader, criterion, optimizer, num_epochs=10):
    model.train()
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            src, tgt = batch["input_ids"].to(DEVICE), batch["labels"].to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Forward pass
            optimizer.zero_grad()
            predictions = model(src, tgt_input)
            loss = criterion(predictions.transpose(1, 2), tgt_output)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            val_loss = validation_loop(model, val_dataloader, loss_func, DEVICE)
            print(f"validace loss: {val_loss:.4f}")

            print(f"Epoch {epoch}/{NUM_EPOCHS}, "
                  f"Train Loss: {loss.item():.4f}")

            # Ukládání modelu
            # if val_loss < best_val_loss:
            #     print(f"Ukládám model s nejlepší validací: {val_loss:.4f}")
            #     best_val_loss = val_loss
            #     torch.save(model.state_dict(), f"epoch_{epoch}_val_loss_{val_loss:.4f}.pt")

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")

train_transformer(model, train_dataloader, loss_func, optimizer)
print("Trénink dokončen.")
