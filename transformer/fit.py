import torch


def train_epoch(model, dataloader, optimizer, loss_func, device):
    model.train()  # Nastaví model do trénovacího módu
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        src = batch['input_ids'].to(device)  # Tvar: [batch_size, src_seq_len]

        # attention_mask pro src se použije implicitně v padding masce modelu
        tgt_input = batch['decoder_input_ids'].to(device)  # Tvar: [batch_size, tgt_seq_len]
        labels = batch['labels'].to(device)  # Tvar: [batch_size, tgt_seq_len]

        optimizer.zero_grad()  # Vynulování gradientů

        # Průchod modelem
        # `tgt_input` jsou posunuté cíle (začínají BOS, končí předposledním tokenem)
        logits = model(src, tgt_input)  # Výstup: [batch_size, tgt_seq_len, tgt_vocab_size]

        # Přeskládání pro CrossEntropyLoss
        # Očekává: Logits: (N, C, ...), Target: (N, ...) kde C je počet tříd
        # Naše logits: (batch_size, tgt_seq_len, tgt_vocab_size) -> (batch_size * tgt_seq_len, tgt_vocab_size)
        # Naše labels: (batch_size, tgt_seq_len) -> (batch_size * tgt_seq_len)
        loss = loss_func(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        loss.backward()  # Zpětná propagace
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Ořezání gradientů
        optimizer.step()  # Aktualizace vah

        total_loss += loss.item()

        if batch_idx % 10 == 0:  # Logování každých 10 batchů
            print(f"batch {batch_idx}/{len(dataloader)}, train loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate_epoch(model, dataloader, loss_func, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch['input_ids'].to(device)
            tgt_input = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(src, tgt_input)
            loss = loss_func(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def validation_loop(model, dataloader, loss_fn, device):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch["input_ids"].to(device), batch["labels"].to(device)
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            pred = model(X, y_input)

            loss = loss_fn(pred.transpose(1, 2), y_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)