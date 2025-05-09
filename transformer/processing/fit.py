import torch


def train_transformer(model, data_loader, loss_fn, optimizer, num_epochs, val_dataloader, device):
    model.train()
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            src = batch['input_ids'].to(device)
            tgt_input = batch['decoder_input_ids'].to(device)
            tgt_output = batch['labels'].to(device)

            # forward pass
            optimizer.zero_grad()
            predictions = model(src, tgt_input)
            loss = loss_fn(predictions.transpose(1, 2), tgt_output)

            # backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            val_loss = validation_loop(model, val_dataloader, loss_fn, device)
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"\ttrain loss: {loss.item():.4f}, "
                  f"\tval loss: {val_loss:.4f}")

        # konec epochy
        val_loss = validation_loop(model, val_dataloader, loss_fn, device)
        print(f"!!! EPOCH END !!! {epoch + 1}/{num_epochs}, "
              f"\tval Loss: {val_loss:.4f}")

        # pokud došlo ke zlepšení
        if val_loss < best_val_loss:
            print(f"saving model with best val loss: {val_loss:.4f}")
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"epoch_best_{epoch + 1}_val_loss_{val_loss:.4f}.pth")

    return model


def validation_loop(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch['input_ids'].to(device)
            tgt_input = batch['decoder_input_ids'].to(device)
            tgt_output = batch['labels'].to(device)

            pred = model(src, tgt_input)

            loss = loss_fn(pred.transpose(1, 2), tgt_output)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)