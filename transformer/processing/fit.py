import torch

def periodic_checkpoint_filename(epoch: int, val_loss: float) -> str:
    return f"checkpoint_last_epoch_{epoch + 1}_val_loss_{val_loss:.4f}.pth"

def best_checkpoint_filename(epoch: int, val_loss: float) -> str:
    return f"checkpoint_best.pth"

def train_transformer(model, train_dataloader, loss_fn, optimizer, scheduler, num_epochs, val_dataloader, device):
    model.train()
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        total_loss = 0

        batches = len(train_dataloader)
        for i, batch in enumerate(train_dataloader):
            src = batch['input_ids'].to(device)
            # shifted right
            tgt_input = batch['decoder_input_ids'].to(device)
            # label
            tgt_label = batch['labels'].to(device)

            # forward pass
            optimizer.zero_grad()
            predictions = model(src, tgt_input)
            loss = loss_fn(predictions.transpose(1, 2), tgt_label)

            # backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs} "
                  f"step {i + 1}/{batches}: "
                  f"\ttrain loss: {loss.item():.4f}")

        # konec epochy
        val_loss_epoch = validation_loop(model, val_dataloader, loss_fn, device)
        print(f"!!! EPOCH END !!! {epoch + 1}/{num_epochs}, "
              f"\tval loss: {val_loss_epoch:.4f}")

        # krok scheduleru
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss_epoch)
        else:
            raise ValueError("Scheduler není typu ReduceLROnPlateau")

        # checkpoint poslední epochy
        checkpoint_last = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'val_loss_at_save': val_loss_epoch,
        }
        torch.save(checkpoint_last, periodic_checkpoint_filename(epoch, val_loss_epoch))

        # checkpoint pokud došlo ke zlepšení
        if val_loss_epoch < best_val_loss:
            print(f"nová nejlepší validační ztráta: {val_loss_epoch:.4f} (předchozí: {best_val_loss:.4f})")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'val_loss_at_save': val_loss_epoch,
            }
            checkpoint_filename = best_checkpoint_filename(epoch, val_loss_epoch)
            torch.save(checkpoint, checkpoint_filename)
            print(f"checkpoint uložen jako {checkpoint_filename}")

            best_val_loss = val_loss_epoch

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