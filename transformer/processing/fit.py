import csv
import os

import torch


def periodic_checkpoint_filename(epoch: int, val_loss: float) -> str:
    return f"checkpoint_last_epoch_{epoch + 1}_val_loss_{val_loss:.4f}.pth"


def best_checkpoint_filename(epoch: int, val_loss: float) -> str:
    return f"checkpoint_best.pth"


def train_transformer(model,
                      train_dataloader,
                      loss_fn,
                      optimizer,
                      scheduler,
                      num_epochs,
                      val_dataloader,
                      device,
                      csv_log_filename: str = "train_log.csv",
                      checkpoint_filename: str = None):
    model.train()
    best_val_loss = float("inf")

    start_epoch = 0
    if checkpoint_filename:
        print(f"načítám checkpoint z {checkpoint_filename}")
        checkpoint = torch.load(checkpoint_filename)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Pokud chceš pokračovat ve trénování od stejné epochy
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        # val_loss_at_save = checkpoint['val_loss_at_save']

    file_is_empty_or_nonexistent = (not os.path.exists(csv_log_filename) or
                                    os.path.getsize(csv_log_filename) == 0)

    with open(csv_log_filename, 'a+', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if file_is_empty_or_nonexistent:
            csv_writer.writerow(['epoch', 'avg_train_loss', 'val_loss', 'learning_rate', 'epoch_duration_s'])

        for epoch in range(start_epoch, num_epochs):
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

            # learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"learning rate: {current_lr:.6e}")
            # CSV
            csv_writer.writerow([
                epoch + 1,
                f"{total_loss / len(train_dataloader):.4f}",
                f"{val_loss_epoch:.4f}",
                f"{current_lr:.6e}"
            ])

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
