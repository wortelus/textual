import os

import torch

model_checkpoints = os.listdir("first_train")
model_checkpoints = [f for f in model_checkpoints if f.endswith('.pth')]
print(model_checkpoints)

states = []
for checkpoint in model_checkpoints:
    checkpoint_path = os.path.join("first_train", checkpoint)
    print(f"Načítám model z {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint.get('model_state_dict')
    optimizer_state_dict = checkpoint.get('optimizer_state_dict')
    scheduler_state_dict = checkpoint.get('scheduler_state_dict')
    states.append({
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "epoch": checkpoint.get('epoch'),
        "best_val_loss": checkpoint.get('best_val_loss'),
    })

print(f"Načteno {len(states)} stavů modelu.")
for state in states:
    print("lr: ", state["scheduler_state_dict"]["_last_lr"])