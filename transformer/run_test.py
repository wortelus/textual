import torch
import os

from transformer.model.seq2seq import Seq2SeqTransformer
from transformer.processing.dataset import load_samsum
from transformer.processing.tokenizer import get_tokenizer

from transformer.const import NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, FFN_HID_DIM, DROPOUT, \
    MAX_MODEL_SEQ_LEN, MAX_SRC_SEQ_LEN, MAX_TGT_SEQ_LEN

def main():
    # tokenizer
    tokenizer, pad_idx, vocab_size = get_tokenizer()
    tokenized_train, tokenized_val, tokenized_test = load_samsum(train_size=1, val_size=1, test_size=20,
                                                                 tokenizer=tokenizer,
                                                                 max_src_seq_len=MAX_SRC_SEQ_LEN,
                                                                 max_tgt_seq_len=MAX_TGT_SEQ_LEN,
                                                                 pad_idx=pad_idx,
                                                                 seed=10)

    # available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_file_path = "checkpoint_last_epoch_2_val_loss_4.9016.3490.pth"
    if not os.path.isfile(checkpoint_file_path):
        print(f"Chyba: Soubor checkpointu nenalezen na {checkpoint_file_path}")

    print(f"Načítám model z {checkpoint_file_path}")
    checkpoint = torch.load(checkpoint_file_path, map_location=device)
    model_state_dict = checkpoint.get('model_state_dict')

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

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Stav modelu načten.")
    else:
        try:
            model.load_state_dict(checkpoint)
            print("Stav modelu načten (načten přímo state_dict).")
        except Exception as e:
            print(f"Chyba při načítání stavu modelu: {e}")
            print("Ujistěte se, že checkpoint obsahuje klíč 'model_state_dict' nebo že soubor obsahuje přímo state_dict.")

    model.to(device)

    model.eval()
    total_test_loss = 0

    with torch.no_grad():
        for batch in tokenized_test:
            src = batch['input_ids'].to(device)
            tgt_input = batch['decoder_input_ids'].to(device)
            predictions = model(src.unsqueeze(0), tgt_input.unsqueeze(0))
            predicted_ids = torch.argmax(predictions, dim=-1)
            print("-" * 30)
            for i in range(predicted_ids.size(0)):
                predicted_text = tokenizer.decode(predicted_ids[i].tolist(), skip_special_tokens=True)
                original_text = tokenizer.decode(batch['labels'], skip_special_tokens=True)
                print(f"Predicted: {predicted_text}")
                print(f"Original: {original_text}")

if __name__ == "__main__":
    main()