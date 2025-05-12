import torch
import os

from rouge_score import rouge_scorer

from transformer.model.seq2seq import Seq2SeqTransformer
from transformer.processing.dataset import load_samsum
from transformer.processing.tokenizer import get_tokenizer

from transformer.const import NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, FFN_HID_DIM, DROPOUT, \
    MAX_MODEL_SEQ_LEN, MAX_SRC_SEQ_LEN, MAX_TGT_SEQ_LEN

def main():
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

    # tokenizer
    tokenizer, pad_idx, vocab_size = get_tokenizer()
    _, _, tokenized_test = load_samsum(train_size=1, val_size=1, test_size=20,
                                                                 tokenizer=tokenizer,
                                                                 max_src_seq_len=MAX_SRC_SEQ_LEN,
                                                                 max_tgt_seq_len=MAX_TGT_SEQ_LEN,
                                                                 pad_idx=pad_idx,
                                                                 seed=10)

    # available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_file_path = "first_train/checkpoint_last_epoch_7_val_loss_4.4270.pth"
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

    sos_idx = pad_idx
    eos_idx = tokenizer.eos_token_id

    r1_list = []
    r2_list = []
    with torch.no_grad():
        for i, batch in enumerate(tokenized_test):
            src = batch['input_ids'].unsqueeze(0).to(device)

            # Smyčka pro generování token po tokenu
            decoder_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
            for step in range(MAX_TGT_SEQ_LEN):
                predictions = model(src, decoder_input)

                # predikované "pravděpodobnosti" následujícího tokenu
                # (batch, seq_len, vocab) -> predictions[:, -1, :]
                last_token_logits = predictions[:, -1, :]

                # greedy (argmax) výběr tokenu a přidání do 'decoder_input'
                predicted_token_id = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)
                decoder_input = torch.cat([decoder_input, predicted_token_id], dim=1)

                # eos token ? pokud ano -> break
                if predicted_token_id.item() == eos_idx:
                    break

            # dekódování predikovaného shrnutí
            # skipneme první token (SOS)
            generated_ids = decoder_input[0, 1:].tolist()
            predicted_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # dekódování originálního shrnutí (labels)
            original_ids_for_decoding = batch['labels'].tolist()
            original_text = tokenizer.decode(original_ids_for_decoding, skip_special_tokens=True)

            print("-" * 30)
            print(f"Příklad {i + 1}:")
            print(f"Predicted: {predicted_text}")
            print(f"Original:  {original_text}")

            individual_scores = scorer.score(original_text, predicted_text)
            r1 = individual_scores['rouge1'].fmeasure
            r2 = individual_scores['rouge2'].fmeasure
            r1_list.append(r1)
            r2_list.append(r2)
            print(f"ROUGE-1 F1: {r1:.4f}")
            print(f"ROUGE-2 F1: {r2:.4f}")

    avg_r1 = sum(r1_list) / len(r1_list)
    avg_r2 = sum(r2_list) / len(r2_list)
    print(f"Průměrný ROUGE-1 F1: {avg_r1:.4f}")
    print(f"Průměrný ROUGE-2 F1: {avg_r2:.4f}")


if __name__ == "__main__":
    main()