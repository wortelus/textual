import torch
import torch.nn as nn
import os

from rouge_score import rouge_scorer

from transformer.model.seq2seq import Seq2SeqTransformer
from transformer.processing.dataset import load_samsum
from transformer.processing.tokenizer import get_tokenizer

from transformer.const import NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, FFN_HID_DIM, DROPOUT, \
    MAX_MODEL_SEQ_LEN, MAX_SRC_SEQ_LEN, MAX_TGT_SEQ_LEN


def beam_search_decode(model: Seq2SeqTransformer,
                       src_tensor: torch.Tensor,
                       tokenizer,
                       device: torch.device,
                       max_len: int,
                       beam_width: int,
                       sos_idx: int,
                       eos_idx: int,
                       pad_idx: int,
                       length_penalty_alpha: float = 0.7) -> str:
    model.eval()

    with torch.no_grad():
        # inicializace paprsků: seznam n-tic
        # sekvence_tenzor má tvar (1, aktualni_delka)
        active_beams = [(torch.tensor([[sos_idx]], dtype=torch.long, device=device), 0.0)]

        src_padding_mask = (src_tensor == pad_idx)
        memory = model.encode(src_tensor)

        completed = []
        for _ in range(max_len):
            # pokud už nejsou aktivni beams
            if not active_beams:
                break

            all_next_candidates = []

            # prochazeni active beams
            for current_seq_tensor, current_score in active_beams:
                last_token_in_seq = current_seq_tensor[0, -1].item()

                if last_token_in_seq == eos_idx:
                    completed.append((current_seq_tensor, current_score))
                    continue

                # kauzalni maska pro dekoder na yet-generated sekvenci
                tgt_seq_len = current_seq_tensor.size(1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)

                # pruchod dekoderem
                decoded = model.decode(
                    tgt=current_seq_tensor,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask
                )

                # pravděpodobnosti všech slovníkových slov pro následující token
                last_token_hidden_state = decoded[:, -1, :]
                # linární vrstva pro generaci pravděpodobností + softmax
                logits = model.generator(last_token_hidden_state)
                log_probs = torch.log_softmax(logits, dim=-1)

                # top-k tokeny (k=beam_width)
                top_log_probs, top_token_ids = torch.topk(log_probs, beam_width, dim=-1)

                # rožšíření aktuálního paprsku o beam_width
                for k in range(beam_width):
                    next_token_id = top_token_ids[0, k].unsqueeze(0).unsqueeze(0)  # (1,1)
                    token_log_prob = top_log_probs[0, k].item()

                    new_seq_tensor = torch.cat([current_seq_tensor, next_token_id], dim=1)
                    new_score = current_score + token_log_prob
                    all_next_candidates.append((new_seq_tensor, new_score))


            if not all_next_candidates:
                break

            # oddělíme kandidáty, které končí EOS nebo dosáhly max_len, od těch aktivních
            active_candidates_for_next_step = []
            for seq, score in all_next_candidates:
                if seq[0, -1].item() == eos_idx or seq.size(1) >= max_len + 1:  # +1 kvůli SOS
                    completed.append((seq, score))
                else:
                    active_candidates_for_next_step.append((seq, score))

            if not active_candidates_for_next_step:  # všichni kandidáti byli dokončeni
                break

            ordered_active_candidates = sorted(active_candidates_for_next_step, key=lambda x: x[1], reverse=True)
            active_beams = ordered_active_candidates[:beam_width]  # aktualizace aktivních paprsků

        # pokud active_beam dosáhl max_len nebo eos tokenu, přidáme je do completed
        for seq, score in active_beams:
            completed.append((seq, score))

        if not completed:
            print("Chyba: Žádné dokončené sekvence.")
            return ""

        # Výběr nejlepší hypotézy z dokončených pomocí délkové penalizace
        # Normalizujeme skóre délkou: score / (length^alpha)
        # Chceme maximalizovat toto normalizované skóre.
        best_hypothesis = max(
            completed,
            key=lambda x: x[1] / ((x[0].size(1) - 1) ** length_penalty_alpha + 1e-6)  # -1 pro SOS, +1e-6 pro stabilitu
        )

        best_hypothesis_tensor, _ = best_hypothesis

        # dekódování nejlepší hypotézy
        generated_ids = best_hypothesis_tensor[0, 1:].cpu().tolist()
        return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    BEAM_WIDTH = 3
    LENGTH_PENALTY_ALPHA = 0.7

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
    print(f"Používám zařízení: {device}")

    checkpoint_file_path = "first_train/checkpoint_last_epoch_7_val_loss_4.4270.pth"
    if not os.path.isfile(checkpoint_file_path):
        print(f"Chyba: Soubor checkpointu nenalezen na {checkpoint_file_path}")
        return

    print(f"Načítám model z {checkpoint_file_path}")
    checkpoint = torch.load(checkpoint_file_path, map_location=device)

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
        print("Stav modelu načten z 'model_state_dict'.")
    else:
        try:
            model.load_state_dict(checkpoint)
            print("Stav modelu načten (načten přímo state_dict).")
        except Exception as e:
            print(f"Chyba při načítání stavu modelu: {e}")
            print(
                "Ujistěte se, že checkpoint obsahuje klíč 'model_state_dict' nebo že soubor obsahuje přímo state_dict.")
            return

    model.eval()

    sos_idx = pad_idx
    eos_idx = tokenizer.eos_token_id

    print(f"Používaná ID tokenů: SOS={sos_idx}, EOS={eos_idx}, PAD={pad_idx}")
    print(f"Parametry Beam Search: BEAM_WIDTH={BEAM_WIDTH}, LENGTH_PENALTY_ALPHA={LENGTH_PENALTY_ALPHA}")

    r1_list = []
    r2_list = []

    print("\nSpouštím Beam Search Inference a výpočet ROUGE...")
    with torch.no_grad():
        for i, batch in enumerate(tokenized_test):
            src = batch['input_ids'].unsqueeze(0).to(device)  # Přidání batch dimenze

            predicted_text = beam_search_decode(
                model=model,
                src_tensor=src,
                tokenizer=tokenizer,
                device=device,
                max_len=MAX_TGT_SEQ_LEN,
                beam_width=BEAM_WIDTH,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
                length_penalty_alpha=LENGTH_PENALTY_ALPHA
            )

            original_ids_for_decoding = batch['labels'].tolist()
            original_text = tokenizer.decode(original_ids_for_decoding, skip_special_tokens=True)

            print("-" * 30)
            print(f"Příklad {i + 1}/{len(tokenized_test)}:")
            print(f"Predicted: {predicted_text}")
            print(f"Original:  {original_text}")

            individual_scores = scorer.score(original_text, predicted_text)
            r1 = individual_scores['rouge1'].fmeasure
            r2 = individual_scores['rouge2'].fmeasure
            r1_list.append(r1)
            r2_list.append(r2)
            print(f"ROUGE-1 F1: {r1:.4f}")
            print(f"ROUGE-2 F1: {r2:.4f}")

    if r1_list and r2_list:
        avg_r1 = sum(r1_list) / len(r1_list)
        avg_r2 = sum(r2_list) / len(r2_list)
        print("-" * 30)
        print(f"\nPrůměrné ROUGE skóre na {len(r1_list)} vzorcích:")
        print(f"  Průměrný ROUGE-1 F1: {avg_r1:.4f}")
        print(f"  Průměrný ROUGE-2 F1: {avg_r2:.4f}")
    else:
        print("Nebyly vypočítány žádné ROUGE hodnoty.")

    print("\nBeam Search Inference a výpočet ROUGE dokončeny.")


if __name__ == "__main__":
    main()
