import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

from transformer.const import NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, FFN_HID_DIM, DROPOUT, \
    MAX_MODEL_SEQ_LEN
from transformer.model.seq2seq import Seq2SeqTransformer
from transformer.run_fit import SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, PAD_IDX, MAX_SRC_SEQ_LEN, tokenizer, DEVICE

MODEL_PATH = "best_samsum_transformer_from_scratch.pth"  # Cesta k vašemu uloženému modelu


# --- Funkce pro načtení modelu ---
def load_trained_model(model_path, device):
    # Inicializujeme architekturu modelu se stejnými parametry jako při tréninku
    model = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        emb_size=EMB_SIZE,
        num_heads=NHEAD,
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        dim_feedforward=FFN_HID_DIM,
        dropout=DROPOUT,  # Dropout je součástí definice, ale .eval() ho vypne
        max_model_seq_len=MAX_MODEL_SEQ_LEN,
        batch_first=True,
        pad_idx=PAD_IDX  # Předáme PAD_IDX do modelu
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Přepnutí modelu do evaluačního módu (vypne dropout atd.)
    print(f"Model načten z {model_path} a přepnut do evaluačního módu.")
    return model


# --- Funkce pro přípravu testovacích dat ---
def prepare_test_data(tokenizer_instance, device):
    print("Načítám testovací data SAMSum...")
    try:
        # Načteme pouze testovací split
        test_dataset_raw = load_dataset("samsum", split="test")
        # Pro rychlejší testování můžeme omezit počet vzorků:
        # test_dataset_raw = load_dataset("samsum", split="test").select(range(100))
    except Exception as e:
        print(f"Chyba při načítání SAMSum test: {e}.")
        return None, None

    dialogues = [item['dialogue'] for item in test_dataset_raw]
    reference_summaries = [item['summary'] for item in test_dataset_raw]

    print(f"Tokenizuji {len(dialogues)} dialogů...")
    # Tokenizujeme pouze vstupy (dialogy)
    inputs_tokenized = tokenizer_instance(
        dialogues,
        max_length=MAX_SRC_SEQ_LEN,
        padding="max_length",  # Nebo 'longest' pokud chcete dynamický padding v collate_fn
        truncation=True,
        return_tensors="pt"  # Přímo PyTorch tenzory
    )

    # Vytvoříme PyTorch Dataset pro snadnější batchování
    # Musíme předat i originální texty pro evaluaci a tisk
    # Přidáme 'id' pro případné párování, i když ho zde nepoužijeme přímo
    ids = [item['id'] for item in test_dataset_raw]

    # Vytvoříme list slovníků, který pak převedeme na Dataset
    processed_data = []
    for i in range(len(dialogues)):
        processed_data.append({
            'id': ids[i],
            'input_ids': inputs_tokenized['input_ids'][i],
            'attention_mask': inputs_tokenized['attention_mask'][i],
            'reference_dialogue': dialogues[i],  # Pro tisk
            'reference_summary': reference_summaries[i]  # Pro ROUGE a tisk
        })

    test_dataset_processed = Dataset.from_list(processed_data)
    test_dataset_processed.set_format(type="torch", columns=["input_ids", "attention_mask"])

    print("Příprava testovacích dat dokončena.")
    return test_dataset_processed, reference_summaries  # Vracíme Dataset a seznam referenčních textů


# --- Funkce pro generování shrnutí (greedy decode) ---
@torch.no_grad()  # Dekorátor pro vypnutí výpočtu gradientů v této funkci
def generate_summaries(model, dataloader, tokenizer_instance, max_gen_len, device):
    model.eval()  # Ujistíme se, že model je v eval módu
    all_generated_summaries = []
    all_dialogues_for_print = []
    all_references_for_print = []

    print("Začínám generování shrnutí...")
    for batch_idx, batch in enumerate(dataloader):
        src = batch['input_ids'].to(device)
        # `attention_mask` z tokenizeru nám říká, co NENÍ padding.
        # Naše `_create_padding_mask` očekává, že True je padding.
        # Takže `encoder_padding_mask = (src == PAD_IDX)` nebo `~batch['attention_mask'].bool()`
        # Pokud attention_mask je 1 pro non-pad a 0 pro pad, pak:
        encoder_padding_mask = (batch['attention_mask'] == 0).to(device)

        # Uložíme si i texty pro pozdější tisk (pokud jsou v batchi)
        if 'reference_dialogue' in batch:
            all_dialogues_for_print.extend(batch['reference_dialogue'])
        if 'reference_summary' in batch:
            all_references_for_print.extend(batch['reference_summary'])

        # Enkódování vstupu - pouze jednou na začátku pro každý batch
        memory = model.encode(src)  # Použije self.pad_idx interně

        # Inicializace dekodéru: začneme BOS tokenem (pro T5 je to pad_token_id)
        current_batch_size = src.size(0)
        # Tvar: (batch_size, 1)
        generated_sequence_ids = torch.full(
            (current_batch_size, 1),
            tokenizer_instance.pad_token_id,  # BOS token pro T5
            dtype=torch.long,
            device=device
        )

        for _ in range(max_gen_len - 1):  # -1 protože už máme BOS token
            # Získáme logity pro další token
            # Použijeme upravenou `decode_step` metodu
            logits = model.decode_step(generated_sequence_ids, memory, encoder_padding_mask)
            # logits má tvar (batch_size, vocab_size)

            # Greedy výběr dalšího tokenu
            next_token_ids = torch.argmax(logits, dim=-1)  # Tvar: (batch_size)

            # Přidání predikovaného tokenu k sekvenci
            generated_sequence_ids = torch.cat(
                [generated_sequence_ids, next_token_ids.unsqueeze(1)], dim=1
            )  # Tvar: (batch_size, current_len + 1)

            # Zastavení, pokud všechny sekvence v batchi vygenerovaly EOS
            # nebo dosáhly maximální délky (to je řízeno vnějším cyklem)
            # Musíme zkontrolovat, zda VŠECHNY `next_token_ids` jsou EOS.
            if torch.all(next_token_ids == tokenizer_instance.eos_token_id).item():
                break

        # Dekódování ID na text
        # `skip_special_tokens=True` odstraní BOS, EOS, PAD atd.
        decoded_batch_summaries = tokenizer_instance.batch_decode(generated_sequence_ids, skip_special_tokens=True)
        all_generated_summaries.extend(decoded_batch_summaries)

        if batch_idx % 5 == 0:  # Logování každých 5 batchů
            print(f"  Generován batch {batch_idx + 1}/{len(dataloader)}")

    print("Generování dokončeno.")
    # Pokud jsme nenačetli texty dialogů a referencí do dataloaderu, vrátíme jen generované
    if not all_dialogues_for_print:
        all_dialogues_for_print = [None] * len(all_generated_summaries)
    if not all_references_for_print:
        all_references_for_print = [None] * len(all_generated_summaries)

    return all_generated_summaries, all_dialogues_for_print, all_references_for_print


# --- Hlavní funkce pro testování ---
def test_model():
    # Načtení modelu
    model = load_trained_model(MODEL_PATH, DEVICE)

    # Příprava testovacích dat
    # Funkce prepare_test_data vrací (Dataset pro DataLoader, seznam referenčních textů shrnutí)
    test_dataset, reference_summaries_list = prepare_test_data(tokenizer, DEVICE)
    if test_dataset is None:
        return

    test_dataloader = DataLoader(test_dataset, batch_size=8)  # Menší batch size pro inferenci může být OK

    # Generování shrnutí
    generated_summaries, dialogues_for_print, references_for_print = generate_summaries(
        model,
        test_dataloader,
        tokenizer,
        MAX_TGT_SEQ_LEN_INFERENCE,  # max délka generovaného shrnutí
        DEVICE
    )

    # Výpis několika příkladů
    print("\n--- Příklady generovaných shrnutí ---")
    num_examples_to_print = 5
    for i in range(min(num_examples_to_print, len(generated_summaries))):
        print(f"\nDialog {i + 1}:")
        print(dialogues_for_print[i] if dialogues_for_print[
            i] else "N/A")  # Změna z test_dataset.dataset['reference_dialogue']
        print(f"\nReferenční shrnutí {i + 1}:")
        print(references_for_print[i] if references_for_print[i] else reference_summaries_list[i])  # Změna
        print(f"\nGenerované shrnutí {i + 1}:")
        print(generated_summaries[i])
        print("-" * 30)

    # Evaluace pomocí ROUGE (pokud jsou k dispozici referenční shrnutí)
    if reference_summaries_list:
        print("\n--- Evaluace ROUGE ---")
        try:
            rouge_metric = evaluate.load('rouge')
            rouge_results = rouge_metric.compute(
                predictions=generated_summaries,
                references=reference_summaries_list  # Použijeme seznam textů
            )
            print(f"ROUGE-1: {rouge_results['rouge1'] * 100:.2f}")
            print(f"ROUGE-2: {rouge_results['rouge2'] * 100:.2f}")
            print(f"ROUGE-L: {rouge_results['rougeL'] * 100:.2f}")
            print(f"ROUGE-Lsum: {rouge_results['rougeLsum'] * 100:.2f}")
        except Exception as e:
            print(f"Nepodařilo se spočítat ROUGE skóre: {e}")
            print("Ujistěte se, že máte nainstalované knihovny 'evaluate' a 'rouge_score'.")
            print("Např. pip install evaluate rouge_score")


if __name__ == '__main__':
    # Ujistěte se, že máte uložený model na cestě MODEL_PATH
    # a že definice tříd modelu jsou dostupné.
    test_model()