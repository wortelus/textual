from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from transformer.processing.preprocess import preprocess_function


def load_samsum(train_size: int,
                val_size: int,
                test_size: int,
                tokenizer,
                max_src_seq_len: int,
                max_tgt_seq_len: int,
                pad_idx: int,
                seed: int = 10,
                stats: bool = False):
    print("načítání datasetu")
    try:
        samsum_dataset_full = load_dataset("samsum")
    except Exception as e:
        print(f"chyba při načítání SAMSum: {e}. Ukončuji.")
        exit()

    print("tokenizace datasetu...")
    small_train_dataset = samsum_dataset_full["train"].shuffle(seed=seed)
    if train_size and train_size > 0:
        small_train_dataset = small_train_dataset.select(range(train_size))
    small_val_dataset = samsum_dataset_full["validation"].shuffle(seed=seed)
    if val_size and val_size > 0:
        small_val_dataset = small_val_dataset.select(range(val_size))
    small_test_dataset = samsum_dataset_full["test"].shuffle(seed=seed)
    if test_size and test_size > 0:
        small_test_dataset = small_test_dataset.select(range(test_size))

    if stats:
        # statistiky datasetu
        dataset_stats(small_train_dataset, small_val_dataset, small_test_dataset)

    preprocess_fn = partial(preprocess_function, tokenizer=tokenizer, max_src_seq_len=max_src_seq_len,
                            max_tgt_seq_len=max_tgt_seq_len,
                            pad_idx=pad_idx)

    tokenized_train = (small_train_dataset
                       .map(preprocess_fn, batched=True, remove_columns=["dialogue", "summary", "id"]))
    tokenized_val = (small_val_dataset
                     .map(preprocess_fn, batched=True, remove_columns=["dialogue", "summary", "id"]))
    tokenized_test = (small_test_dataset
                      .map(preprocess_fn, batched=True, remove_columns=["dialogue", "summary", "id"]))

    # set_format na PyTorch tenzory
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "labels"])
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "labels"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "labels"])

    print("načítání datasetu dokončeno")
    return tokenized_train, tokenized_val, tokenized_test


def dataset_stats(train, val, test, plot=True):
    print("Statistiky datasetu:")
    print(f"Počet trénovacích vzorků: {len(train)}")
    print(f"Počet validačních vzorků: {len(val)}")
    print(f"Počet testovacích vzorků: {len(test)}")

    # max a průměrná délka dialogu a shrnutí
    max_dialogue_length = max(len(item["dialogue"]) for item in train)
    avg_dialogue_length = sum(len(item["dialogue"]) for item in train) / len(train)

    max_summary_length = max(len(item["summary"]) for item in train)
    avg_summary_length = sum(len(item["summary"]) for item in train) / len(train)

    print(f"Maximální délka dialogu: {max_dialogue_length}")
    print(f"Průměrná délka dialogu: {avg_dialogue_length:.2f}")
    print(f"Maximální délka shrnutí: {max_summary_length}")
    print(f"Průměrná délka shrnutí: {avg_summary_length:.2f}")

    # histogramy
    if not plot:
        return

    sns.histplot([len(item["dialogue"]) for item in train], bins=50)
    plt.title("Histogram délky dialogu")
    plt.xlabel("Délka dialogu")
    plt.ylabel("Počet vzorků")
    plt.show()

    sns.histplot([len(item["summary"]) for item in train], bins=50)
    plt.title("Histogram délky shrnutí")
    plt.xlabel("Délka shrnutí")
    plt.ylabel("Počet vzorků")
    plt.show()