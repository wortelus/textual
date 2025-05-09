def preprocess_function(data, tokenizer, max_src_seq_len, max_tgt_seq_len, pad_idx):
    dialogues = [dialogue for dialogue in data["dialogue"]]
    summaries = [summary for summary in data["summary"]]

    model_inputs = tokenizer(
        dialogues,
        max_length=max_src_seq_len,
        padding="max_length",
        truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels_output = tokenizer(
            summaries,
            max_length=max_tgt_seq_len,
            padding="max_length",
            truncation=True
        )
    model_inputs["labels"] = labels_output["input_ids"]

    decoder_input_ids_batch = []
    for label_ids in model_inputs["labels"]:
        shifted_labels = [pad_idx] + label_ids[:-1]
        decoder_input_ids_batch.append(shifted_labels)
    model_inputs["decoder_input_ids"] = decoder_input_ids_batch

    return model_inputs