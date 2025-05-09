import torch
from torch import nn

from transformer.const import MAX_MODEL_SEQ_LEN
from transformer.model.layers import TokenEmbedding, PositionalEncoding


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int,
                 dropout: float,
                 pad_idx: int,
                 max_model_seq_len: int = MAX_MODEL_SEQ_LEN,
                 batch_first: bool = True):
        super(Seq2SeqTransformer, self).__init__()
        self.pad_idx = pad_idx

        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_model_seq_len)
        self.src_embedding = TokenEmbedding(src_vocab_size, emb_size, padding_idx=pad_idx)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, emb_size, padding_idx=pad_idx)

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )

        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def _generate_square_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        # maska pro self-attention v dekodéru.
        # buď použijeme toto
        # nebo torch.nn.Transformer.generate_square_subsequent_mask
        # ref: https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer.generate_square_subsequent_mask

        mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.float), diagonal=0)
        # nuly v horní části matice na -inf
        mask = mask.masked_fill(mask == 0, float('-inf'))
        # jedničky v dolní části matice na 0
        mask = mask.masked_fill(mask == 1, float(0.0))

        return mask

    def _create_padding_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        # [batch_size, seq_len]
        # maska, kde True znamená, že pozice je padding
        return sequence == self.pad_idx

    def forward(self,
                # Tvar: [batch_size, src_seq_len]
                src: torch.Tensor,
                # Tvar: [batch_size, tgt_seq_len] (posunutý cíl)
                tgt: torch.Tensor):
        # zařízení, na kterém jsou data
        device = src.device

        # casual (look-ahead) maska pro dekodér
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)  # Shape: [tgt_seq_len, tgt_seq_len]

        # padding masky
        # [batch_size, src_seq_len]
        src_padding_mask = self._create_padding_mask(src)
        # [batch_size, tgt_seq_len]
        tgt_padding_mask = self._create_padding_mask(tgt)

        memory_key_padding_mask = src_padding_mask

        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        transformer_out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # [batch_size, tgt_seq_len, emb_size]
        return self.generator(transformer_out)

    # Metody pro inferenci (generování) - pokud byste je chtěli implementovat později
    def encode(self, src: torch.Tensor):
        src_padding_mask = self._create_padding_mask(src)
        src_emb = self.positional_encoding(self.src_tok_emb(src))

        # Pro batch_first=True, encoder očekává src: (N,S,E), src_key_padding_mask: (N,S)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor,  # kauzální maska
               memory_key_padding_mask: torch.Tensor = None):
        # padding maska pro memory (z enkodéru)
        tgt_padding_mask = self._create_padding_mask(tgt)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        # Pro batch_first=True, decoder očekává:
        # tgt: (N,T,E), memory: (N,S,E)
        # tgt_mask (kauzální): (T,T)
        # tgt_key_padding_mask: (N,T)
        # memory_key_padding_mask: (N,S)
        return self.transformer.decoder(tgt_emb, memory,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)
