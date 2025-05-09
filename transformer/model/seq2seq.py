import math

import torch
from torch import nn

from transformer.const import MAX_MODEL_SEQ_LEN
# naše vrstvy
from transformer.model.layers import TokenEmbedding, PositionalEncoding


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_model_seq_len: int = MAX_MODEL_SEQ_LEN,  # pro PositionalEncoding
                 pad_idx: int = 0,
                 batch_first: bool = True):
        super(Seq2SeqTransformer, self).__init__()

        self.emb_size = emb_size
        self.pad_idx = pad_idx
        self.batch_first = batch_first

        # self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size, padding_idx=pad_idx)
        # self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size, padding_idx=pad_idx)

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
            batch_first=batch_first  # DŮLEŽITÉ!
        )

        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        # maska pro self-attention v dekodéru.
        # pozice, které jsou True/1, jsou maskované
        # torch.triu nám dává pravou symetrickou část matice
        # mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=1)) == 1
        # return mask  # Shape: [sz, sz]

        mask = torch.tril(torch.ones(sz, sz) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        return mask.to(device)

    def _create_padding_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        # Vstup: sequence má tvar [batch_size, seq_len]
        # Výstup: maska má tvar [batch_size, seq_len], kde True znamená, že pozice je padding.
        return (sequence == self.pad_idx)  # Shape: [batch_size, seq_len]

    def forward(self,
                # Tvar: [batch_size, src_seq_len]
                src: torch.Tensor,
                # Tvar: [batch_size, tgt_seq_len] (posunutý cíl)
                tgt: torch.Tensor,
                ):
        device = src.device  # Získáme zařízení, na kterém jsou data

        # Masky
        # Causal (look-ahead) mask pro dekodér
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)  # Shape: [tgt_seq_len, tgt_seq_len]

        # Padding masky
        src_padding_mask = self._create_padding_mask(src)  # Shape: [batch_size, src_seq_len]
        tgt_padding_mask = self._create_padding_mask(tgt)  # Shape: [batch_size, tgt_seq_len]
        memory_key_padding_mask = src_padding_mask

        src = self.src_embedding(src) * math.sqrt(self.emb_size)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.emb_size)
        src_emb = self.positional_encoding(src)
        tgt_emb = self.positional_encoding(tgt)

        transformer_out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        # transformer_out má tvar: [batch_size, tgt_seq_len, emb_size]

        return self.generator(transformer_out)  # [batch_size, tgt_seq_len, tgt_vocab_size]

    # Metody pro inferenci (generování) - pokud byste je chtěli implementovat později
    def encode(self, src: torch.Tensor):
        src_padding_mask = self._create_padding_mask(src)
        src_emb = self.positional_encoding(self.src_tok_emb(src))

        # Pro batch_first=True, encoder očekává src: (N,S,E), src_key_padding_mask: (N,S)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor,  # kauzální maska
               memory_key_padding_mask: torch.Tensor = None):  # padding maska pro memory (z enkodéru)

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
