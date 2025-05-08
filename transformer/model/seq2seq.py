import torch
from torch import nn

from transformer.const import MAX_MODEL_SEQ_LEN
# naše vrstvy
from transformer.model.layers import TokenEmbedding, PositionalEncoding, T5_PADDING_TOKEN_ID


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
                 batch_first: bool = True):
        super(Seq2SeqTransformer, self).__init__()

        self.batch_first = batch_first

        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)

        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_model_seq_len)

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
        # pozice, které jsou True nebo -inf, jsou maskované
        # torch.triu nám dává pravou symetrickou část matice
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=1)) == 1
        return mask  # Shape: [sz, sz]

    def _create_padding_mask(self, sequence: torch.Tensor, pad_idx: int) -> torch.Tensor:
        # Vstup: sequence má tvar [batch_size, seq_len]
        # Výstup: maska má tvar [batch_size, seq_len], kde True znamená, že pozice je padding.
        return (sequence == pad_idx)  # Shape: [batch_size, seq_len]

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
        src_padding_mask = self._create_padding_mask(src, T5_PADDING_TOKEN_ID)  # Shape: [batch_size, src_seq_len]
        tgt_padding_mask = self._create_padding_mask(tgt, T5_PADDING_TOKEN_ID)  # Shape: [batch_size, tgt_seq_len]
        # memory_key_padding_mask je pro výstup enkodéru (memory), takže je to src_padding_mask
        memory_key_padding_mask = src_padding_mask

        # Embedding a poziční encoding
        # Pokud batch_first=False (default pro nn.Transformer), museli bychom transponovat
        # src a tgt před embeddingem nebo emb po PE. S batch_first=True je to přímočařejší.
        src_emb = self.positional_encoding(self.src_tok_emb(src))  # [batch_size, src_seq_len, emb_size]
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))  # [batch_size, tgt_seq_len, emb_size]

        # Průchod Transformerem
        # nn.Transformer s batch_first=True očekává:
        # src: (N, S, E), tgt: (N, T, E)
        # src_key_padding_mask: (N, S)
        # tgt_key_padding_mask: (N, T)
        # memory_key_padding_mask: (N, S)
        # tgt_mask (causal): (T, T)

        transformer_out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,  # kauzální maska pro dekodér
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        # transformer_out má tvar: [batch_size, tgt_seq_len, emb_size]

        return self.generator(transformer_out)  # [batch_size, tgt_seq_len, tgt_vocab_size]

    # Metody pro inferenci (generování) - pokud byste je chtěli implementovat později
    def encode(self, src: torch.Tensor, src_pad_idx: int = T5_PADDING_TOKEN_ID):
        src_padding_mask = self._create_padding_mask(src, src_pad_idx)
        src_emb = self.positional_encoding(self.src_tok_emb(src))

        # Pro batch_first=True, encoder očekává src: (N,S,E), src_key_padding_mask: (N,S)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor,  # kauzální maska
               tgt_pad_idx: int = T5_PADDING_TOKEN_ID,
               memory_key_padding_mask: torch.Tensor = None):  # padding maska pro memory (z enkodéru)

        tgt_padding_mask = self._create_padding_mask(tgt, tgt_pad_idx)
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