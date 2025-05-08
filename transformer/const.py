EMB_SIZE = 256          # Dimenze embeddingu (d_model v nn.Transformer)
NHEAD = 4               # Počet pozornostních hlav (musí dělit EMB_SIZE)
FFN_HID_DIM = 512       # Dimenze skryté vrstvy v feed-forward síti uvnitř Transformeru
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1

# maximální délky sekvencí pro model
# (pro PositionalEncoding, pokud jej neimplementujete dynamicky)
# tyto hodnoty by měly být větší nebo rovny MAX_SRC_SEQ_LEN a MAX_TGT_SEQ_LEN,
# které použijete při tokenizaci
MAX_MODEL_SEQ_LEN = 512