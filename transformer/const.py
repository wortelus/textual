TOKENIZER_NAME = "t5-small"

# Dimenze embeddingu (d_model v nn.Transformer)
EMB_SIZE = 256

# Počet pozornostních hlav (musí dělit EMB_SIZE)
NHEAD = 4

# Dimenze skryté vrstvy v feed-forward síti uvnitř Transformeru
FFN_HID_DIM = 512
# Počet vrstev v encoderu a decoderu
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
# Dropout pro vrstvy Transformeru
# použito pro layers/PositionalEncoding
DROPOUT = 0.1

# tyto hodnoty by měly být větší nebo rovny MAX_SRC_SEQ_LEN a MAX_TGT_SEQ_LEN,
# které použijete při tokenizaci
MAX_MODEL_SEQ_LEN = 512