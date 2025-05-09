# většinu parametrů jsem nastavil per
# ref: https://medium.com/we-talk-data/in-depth-guide-on-pytorchs-nn-transformer-901ad061a195

TOKENIZER_NAME = "t5-small"

# Dimenze embeddingu (d_model v nn.Transformer)
EMB_SIZE = 256

# Počet pozornostních hlav (musí dělit EMB_SIZE)
NHEAD = 8

# dimenze skryté vrstvy v feed-forward síti uvnitř Transformeru
FFN_HID_DIM = 1024

# Počet vrstev v encoderu a decoderu
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

# dropout rate pro vrstvy Transformeru
# použito pro layers/PositionalEncoding a nn.Transformer
DROPOUT = 0.1

# maximální délky pro tokenizaci (as per samsum)
MAX_SRC_SEQ_LEN = 512
MAX_TGT_SEQ_LEN = 300

# tyto hodnoty by měly být větší nebo rovny MAX_SRC_SEQ_LEN a MAX_TGT_SEQ_LEN,
# které použijete při tokenizaci
MAX_MODEL_SEQ_LEN = 512