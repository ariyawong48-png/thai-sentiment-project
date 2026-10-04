import pickle
from pathlib import Path

import torch

from pythainlp.tokenize import word_tokenize

from .model import SentimentModel


# ==========================================
# PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "weights.pth"
VOCAB_PATH = BASE_DIR / "vocab.pkl"


# ==========================================
# MODEL CONFIG
# ==========================================

EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_CLASSES = 4
MAX_LEN = 50


# ==========================================
# LABEL
# ==========================================

ID_TO_LABEL = {
    0: "Positive",
    1: "Neutral",
    2: "Negative",
    3: "Question",
}


# ==========================================
# LOAD VOCAB
# ==========================================

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)


# ==========================================
# CREATE MODEL
# ==========================================

model = SentimentModel(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES
)


# ==========================================
# LOAD WEIGHTS
# ==========================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)


# ==========================================
# EVALUATION MODE
# ==========================================

model.eval()


# ==========================================
# ENCODE TEXT
# ==========================================

def encode_text(text):

    tokens = word_tokenize(
        text,
        engine="newmm"
    )

    ids = []

    for token in tokens:

        if token in vocab:
            ids.append(vocab[token])

        else:
            ids.append(vocab["<UNK>"])

    return ids


# ==========================================
# PADDING
# ==========================================

def pad_sequence(ids):

    if len(ids) < MAX_LEN:

        ids = ids + [0] * (
            MAX_LEN - len(ids)
        )

    else:

        ids = ids[:MAX_LEN]

    return ids


# ==========================================
# PREDICT
# ==========================================

def predict_text(text):

    # แปลงข้อความเป็นตัวเลข
    ids = encode_text(text)

    # padding
    ids = pad_sequence(ids)

    # Tensor
    x = torch.tensor(
        [ids],
        dtype=torch.long
    )

    # ไม่คำนวณ gradient
    with torch.no_grad():

        output = model(x)

        # แปลงเป็น probability
        probabilities = torch.softmax(
            output,
            dim=1
        )

        # หาค่าที่มากที่สุด
        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        # confidence
        confidence = probabilities[
            0,
            prediction
        ].item()

    return {
        "label": ID_TO_LABEL[prediction],
        "confidence": confidence
    }