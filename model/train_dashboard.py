import pickle

import torch
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from pythainlp.tokenize import word_tokenize

from .model import SentimentModel


# ==========================================
# CONFIG
# ==========================================

MAX_LEN = 50
BATCH_SIZE = 32

EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_CLASSES = 4

LEARNING_RATE = 0.001
EPOCHS = 5


# ==========================================
# LABEL
# ==========================================

# LABEL_TO_ID = {
#     "pos": 0,
#     "neu": 1,
#     "neg": 2,
#     "q": 3
# }


# ==========================================
# LOAD DATASET
# ==========================================

def load_data():

    dataset = load_dataset(
        "pythainlp/wisesight_sentiment"
    )

    return dataset


# ==========================================
# BUILD VOCAB
# ==========================================

def build_vocab(dataset):

    from collections import Counter

    counter = Counter()

    for item in dataset:

        tokens = word_tokenize(
            item["texts"],
            engine="newmm"
        )

        counter.update(tokens)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word in counter:

        vocab[word] = len(vocab)

    return vocab


# ==========================================
# ENCODE
# ==========================================

def encode_text(text, vocab):

    tokens = word_tokenize(
        text,
        engine="newmm"
    )

    ids = []

    for token in tokens:

        if token in vocab:
            ids.append(vocab[token])

        else:
            ids.append(
                vocab["<UNK>"]
            )

    return ids


# ==========================================
# PADDING
# ==========================================

def pad_sequence(ids):

    if len(ids) < MAX_LEN:

        ids = ids + [
            0
        ] * (
            MAX_LEN - len(ids)
        )

    else:

        ids = ids[:MAX_LEN]

    return ids


# ==========================================
# DATASET CLASS
# ==========================================

class SentimentDataset(Dataset):

    def __init__(
        self,
        dataset,
        vocab
    ):

        self.dataset = dataset
        self.vocab = vocab

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        text = item["texts"]

        label = item["category"]

        ids = encode_text(
            text,
            self.vocab
        )

        ids = pad_sequence(ids)

        return (
            torch.tensor(
                ids,
                dtype=torch.long
            ),
            torch.tensor(
                # LABEL_TO_ID[label],
                label,
                dtype=torch.long
            )
        )

def evaluate_model(
    model,
    data_loader,
    criterion
):

    model.eval()

    total_loss = 0

    correct = 0

    total = 0


    with torch.no_grad():

        for x, y in data_loader:

            output = model(x)

            loss = criterion(
                output,
                y
            )

            total_loss += loss.item()


            predictions = output.argmax(
                dim=1
            )


            correct += (
                predictions == y
            ).sum().item()


            total += y.size(0)


    average_loss = (
        total_loss /
        len(data_loader)
    )


    accuracy = (
        correct /
        total
    )


    return average_loss, accuracy
# ==========================================
# TRAINING GENERATOR
# ==========================================

def train_model():

    # -----------------------------
    # Load dataset
    # -----------------------------

    dataset = load_data()

    # -----------------------------
    # Build vocabulary
    # -----------------------------

    vocab = build_vocab(
        dataset["train"]
    )

    # -----------------------------
    # Dataset
    # -----------------------------

    train_dataset = SentimentDataset(
        dataset["train"],
        vocab
    )
    val_dataset = SentimentDataset(
    dataset["validation"],
    vocab
    )

    test_dataset = SentimentDataset(
        dataset["test"],
        vocab
    )

    # -----------------------------
    # DataLoader
    # -----------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -----------------------------
    # Model
    # -----------------------------

    model = SentimentModel(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    )

    # -----------------------------
    # Loss
    # -----------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # Optimizer
    # -----------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------
    # Training
    # -----------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        correct = 0

        total = 0

        for x, y in train_loader:

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(
                output,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            predictions = (
                output.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)

        accuracy = (
            correct / total
        )

        average_loss = (
            total_loss /
            len(train_loader)
        )
        
        val_loss, val_accuracy = evaluate_model(
            model,
            val_loader,
            criterion
        )

        yield {
            "epoch": epoch + 1,
            "loss": average_loss,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }
        
        test_loss, test_accuracy = evaluate_model(
            model,
            test_loader,
            criterion
        )
        

    # -----------------------------
    # Save model
    # -----------------------------

    torch.save(
        model.state_dict(),
        "model/weights.pth"
    )

    # -----------------------------
    # Save vocabulary
    # -----------------------------

    with open(
        "model/vocab.pkl",
        "wb"
    ) as f:

        pickle.dump(
            vocab,
            f
        )
        
    yield {
        "type": "final",
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }