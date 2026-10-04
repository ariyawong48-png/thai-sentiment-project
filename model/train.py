import torch.nn as nn
import pickle
from datasets import load_dataset
from collections import Counter
from pythainlp.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset, DataLoader
from model import SentimentModel
# หรือใช้ Relative Import:
# from .model import SentimentModel

# ==================================
# 1. Load Dataset
# ==================================

dataset = load_dataset(
    "pythainlp/wisesight_sentiment"
)


print(dataset)


# ==================================
# 2. Dataset information
# ==================================

print("\nColumns:")
print(dataset["train"].column_names)

print("\nNumber of data:")
print("Train:", len(dataset["train"]))
print("Validation:", len(dataset["validation"]))
print("Test:", len(dataset["test"]))


# ==================================
# 3. Label mapping
# ==================================

label_to_id = {
    "pos": 0,
    "neu": 1,
    "neg": 2,
    "q": 3
}

id_to_label = {
    0: "pos",
    1: "neu",
    2: "neg",
    3: "q"
}


print("\nLabel mapping:")
print(label_to_id)


# ==================================
# 4. Tokenization example
# ==================================

text = "ร้านนี้อาหารอร่อยมาก"

tokens = word_tokenize(
    text,
    engine="newmm"
)

print("\nTokenization:")
print(tokens)


# ==================================
# 5. Build vocabulary
# ==================================

def build_vocab(dataset):

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


vocab = build_vocab(
    dataset["train"]
)


print("\nVocabulary size:")
print(len(vocab))


# ==================================
# 6. Encode text
# ==================================

def encode_text(
    text,
    vocab
):

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


encoded = encode_text(
    text,
    vocab
)

print("\nEncoded:")
print(encoded)


# ==================================
# 7. Padding
# ==================================

def pad_sequence(
    ids,
    max_len
):

    if len(ids) < max_len:

        ids = ids + [0] * (
            max_len - len(ids)
        )

    else:

        ids = ids[:max_len]

    return ids


MAX_LEN = 50

padded = pad_sequence(
    encoded,
    MAX_LEN
)


print("\nPadded:")
print(padded)
print("Length:", len(padded))

class SentimentDataset(Dataset):

    def __init__(
        self,
        dataset,
        vocab,
        # label_to_id,
        max_len
    ):

        self.dataset = dataset
        self.vocab = vocab
        # self.label_to_id = label_to_id
        self.max_len = max_len
    
    def __getitem__(self, index):

        item = self.dataset[index]

        text = item["texts"]

        label = item["category"]

        ids = encode_text(
            text,
            self.vocab
        )

        ids = pad_sequence(
            ids,
            self.max_len
        )

        return (
            torch.tensor(
                ids,
                dtype=torch.long
            ),
            torch.tensor(
                # self.label_to_id[label],
                label,
                dtype=torch.long
            )
        )
        
    def __len__(self):

        return len(self.dataset)

train_dataset = SentimentDataset(
    dataset["train"],
    vocab,
    # label_to_id,
    MAX_LEN
)

val_dataset = SentimentDataset(
    dataset["validation"],
    vocab,
    # label_to_id,
    MAX_LEN
)

test_dataset = SentimentDataset(
    dataset["test"],
    vocab,
    # label_to_id,
    MAX_LEN
)

x, y = train_dataset[0]

print("\nX:")
print(x)

print("\nY:")
print(y)

print("\nX shape:")
print(x.shape)

print("\nY shape:")
print(y.shape)

BATCH_SIZE = 32
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
x_batch, y_batch = next(
    iter(train_loader)
)
def evaluate(model, data_loader, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for x_batch, y_batch in data_loader:

            output = model(x_batch)

            loss = criterion(
                output,
                y_batch
            )

            total_loss += loss.item()

            prediction = output.argmax(
                dim=1
            )

            correct += (
                prediction == y_batch
            ).sum().item()

            total += y_batch.size(0)

    accuracy = correct / total

    average_loss = (
        total_loss / len(data_loader)
    )

    return average_loss, accuracy

print("\nBatch X:")
print(x_batch.shape)

print("\nBatch Y:")
print(y_batch.shape)

VOCAB_SIZE = len(vocab)

EMBED_DIM = 64

HIDDEN_DIM = 64

NUM_CLASSES = 4

model = SentimentModel(
    VOCAB_SIZE,
    EMBED_DIM,
    HIDDEN_DIM,
    NUM_CLASSES
)
output = model(x_batch)
print("\nModel output:")
print(output.shape)

prediction = output.argmax(
    dim=1
)
print(prediction)
# nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
EPOCHS = 5
for epoch in range(EPOCHS):

    # ==========================
    # TRAIN
    # ==========================

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x_batch, y_batch in train_loader:

        optimizer.zero_grad()

        output = model(
            x_batch
        )

        loss = criterion(
            output,
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        prediction = output.argmax(
            dim=1
        )

        correct += (
            prediction == y_batch
        ).sum().item()

        total += y_batch.size(0)

    train_accuracy = correct / total

    train_loss = (
        total_loss /
        len(train_loader)
    )


    # ==========================
    # VALIDATION
    # ==========================

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion
    )


    # ==========================
    # PRINT
    # ==========================

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.4f}"
    )
print("\nVocabulary size:")
print(len(vocab))

print("\nBatch shape:")
print(x_batch.shape)

print("\nLabel shape:")
print(y_batch.shape)

print("\nModel:")
print(model)

print("\nOutput shape:")
print(output.shape)

test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    criterion
)

print("\n========== TEST RESULT ==========")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)
torch.save(
    model.state_dict(),
    "model/weights.pth"
)
print("\nModel saved!")

loaded_model = SentimentModel(
    VOCAB_SIZE,
    EMBED_DIM,
    HIDDEN_DIM,
    NUM_CLASSES
)
loaded_model.load_state_dict(
    torch.load(
        "model/weights.pth",
        weights_only=True
    )
)
loaded_model.eval()
print(
    "\nModel loaded successfully!"
)
x_test, y_test = next(
    iter(test_loader)
)
with torch.no_grad():

    output = loaded_model(
        x_test
    )

    prediction = output.argmax(
        dim=1
    )
print("\nPrediction:")
print(prediction[:10])

print("\nActual:")
print(y_test[:10])

with open(
    "model/vocab.pkl",
    "wb"
) as f:

    pickle.dump(
        vocab,
        f
    )
    
print("Vocabulary saved!")