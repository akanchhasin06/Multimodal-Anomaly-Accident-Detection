import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight


# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEQUENCE_DIR = os.path.join(BASE_DIR, "sequences_v2")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# Configuration
# =========================

INPUT_SIZE = 6
HIDDEN_SIZE = 128
NUM_LAYERS = 2

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# =========================
# Load data
# =========================

X_train = np.load(os.path.join(SEQUENCE_DIR, "X_train.npy"))
y_train = np.load(os.path.join(SEQUENCE_DIR, "y_train.npy"))

X_val = np.load(os.path.join(SEQUENCE_DIR, "X_val.npy"))
y_val = np.load(os.path.join(SEQUENCE_DIR, "y_val.npy"))

print("Train:", X_train.shape)
print("Val:", X_val.shape)


# =========================
# Normalize features
# =========================

mean = X_train.reshape(-1, INPUT_SIZE).mean(axis=0)
std = X_train.reshape(-1, INPUT_SIZE).std(axis=0)

std[std == 0] = 1

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

np.save(
    os.path.join(MODEL_DIR, "feature_mean_v2.npy"),
    mean
)

np.save(
    os.path.join(MODEL_DIR, "feature_std_v2.npy"),
    std
)


# =========================
# PyTorch datasets
# =========================

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
)

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


# =========================
# GRU Model
# =========================

class AnomalyGRU(nn.Module):

    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        output, _ = self.gru(x)

        last_output = output[:, -1, :]

        return self.classifier(last_output).squeeze(1)


model = AnomalyGRU().to(DEVICE)


# =========================
# Loss
# =========================

normal_count = np.sum(y_train == 0)
anomaly_count = np.sum(y_train == 1)

pos_weight = torch.tensor(
    normal_count / anomaly_count,
    dtype=torch.float32
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# =========================
# Training
# =========================

best_f1 = 0
patience = 5
epochs_without_improvement = 0

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_anomaly_gru_v2.pt"
)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(logits, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)

            probs = torch.sigmoid(logits)

            preds = (probs >= 0.5).cpu().numpy()

            predictions.extend(preds)
            actual.extend(y_batch.numpy())

    val_f1 = f1_score(actual, predictions)

    scheduler.step(val_f1)

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    # -------------------------
    # Save best model
    # -------------------------

    if val_f1 > best_f1:

        best_f1 = val_f1

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        epochs_without_improvement = 0

        print("  ✓ Best model saved")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:

        print("Early stopping.")

        break


print("\nTraining complete!")
print("Best validation F1:", best_f1)
print("Model:", MODEL_PATH)