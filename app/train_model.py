from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Akanchha\Desktop\Multimodal anamoly and accident detection"
)

SEQUENCE_DIR = BASE_DIR / "sequences"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

INPUT_SIZE = 4
HIDDEN_SIZE = 128
NUM_LAYERS = 2

DROPOUT = 0.3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    X_train = np.load(SEQUENCE_DIR / "X_train.npy")
    y_train = np.load(SEQUENCE_DIR / "y_train.npy")

    X_val = np.load(SEQUENCE_DIR / "X_val.npy")
    y_val = np.load(SEQUENCE_DIR / "y_val.npy")

    print("Data loaded:")
    print(f"Train: {X_train.shape}")
    print(f"Val  : {X_val.shape}")

    return (
        X_train,
        y_train,
        X_val,
        y_val
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(
    X_train,
    X_val
):

    # Calculate statistics ONLY from training data.
    # This prevents validation information leaking into training.

    mean = X_train.mean(
        axis=(0, 1),
        keepdims=True
    )

    std = X_train.std(
        axis=(0, 1),
        keepdims=True
    )

    std = np.maximum(
        std,
        1e-6
    )

    X_train = (
        X_train - mean
    ) / std

    X_val = (
        X_val - mean
    ) / std

    # Save normalization statistics for inference later.

    np.save(
        MODEL_DIR / "feature_mean.npy",
        mean
    )

    np.save(
        MODEL_DIR / "feature_std.npy",
        std
    )

    return X_train, X_val


# ============================================================
# GRU MODEL
# ============================================================

class AnomalyGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        # x:
        # (batch, sequence_length, features)

        output, hidden = self.gru(x)

        # Last timestep representation
        last_output = output[:, -1, :]

        logits = self.classifier(
            last_output
        )

        return logits.squeeze(1)


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for X, y in loader:

        X = X.to(DEVICE)
        y = y.float().to(DEVICE)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item() * len(y)
        )

        predictions = (
            torch.sigmoid(logits) >= 0.5
        )

        correct += (
            predictions == y.bool()
        ).sum().item()

        total += len(y)

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.float().to(DEVICE)

            logits = model(X)

            loss = criterion(
                logits,
                y
            )

            total_loss += (
                loss.item() * len(y)
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            )

            correct += (
                predictions == y.bool()
            ).sum().item()

            total += len(y)

            tp += (
                ((predictions == 1) & (y == 1))
            ).sum().item()

            tn += (
                ((predictions == 0) & (y == 0))
            ).sum().item()

            fp += (
                ((predictions == 1) & (y == 0))
            ).sum().item()

            fn += (
                ((predictions == 0) & (y == 1))
            ).sum().item()

    precision = tp / max(
        tp + fp,
        1
    )

    recall = tp / max(
        tp + fn,
        1
    )

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-8)
    )

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ANOMALY DETECTION - GRU TRAINING")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val
    ) = load_data()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    X_train, X_val = normalize_data(
        X_train,
        X_val
    )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_train = torch.from_numpy(
        X_train.astype(np.float32)
    )

    y_train = torch.from_numpy(
        y_train.astype(np.int64)
    )

    X_val = torch.from_numpy(
        X_val.astype(np.float32)
    )

    y_val = torch.from_numpy(
        y_val.astype(np.int64)
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    val_dataset = TensorDataset(
        X_val,
        y_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------

    normal_count = (
        y_train == 0
    ).sum().item()

    anomaly_count = (
        y_train == 1
    ).sum().item()

    # Give more importance to the minority class.
    pos_weight = (
        normal_count /
        anomaly_count
    )

    print(
        f"\nNormal samples : {normal_count:,}"
    )

    print(
        f"Anomaly samples: {anomaly_count:,}"
    )

    print(
        f"Positive weight: {pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AnomalyGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    print(
        f"\nModel parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            device=DEVICE,
            dtype=torch.float32
        )
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_f1 = 0.0

    patience = 5
    epochs_without_improvement = 0

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_metrics["f1"]
        )

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Acc : {train_acc:.4f}"
        )

        print(
            f"Val Loss  : {val_metrics['loss']:.4f}"
        )

        print(
            f"Val Acc   : {val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Prec  : {val_metrics['precision']:.4f}"
        )

        print(
            f"Val Recall: {val_metrics['recall']:.4f}"
        )

        print(
            f"Val F1    : {val_metrics['f1']:.4f}"
        )

        print(
            f"Confusion: "
            f"TP={val_metrics['tp']} "
            f"TN={val_metrics['tn']} "
            f"FP={val_metrics['fp']} "
            f"FN={val_metrics['fn']}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_size":
                        INPUT_SIZE,

                    "hidden_size":
                        HIDDEN_SIZE,

                    "num_layers":
                        NUM_LAYERS,

                    "dropout":
                        DROPOUT,

                    "best_val_f1":
                        best_f1
                },

                MODEL_DIR / "best_anomaly_gru.pt"
            )

            print(
                "✓ Best model saved!"
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "\nEarly stopping."
            )

            break

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation F1: {best_f1:.4f}"
    )

    print(
        f"\nModel saved at:"
        f"\n{MODEL_DIR / 'best_anomaly_gru.pt'}"
    )


if __name__ == "__main__":
    main()