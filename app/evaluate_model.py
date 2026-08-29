from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Akanchha\Desktop\Multimodal anamoly and accident detection"
)

SEQUENCE_DIR = BASE_DIR / "sequences"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_anomaly_gru.pt"


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 256
INPUT_SIZE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
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
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        last_output = output[:, -1, :]

        logits = self.classifier(last_output)

        return logits.squeeze(1)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ANOMALY DETECTION - TEST EVALUATION")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test = np.load(
        SEQUENCE_DIR / "X_test.npy"
    )

    y_test = np.load(
        SEQUENCE_DIR / "y_test.npy"
    )

    print("\nTest data:")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    # --------------------------------------------------------
    # Load normalization statistics
    # --------------------------------------------------------

    mean = np.load(
        MODEL_DIR / "feature_mean.npy"
    )

    std = np.load(
        MODEL_DIR / "feature_std.npy"
    )

    X_test = (
        X_test - mean
    ) / std

    # --------------------------------------------------------
    # Tensor dataset
    # --------------------------------------------------------

    X_test = torch.from_numpy(
        X_test.astype(np.float32)
    )

    y_test_tensor = torch.from_numpy(
        y_test.astype(np.int64)
    )

    test_dataset = TensorDataset(
        X_test,
        y_test_tensor
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Load model checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = AnomalyGRU(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"\nBest validation F1: "
        f"{checkpoint['best_val_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    all_probabilities = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(DEVICE)

            logits = model(X)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).long()

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                y.numpy()
            )

    y_true = np.asarray(
        all_targets
    )

    y_pred = np.asarray(
        all_predictions
    )

    y_prob = np.asarray(
        all_probabilities
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC   : {pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(
        "\n                 Predicted"
    )

    print(
        "              Normal  Anomaly"
    )

    print(
        f"Actual Normal  {cm[0][0]:7d}  {cm[0][1]:7d}"
    )

    print(
        f"Actual Anomaly {cm[1][0]:7d}  {cm[1][1]:7d}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Normal",
                "Anomaly"
            ],
            zero_division=0
        )
    )

    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()