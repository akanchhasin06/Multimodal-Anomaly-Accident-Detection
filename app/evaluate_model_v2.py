import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEQUENCE_DIR = os.path.join(BASE_DIR, "sequences_v2")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 6
HIDDEN_SIZE = 128
NUM_LAYERS = 2


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


# Load test data
X_test = np.load(os.path.join(SEQUENCE_DIR, "X_test.npy"))
y_test = np.load(os.path.join(SEQUENCE_DIR, "y_test.npy"))

# Load training normalization
mean = np.load(
    os.path.join(MODEL_DIR, "feature_mean_v2.npy")
)

std = np.load(
    os.path.join(MODEL_DIR, "feature_std_v2.npy")
)

X_test = (X_test - mean) / std

X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

# Load model
model = AnomalyGRU().to(DEVICE)

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "best_anomaly_gru_v2.pt"),
        map_location=DEVICE
    )
)

model.eval()

# Inference
with torch.no_grad():
    logits = model(X_test)
    probabilities = torch.sigmoid(logits).cpu().numpy()

predictions = (probabilities >= 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)
pr_auc = average_precision_score(y_test, probabilities)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions
).ravel()

print("\n===== V2 TEST RESULTS =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print("\nFalse Positive Rate:",
      f"{fp / (fp + tn):.4f}")