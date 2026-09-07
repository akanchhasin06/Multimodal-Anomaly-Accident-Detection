import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_DIR = os.path.join(BASE_DIR, "dataset_features_v2")
OUTPUT_DIR = os.path.join(BASE_DIR, "sequences_v2")

FEATURES = [
    "dx",
    "dy",
    "speed",
    "acceleration",
    "direction_change",
    "jerk"
]

SEQ_LEN = 30
STRIDE = 10


def generate_sequences(split):
    split_dir = os.path.join(FEATURE_DIR, split)
    X = []
    y = []

    for category in os.listdir(split_dir):
        category_dir = os.path.join(split_dir, category)

        if not os.path.isdir(category_dir):
            continue

        label = 0 if category == "Normal" else 1

        for file in os.listdir(category_dir):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(category_dir, file)
            df = pd.read_csv(file_path)

            for track_id, track in df.groupby("track_id"):
                track = track.sort_values("frame")

                values = track[FEATURES].values

                if len(values) < SEQ_LEN:
                    continue

                for start in range(0, len(values) - SEQ_LEN + 1, STRIDE):
                    sequence = values[start:start + SEQ_LEN]

                    X.append(sequence)
                    y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(
        os.path.join(OUTPUT_DIR, f"X_{split}.npy"),
        X
    )

    np.save(
        os.path.join(OUTPUT_DIR, f"y_{split}.npy"),
        y
    )

    print(f"\n{split.upper()}")
    print("Sequences:", len(X))
    print("Normal:", np.sum(y == 0))
    print("Anomaly:", np.sum(y == 1))
    print("Shape:", X.shape)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        generate_sequences(split)

    print("\nSequence generation complete!")