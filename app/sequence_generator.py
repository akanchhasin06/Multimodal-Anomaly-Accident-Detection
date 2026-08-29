from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

FEATURES_ROOT = Path(
    r"C:\Users\Akanchha\Desktop\Multimodal anamoly and accident detection\dataset_features"
)

SEQUENCE_ROOT = Path(
    r"C:\Users\Akanchha\Desktop\Multimodal anamoly and accident detection\sequences"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 30
STRIDE = 10

FEATURE_COLUMNS = [
    "dx",
    "dy",
    "speed",
    "acceleration"
]


# ============================================================
# LABELS
# ============================================================

ANOMALY_CLASSES = {
    "Abuse",
    "Explosion",
    "Fighting",
    "RoadAccidents"
}


# ============================================================
# CREATE SEQUENCES FOR ONE VIDEO
# ============================================================

def create_sequences(df):

    sequences = []

    # --------------------------------------------------------
    # Process each tracked object independently
    # --------------------------------------------------------

    for track_id, track_df in df.groupby("track_id"):

        track_df = track_df.reset_index(drop=True)

        # Not enough observations for one sequence
        if len(track_df) < SEQUENCE_LENGTH:
            continue

        features = track_df[FEATURE_COLUMNS].values

        # ----------------------------------------------------
        # Sliding window
        # ----------------------------------------------------

        for start in range(
            0,
            len(features) - SEQUENCE_LENGTH + 1,
            STRIDE
        ):

            end = start + SEQUENCE_LENGTH

            sequence = features[start:end]

            sequences.append(sequence)

    return sequences


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split):

    split_path = FEATURES_ROOT / split

    all_sequences = []
    all_labels = []

    video_count = 0
    sequence_count = 0

    print("\n" + "=" * 60)
    print(f"PROCESSING {split.upper()}")
    print("=" * 60)

    # --------------------------------------------------------
    # Loop through classes
    # --------------------------------------------------------

    for class_dir in sorted(split_path.iterdir()):

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        # Normal = 0
        # Everything else = 1
        label = 1 if class_name in ANOMALY_CLASSES else 0

        csv_files = sorted(
            class_dir.glob("*.csv")
        )

        print(
            f"\n{class_name}: "
            f"{len(csv_files)} videos"
        )

        # ----------------------------------------------------
        # Process videos
        # ----------------------------------------------------

        for csv_file in csv_files:

            try:

                df = pd.read_csv(csv_file)

                sequences = create_sequences(df)

                for sequence in sequences:

                    all_sequences.append(sequence)
                    all_labels.append(label)

                video_count += 1
                sequence_count += len(sequences)

            except Exception as e:

                print(
                    f"ERROR: {csv_file.name}"
                )

                print(e)

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    if all_sequences:

        X = np.asarray(
            all_sequences,
            dtype=np.float32
        )

        y = np.asarray(
            all_labels,
            dtype=np.int64
        )

    else:

        X = np.empty(
            (0, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)),
            dtype=np.float32
        )

        y = np.empty(
            (0,),
            dtype=np.int64
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    SEQUENCE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        SEQUENCE_ROOT / f"X_{split}.npy",
        X
    )

    np.save(
        SEQUENCE_ROOT / f"y_{split}.npy",
        y
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    normal_count = int(
        np.sum(y == 0)
    )

    anomaly_count = int(
        np.sum(y == 1)
    )

    print("\n" + "-" * 60)
    print(f"{split.upper()} SUMMARY")
    print("-" * 60)

    print(f"Videos processed : {video_count}")
    print(f"Sequences        : {len(X)}")
    print(f"Normal sequences : {normal_count}")
    print(f"Anomaly sequences: {anomaly_count}")
    print(f"Shape            : {X.shape}")

    return X, y


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TEMPORAL SEQUENCE GENERATOR")
    print("=" * 60)

    print(
        f"\nSequence length : {SEQUENCE_LENGTH}"
    )

    print(
        f"Stride          : {STRIDE}"
    )

    print(
        f"Features        : {FEATURE_COLUMNS}"
    )

    # --------------------------------------------------------
    # Process all splits
    # --------------------------------------------------------

    for split in ["train", "val", "test"]:

        process_split(split)

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SEQUENCE GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"\nSequences saved at:\n{SEQUENCE_ROOT}"
    )


if __name__ == "__main__":
    main()