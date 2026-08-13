import pandas as pd
import numpy as np
from pathlib import Path


SEQUENCE_LENGTH = 30

FEATURE_COLUMNS = [
    "x",
    "y",
    "dx",
    "dy",
    "speed",
    "acceleration"
]


def create_sequences(csv_path, output_dir):

    df = pd.read_csv(csv_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_count = 0

    for track_id, track_df in df.groupby("track_id"):

        track_df = track_df.sort_values("frame")

        features = track_df[FEATURE_COLUMNS].values

        if len(features) < SEQUENCE_LENGTH:
            continue

        for start in range(
            0,
            len(features) - SEQUENCE_LENGTH + 1
        ):

            sequence = features[
                start:start + SEQUENCE_LENGTH
            ]

            output_path = (
                output_dir /
                f"sequence_{sequence_count:05d}.npy"
            )

            np.save(output_path, sequence)

            sequence_count += 1

    print(
        f"Created {sequence_count} sequences"
    )


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    create_sequences(
        BASE_DIR / "features.csv",
        BASE_DIR / "sequences"
    )