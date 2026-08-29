from pathlib import Path
import shutil


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATASET_ROOT = Path(r"C:\Datasets\UCF-Crime")

SPLIT_ROOT = Path(
    r"C:\Users\Akanchha\Desktop\Multimodal anamoly and accident detection"
)

OUTPUT_ROOT = DATASET_ROOT / "processed"


# --------------------------------------------------
# Classes we currently want
# --------------------------------------------------

TARGET_CLASSES = {
    "Arson",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Normal_Videos_event"
}


# --------------------------------------------------
# Number of videos per split
# --------------------------------------------------

TRAIN_COUNT = 30
VAL_COUNT = 5
TEST_COUNT = 3


# --------------------------------------------------
# Training list files
# --------------------------------------------------

TRAIN_LISTS = [
    "train_001",
    "train_002",
    "train_003",
    "train_004"
]


# --------------------------------------------------
# Read all training videos
# --------------------------------------------------

def load_training_videos():

    videos = []

    for list_name in TRAIN_LISTS:

        list_path = (
            SPLIT_ROOT /
            "dataset_splits" /
            list_name
        )

        if not list_path.exists():
            print(f"Missing split file: {list_path}")
            continue

        with open(
            list_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                class_name = line.split("/")[0]

                if class_name in TARGET_CLASSES:

                    videos.append(line)

    return videos


# --------------------------------------------------
# Find actual video
# --------------------------------------------------

def find_video(relative_path):

    possible_locations = [
        DATASET_ROOT / "Part1" / relative_path,
        DATASET_ROOT / "Part2" / relative_path,
        DATASET_ROOT / "Part3" / relative_path,
        DATASET_ROOT / "Normal" / relative_path
    ]

    for path in possible_locations:

        if path.exists():
            return path

    return None


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    videos = load_training_videos()

    print(
        f"Found {len(videos)} training entries"
    )

    class_videos = {}

    for relative_path in videos:

        class_name = relative_path.split("/")[0]

        if class_name not in class_videos:
            class_videos[class_name] = []

        class_videos[class_name].append(
            relative_path
        )

    for class_name, video_list in class_videos.items():

        print(
            f"{class_name}: "
            f"{len(video_list)} videos"
        )


if __name__ == "__main__":
    main()