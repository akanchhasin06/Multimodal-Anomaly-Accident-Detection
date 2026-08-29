from pathlib import Path
import random
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(r"C:\Datasets\UCF-Crime")

OUTPUT_ROOT = DATASET_ROOT / "processed"

RANDOM_SEED = 42


# ============================================================
# DATASET CLASSES
# ============================================================

CLASSES = {
    "Abuse": "Abuse",
    "Explosion": "Explosion",
    "Fighting": "Fighting",
    "RoadAccidents": "RoadAccidents",
    "Normal": "Normal_Videos_for_Event_Recognition",
}


# ============================================================
# FIND VIDEOS
# ============================================================

def find_videos(folder_name):

    videos = []

    for video in DATASET_ROOT.rglob("*.mp4"):

        # For anomaly classes:
        # folder name is the parent directory.
        if video.parent.name.lower() == folder_name.lower():
            videos.append(video)

    return videos


# ============================================================
# CALCULATE SPLIT
# ============================================================

def calculate_split(total):

    # Very small class
    if total < 20:

        train = int(total * 0.67)
        val = int(total * 0.17)
        test = total - train - val

    else:

        train = min(30, int(total * 0.70))
        val = min(5, int(total * 0.15))

        test = total - train - val

        # Keep test manageable
        if test > 10:
            test = 5

    return train, val, test


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    for split in ["train", "val", "test"]:

        for class_name in CLASSES:

            folder = (
                OUTPUT_ROOT /
                split /
                class_name
            )

            folder.mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# COPY FILES
# ============================================================

def copy_split(videos, class_name):

    random.shuffle(videos)

    total = len(videos)

    train_count, val_count, test_count = calculate_split(total)

    train_videos = videos[
        :train_count
    ]

    val_videos = videos[
        train_count:
        train_count + val_count
    ]

    test_videos = videos[
        train_count + val_count:
        train_count + val_count + test_count
    ]

    splits = {
        "train": train_videos,
        "val": val_videos,
        "test": test_videos,
    }

    for split, split_videos in splits.items():

        destination = (
            OUTPUT_ROOT /
            split /
            class_name
        )

        for video in split_videos:

            target = destination / video.name

            shutil.copy2(
                video,
                target
            )

    print(
        f"{class_name}: "
        f"Total={total} | "
        f"Train={len(train_videos)} | "
        f"Val={len(val_videos)} | "
        f"Test={len(test_videos)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("UCF-CRIME DATASET SPLITTER")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    create_directories()

    for class_name, folder_name in CLASSES.items():

        print(
            f"\nSearching for {class_name}..."
        )

        videos = find_videos(folder_name)

        if len(videos) == 0:

            print(
                f"ERROR: No videos found for {class_name}"
            )

            continue

        copy_split(
            videos,
            class_name
        )

    print("\n" + "=" * 60)
    print("DATASET SPLIT COMPLETE")
    print("=" * 60)

    print(
        f"\nDataset created at:\n{OUTPUT_ROOT}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()