from pathlib import Path
import csv

from feature_extractor import extract_features


# ============================================================
# PATHS
# ============================================================

DATASET_ROOT = Path(r"C:\Datasets\UCF-Crime\processed")

OUTPUT_ROOT = Path(
    r"C:\Users\Akanchha\Desktop\Multimodal anamoly and accident detection"
)

FEATURES_ROOT = OUTPUT_ROOT / "dataset_features"


# ============================================================
# DATASET SPLITS
# ============================================================

SPLITS = [
    "train",
    "val",
    "test"
]


# ============================================================
# EXTRACT FEATURES FOR ONE VIDEO
# ============================================================

def process_video(video_path, output_path):

    print(f"\nProcessing:")
    print(video_path)

    try:

        extract_features(
            str(video_path),
            str(output_path)
        )

        print("Feature extraction successful.")

        return True

    except Exception as e:

        print(
            f"ERROR processing {video_path}"
        )

        print(e)

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DATASET FEATURE EXTRACTION")
    print("=" * 60)

    FEATURES_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    total_videos = 0
    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process train / val / test separately
    # --------------------------------------------------------

    for split in SPLITS:

        split_path = DATASET_ROOT / split

        output_split_path = (
            FEATURES_ROOT / split
        )

        output_split_path.mkdir(
            parents=True,
            exist_ok=True
        )

        print("\n" + "=" * 60)
        print(f"PROCESSING {split.upper()} DATA")
        print("=" * 60)

        # ----------------------------------------------------
        # Each class
        # ----------------------------------------------------

        for class_folder in split_path.iterdir():

            if not class_folder.is_dir():
                continue

            class_name = class_folder.name

            output_class_path = (
                output_split_path /
                class_name
            )

            output_class_path.mkdir(
                parents=True,
                exist_ok=True
            )

            videos = list(
                class_folder.glob("*.mp4")
            )

            print(
                f"\n{class_name}: "
                f"{len(videos)} videos"
            )

            # ------------------------------------------------
            # Process every video
            # ------------------------------------------------

            for video_path in videos:

                total_videos += 1

                output_file = (
                    output_class_path /
                    f"{video_path.stem}.csv"
                )

                # Skip if already processed
                if output_file.exists():

                    print(
                        f"Already processed: "
                        f"{video_path.name}"
                    )

                    successful += 1
                    continue

                success = process_video(
                    video_path,
                    output_file
                )

                if success:
                    successful += 1
                else:
                    failed += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Total videos : {total_videos}"
    )

    print(
        f"Successful   : {successful}"
    )

    print(
        f"Failed       : {failed}"
    )

    print(
        f"\nFeatures saved at:\n{FEATURES_ROOT}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()