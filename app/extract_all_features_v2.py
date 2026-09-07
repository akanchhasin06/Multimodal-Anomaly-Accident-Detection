from pathlib import Path

from feature_extractor_v2 import extract_features


BASE_DIR = Path(
    r"C:\Users\Akanchha\Desktop\Multimodal anamoly and accident detection"
)

PROCESSED_DIR = Path(
    r"C:\Datasets\UCF-Crime\processed"
)

OUTPUT_DIR = BASE_DIR / "dataset_features_v2"


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    splits = [
        "train",
        "val",
        "test"
    ]

    total = 0
    successful = 0
    failed = 0

    for split in splits:

        split_dir = PROCESSED_DIR / split

        if not split_dir.exists():
            continue

        for category_dir in split_dir.iterdir():

            if not category_dir.is_dir():
                continue

            output_category_dir = (
                OUTPUT_DIR /
                split /
                category_dir.name
            )

            output_category_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            videos = list(
                category_dir.glob("*.mp4")
            )

            for video_path in videos:

                total += 1

                output_csv = (
                    output_category_dir /
                    f"{video_path.stem}.csv"
                )

                # Skip videos that were already processed.
                if output_csv.exists() and output_csv.stat().st_size > 0:

                    print(
                        f"\n[{total}] SKIPPING "
                        f"{split}/{category_dir.name}/"
                        f"{video_path.name}"
                    )

                    successful += 1
                    continue

                print(
                    f"\n[{total}] "
                    f"{split}/{category_dir.name}/"
                    f"{video_path.name}"
                )

                try:

                    extract_features(
                        str(video_path),
                        str(output_csv)
                    )

                    successful += 1

                except Exception as e:

                    failed += 1

                    print(
                    f"FAILED: {e}"
                    )

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION V2 COMPLETE")
    print("=" * 60)

    print(
        f"Total videos : {total}"
    )

    print(
        f"Successful   : {successful}"
    )

    print(
        f"Failed       : {failed}"
    )

    print(
        f"\nFeatures saved at:"
    )

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()