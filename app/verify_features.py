from pathlib import Path
import pandas as pd


FEATURES_ROOT = Path(
    r"C:\Users\Akanchha\Desktop\Multimodal anamoly and accident detection\dataset_features"
)

EXPECTED_COLUMNS = [
    "track_id",
    "dx",
    "dy",
    "speed",
    "acceleration"
]


def main():

    print("=" * 60)
    print("FEATURE DATASET VERIFICATION")
    print("=" * 60)

    total_csv = 0
    valid_csv = 0
    empty_csv = 0
    invalid_csv = 0

    total_rows = 0

    problems = []

    for split in ["train", "val", "test"]:

        split_path = FEATURES_ROOT / split

        print(f"\n{split.upper()}")

        for class_dir in sorted(split_path.iterdir()):

            if not class_dir.is_dir():
                continue

            csv_files = list(class_dir.glob("*.csv"))

            print(
                f"  {class_dir.name}: "
                f"{len(csv_files)} CSV files"
            )

            for csv_file in csv_files:

                total_csv += 1

                try:
                    df = pd.read_csv(csv_file)

                    # Check empty
                    if df.empty:
                        empty_csv += 1
                        problems.append(
                            f"EMPTY: {csv_file}"
                        )
                        continue

                    # Check columns
                    missing = [
                        col for col in EXPECTED_COLUMNS
                        if col not in df.columns
                    ]

                    if missing:
                        invalid_csv += 1
                        problems.append(
                            f"MISSING {missing}: {csv_file}"
                        )
                        continue

                    # Check NaN
                    if df[EXPECTED_COLUMNS].isnull().any().any():

                        invalid_csv += 1
                        problems.append(
                            f"NaN values: {csv_file}"
                        )
                        continue

                    valid_csv += 1
                    total_rows += len(df)

                except Exception as e:

                    invalid_csv += 1

                    problems.append(
                        f"ERROR: {csv_file} -> {e}"
                    )

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    print(f"Total CSV files : {total_csv}")
    print(f"Valid CSV files : {valid_csv}")
    print(f"Empty CSV files : {empty_csv}")
    print(f"Invalid CSVs    : {invalid_csv}")
    print(f"Total rows      : {total_rows:,}")

    print("\n" + "=" * 60)

    if problems:

        print("PROBLEMS FOUND")
        print("=" * 60)

        for problem in problems[:20]:
            print(problem)

        if len(problems) > 20:
            print(
                f"\n...and {len(problems) - 20} more."
            )

    else:

        print("ALL FEATURE FILES ARE VALID.")
        print("=" * 60)


if __name__ == "__main__":
    main()