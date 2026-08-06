from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


INPUT_FOLDER = Path("outputs/newscore")
OUTPUT_FILE = INPUT_FOLDER / "overall_estimator_k100_and_above.csv"

MINIMUM_K = 100


def extract_k_from_filename(filename: str) -> int | None:
    """
    Extract k from filenames containing patterns such as:

        k[100]
        k[150]
        k[200]

    Returns None when no valid k value is found.
    """
    match = re.search(
        r"k\[(\d+)\]",
        filename,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return int(match.group(1))


def main() -> None:
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Input folder not found:\n"
            f"{INPUT_FOLDER.resolve()}"
        )

    all_workbooks = sorted(
        INPUT_FOLDER.glob("*_estimator_averages.xlsx")
    )

    if not all_workbooks:
        raise FileNotFoundError(
            f"No estimator workbooks were found in:\n"
            f"{INPUT_FOLDER.resolve()}"
        )

    selected_workbooks: list[tuple[Path, int]] = []

    for workbook in all_workbooks:
        k_value = extract_k_from_filename(
            workbook.name
        )

        if (
            k_value is not None
            and k_value >= MINIMUM_K
        ):
            selected_workbooks.append(
                (workbook, k_value)
            )

    if not selected_workbooks:
        raise FileNotFoundError(
            f"No workbooks with k >= {MINIMUM_K} "
            f"were found in:\n"
            f"{INPUT_FOLDER.resolve()}"
        )

    print(
        f"Found {len(selected_workbooks)} workbook(s) "
        f"with k >= {MINIMUM_K}."
    )

    summaries: list[pd.DataFrame] = []

    for file_number, (
        workbook,
        k_value,
    ) in enumerate(
        selected_workbooks,
        start=1,
    ):
        print(
            f"[{file_number}/{len(selected_workbooks)}] "
            f"Reading {workbook.name} "
            f"(k={k_value})"
        )

        try:
            summary = pd.read_excel(
                workbook,
                sheet_name="Estimator Averages",
            )

        except Exception as error:
            print(
                f"Skipping {workbook.name}: {error}"
            )
            continue

        required_columns = [
            "Estimator",
            "Average p",
            "Average CI Length",
            "Quality Score",
            "Bias",
            "Variance",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in summary.columns
        ]

        if missing_columns:
            print(
                f"Skipping {workbook.name}: "
                f"missing columns {missing_columns}"
            )
            continue

        selected = summary[
            required_columns
        ].copy()

        selected["Source File"] = workbook.name
        selected["k"] = k_value

        summaries.append(selected)

    if not summaries:
        raise RuntimeError(
            "None of the selected workbooks could be processed."
        )

    combined = pd.concat(
        summaries,
        ignore_index=True,
    )

    numeric_columns = [
        "Average p",
        "Average CI Length",
        "Quality Score",
        "Bias",
        "Variance",
    ]

    for column in numeric_columns:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce",
        )

    overall = (
        combined
        .groupby(
            "Estimator",
            as_index=False,
        )
        .agg(
            Average_Coverage=(
                "Average p",
                "mean",
            ),
            Average_CI_Length=(
                "Average CI Length",
                "mean",
            ),
            Average_Quality_Score=(
                "Quality Score",
                "mean",
            ),
            Average_Bias=(
                "Bias",
                "mean",
            ),
            Average_Variance=(
                "Variance",
                "mean",
            ),
            Number_of_Files=(
                "Source File",
                "nunique",
            ),
            Minimum_k=(
                "k",
                "min",
            ),
            Maximum_k=(
                "k",
                "max",
            ),
        )
    )

    overall = overall.sort_values(
        by="Average_Quality_Score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    overall.insert(
        0,
        "Rank",
        range(1, len(overall) + 1),
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 90)
    print(
        f"OVERALL ESTIMATOR AVERAGES — "
        f"k >= {MINIMUM_K}"
    )
    print("=" * 90)

    print(
        overall.to_string(
            index=False,
        )
    )

    print()
    print("Saved result to:")
    print(OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()