from pathlib import Path

import pandas as pd


INPUT_FOLDER = Path("outputs/newscore")
OUTPUT_FILE = INPUT_FOLDER / "overall_estimator_averages.csv"


def main() -> None:
    excel_files = sorted(
        INPUT_FOLDER.glob("*_estimator_averages.xlsx")
    )

    if not excel_files:
        raise FileNotFoundError(
            f"No estimator workbooks found in:\n{INPUT_FOLDER.resolve()}"
        )

    summaries = []

    for workbook in excel_files:
        print(f"Reading {workbook.name}")

        summary = pd.read_excel(
            workbook,
            sheet_name="Estimator Averages",
        )

        summary["Source File"] = workbook.name

        summaries.append(summary)

    combined = pd.concat(
        summaries,
        ignore_index=True,
    )

    overall = (
        combined
        .groupby("Estimator", as_index=False)
        .agg(
            {
                "Average p": "mean",
                "Average CI Length": "mean",
                "Quality Score": "mean",
                "Bias": "mean",
                "Variance": "mean",
                "Source File": "count",
            }
        )
        .rename(
            columns={
                "Source File": "Number of Files"
            }
        )
        .sort_values(
            "Quality Score",
            ascending=False,
        )
    )

    overall.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE.resolve())
    print()
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()