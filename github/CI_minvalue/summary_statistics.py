
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


# ---------------------------------------------------------------------
# INPUT AND OUTPUT FOLDERS
# ---------------------------------------------------------------------
INPUT_FOLDER = Path("outputs/raw_csv")
OUTPUT_FOLDER = Path("outputs/newscore")


# ---------------------------------------------------------------------
# QUALITY SCORE SETTINGS
# ---------------------------------------------------------------------
TARGET_COVERAGE = 0.95
COVERAGE_WEIGHT = 50


def find_estimators(columns: list[str]) -> list[str]:
    """
    Find estimator names from columns such as:

        alpha (Hill)
        alpha (Mixed Kernel)
        alpha (GPD MLE)
    """
    estimators: list[str] = []

    for column in columns:
        match = re.fullmatch(r"alpha \((.+)\)", str(column))

        if match:
            estimator = match.group(1)

            if estimator not in estimators:
                estimators.append(estimator)

    if "True Alpha" in columns:
        estimators.append("True Alpha")

    return estimators


def numeric_mean(series: pd.Series) -> float:
    """
    Convert a series to numeric values and calculate its mean.

    Invalid values and infinite values are treated as missing.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return float(values.mean())


def numeric_variance(series: pd.Series) -> float:
    """
    Convert a series to numeric values and calculate its sample variance.

    Invalid values and infinite values are treated as missing.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return float(values.var(ddof=1))


def calculate_quality_score(
    average_p: float,
    average_ci_length: float,
) -> float:
    """
    Calculate a symmetric confidence interval quality score.

    Higher scores indicate better confidence interval performance.

    The score:

        - rewards shorter confidence intervals,
        - targets 95% empirical coverage,
        - penalizes undercoverage and overcoverage equally.

    Formula:

        Quality Score
        =
        1 / [
            average_ci_length
            * (
                1
                + 50 * abs(average_p - 0.95)
            )
        ]

    A coverage value equally far above or below 0.95 receives
    the same penalty.
    """
    if (
        pd.isna(average_p)
        or pd.isna(average_ci_length)
        or average_ci_length <= 0
    ):
        return float("nan")

    coverage_error = abs(
        average_p - TARGET_COVERAGE
    )

    coverage_penalty = (
        1.0
        + COVERAGE_WEIGHT * coverage_error
    )

    return 1.0 / (
        average_ci_length * coverage_penalty
    )


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one summary row for every estimator.

    Output columns:

        Estimator
        Average p
        Average CI Length
        Quality Score
        Bias
        Variance

    Bias and variance refer to the alpha estimates.

        Bias
        =
        mean(alpha_hat - true_alpha)

        Variance
        =
        sample variance of alpha_hat
    """
    estimators = find_estimators(
        list(results.columns)
    )

    if "True Alpha" not in results.columns:
        raise ValueError(
            "The CSV must contain a 'True Alpha' column."
        )

    true_alpha = pd.to_numeric(
        results["True Alpha"],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    summary_rows: list[dict[str, object]] = []

    for estimator in estimators:
        alpha_column = (
            "True Alpha"
            if estimator == "True Alpha"
            else f"alpha ({estimator})"
        )

        p_column = (
            f"P (1 or 0) ({estimator})"
        )

        length_column = (
            f"(Upper - Lower) ({estimator})"
        )

        required_columns = (
            alpha_column,
            p_column,
            length_column,
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in results.columns
        ]

        if missing_columns:
            print(
                f"Skipping {estimator}: "
                f"missing columns {missing_columns}"
            )
            continue

        alpha_estimates = pd.to_numeric(
            results[alpha_column],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid_bias_rows = (
            alpha_estimates.notna()
            & true_alpha.notna()
        )

        bias = float(
            (
                alpha_estimates[valid_bias_rows]
                - true_alpha[valid_bias_rows]
            ).mean()
        )

        average_p = numeric_mean(
            results[p_column]
        )

        average_ci_length = numeric_mean(
            results[length_column]
        )

        quality_score = calculate_quality_score(
            average_p=average_p,
            average_ci_length=average_ci_length,
        )

        variance = numeric_variance(
            alpha_estimates
        )

        summary_rows.append(
            {
                "Estimator": estimator,
                "Average p": average_p,
                "Average CI Length": average_ci_length,
                "Quality Score": quality_score,
                "Bias": bias,
                "Variance": variance,
            }
        )

    return pd.DataFrame(summary_rows)


def format_workbook(path: Path) -> None:
    """
    Apply formatting to every worksheet in an Excel workbook.
    """
    workbook = load_workbook(path)

    header_fill = PatternFill(
        fill_type="solid",
        fgColor="1F4E78",
    )

    header_font = Font(
        bold=True,
        color="FFFFFF",
    )

    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]

        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = (
            worksheet.dimensions
        )

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )

        worksheet.row_dimensions[1].height = 32

        for column_number in range(
            1,
            worksheet.max_column + 1,
        ):
            header = str(
                worksheet.cell(
                    row=1,
                    column=column_number,
                ).value
                or ""
            )

            if header == "Estimator":
                width = 24

            elif header == "Average CI Length":
                width = 22

            elif header == "Quality Score":
                width = 20

            else:
                width = 18

            column_letter = get_column_letter(
                column_number
            )

            worksheet.column_dimensions[
                column_letter
            ].width = width

        for row in worksheet.iter_rows(
            min_row=2,
        ):
            for cell in row:
                header = str(
                    worksheet.cell(
                        row=1,
                        column=cell.column,
                    ).value
                    or ""
                )

                if isinstance(
                    cell.value,
                    (float, np.floating),
                ):
                    if header == "Average p":
                        cell.number_format = "0.000"

                    elif header == "Quality Score":
                        cell.number_format = "0.000000"

                    else:
                        cell.number_format = (
                            "0.0000000000"
                        )

                cell.alignment = Alignment(
                    horizontal=(
                        "left"
                        if header == "Estimator"
                        else "center"
                    ),
                    vertical="center",
                )

    workbook.save(path)


def process_csv(
    input_csv: Path,
    output_xlsx: Path,
) -> pd.DataFrame:
    """
    Read one raw CSV, build its summary, and save its workbook.
    """
    results = pd.read_csv(input_csv)

    summary = build_summary(results)

    with pd.ExcelWriter(
        output_xlsx,
        engine="openpyxl",
    ) as writer:
        summary.to_excel(
            writer,
            sheet_name="Estimator Averages",
            index=False,
        )

        results.to_excel(
            writer,
            sheet_name="Raw Data",
            index=False,
        )

    format_workbook(output_xlsx)

    return summary


def main() -> None:
    """
    Process every CSV file in outputs/rawcsv.

    Each resulting workbook is saved in outputs/newscore.
    """
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            "Could not find the input folder:\n"
            f"{INPUT_FOLDER.resolve()}"
        )

    if not INPUT_FOLDER.is_dir():
        raise NotADirectoryError(
            "The input path is not a folder:\n"
            f"{INPUT_FOLDER.resolve()}"
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_files = sorted(
        INPUT_FOLDER.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files were found in:\n"
            f"{INPUT_FOLDER.resolve()}"
        )

    successful_files = 0
    failed_files = 0

    print(
        f"Found {len(csv_files)} CSV file(s)."
    )

    print(
        f"Input folder: {INPUT_FOLDER.resolve()}"
    )

    print(
        f"Output folder: {OUTPUT_FOLDER.resolve()}"
    )

    for file_number, input_csv in enumerate(
        csv_files,
        start=1,
    ):
        output_xlsx = (
            OUTPUT_FOLDER
            / (
                f"{input_csv.stem}"
                "_estimator_averages.xlsx"
            )
        )

        print()
        print("=" * 70)

        print(
            f"File {file_number}/{len(csv_files)}"
        )

        print(
            f"Processing: {input_csv.name}"
        )

        try:
            summary = process_csv(
                input_csv=input_csv,
                output_xlsx=output_xlsx,
            )

            print(
                f"Saved workbook to: "
                f"{output_xlsx.resolve()}"
            )

            print()
            print(
                summary.to_string(
                    index=False,
                )
            )

            successful_files += 1

        except Exception as error:
            failed_files += 1

            print(
                f"Failed to process "
                f"{input_csv.name}"
            )

            print(
                f"Error: {error}"
            )

    print()
    print("=" * 70)
    print("Finished processing all CSV files.")
    print(
        f"Successfully processed: "
        f"{successful_files}"
    )
    print(
        f"Failed: {failed_files}"
    )
    print(
        f"Output folder: "
        f"{OUTPUT_FOLDER.resolve()}"
    )


if __name__ == "__main__":
    main()

