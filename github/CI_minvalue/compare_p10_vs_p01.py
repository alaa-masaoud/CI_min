from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FOLDER = Path("outputs/newscore")

OUTPUT_CSV = INPUT_FOLDER / "p10_vs_p01_comparison.csv"
OUTPUT_EXCEL = INPUT_FOLDER / "p10_vs_p01_comparison.xlsx"

MINIMUM_K = 100

TARGET_COVERAGE = 0.95
COVERAGE_WEIGHT = 50


# ============================================================
# FILENAME EXTRACTION
# ============================================================

def extract_k(filename: str) -> int | None:
    """
    Extracts k from filenames such as:

        k[100]
        k[150]
        k[200]
        k[250]
    """
    match = re.search(
        r"k\[(\d+)\]",
        filename,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return int(match.group(1))


def extract_p_group(filename: str) -> str | None:
    """
    Classifies a filename as p=0.10 or p=0.01.

    Recognized examples for p=0.10:
        p[10]
        p[_10]
        p[0.10]
        p[.10]
        p[0_10]
        p[_1]
        p[0.1]

    Recognized examples for p=0.01:
        p[01]
        p[_01]
        p[0.01]
        p[.01]
        p[0_01]
    """
    name = filename.lower().replace(" ", "")

    match = re.search(
        r"p\[([^\]]+)\]",
        name,
    )

    if match is None:
        return None

    raw_p = match.group(1)

    # Normalize common filename conventions.
    normalized = raw_p.replace("_", ".")

    # Handle p[_01] -> .01
    # Handle p[_1]  -> .1
    if normalized.startswith("."):
        try:
            value = float(normalized)
        except ValueError:
            return None

    else:
        try:
            value = float(normalized)
        except ValueError:
            return None

        # Handle percentage-style labels:
        # p[10] means 10%, not 10.0.
        if value > 1:
            value /= 100

        # Handle p[01], which Python reads as 1.
        # Under your filename convention, "01" means 1%.
        if raw_p in {"01", "_01"}:
            value = 0.01

    if abs(value - 0.10) < 1e-9:
        return "p10"

    if abs(value - 0.01) < 1e-9:
        return "p01"

    return None


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality_score(
    coverage: pd.Series,
    ci_length: pd.Series,
) -> pd.Series:
    """
    Q = 1 / [L * (1 + 50 * |C - 0.95|)]
    """
    coverage_error = (
        coverage - TARGET_COVERAGE
    ).abs()

    coverage_penalty = (
        1
        + COVERAGE_WEIGHT * coverage_error
    )

    return 1 / (
        ci_length * coverage_penalty
    )


# ============================================================
# COLUMN HANDLING
# ============================================================

def find_column(
    dataframe: pd.DataFrame,
    possible_names: list[str],
) -> str | None:
    """
    Finds a column without being sensitive to capitalization
    or extra spaces.
    """
    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for possible_name in possible_names:
        normalized_name = (
            possible_name.strip().lower()
        )

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


def standardize_summary(
    dataframe: pd.DataFrame,
    source_file: str,
    k_value: int,
    p_group: str,
) -> pd.DataFrame:
    """
    Converts the estimator summary into consistent column names.
    """
    estimator_column = find_column(
        dataframe,
        ["Estimator"],
    )

    coverage_column = find_column(
        dataframe,
        [
            "Average p",
            "Average Coverage",
            "Average_Coverage",
        ],
    )

    ci_column = find_column(
        dataframe,
        [
            "Average CI Length",
            "Average_CI_Length",
        ],
    )

    quality_column = find_column(
        dataframe,
        [
            "Quality Score",
            "Average Quality Score",
            "Average_Quality_Score",
        ],
    )

    bias_column = find_column(
        dataframe,
        [
            "Bias",
            "Average Bias",
            "Average_Bias",
        ],
    )

    variance_column = find_column(
        dataframe,
        [
            "Variance",
            "Average Variance",
            "Average_Variance",
        ],
    )

    required = {
        "Estimator": estimator_column,
        "Coverage": coverage_column,
        "CI Length": ci_column,
        "Bias": bias_column,
        "Variance": variance_column,
    }

    missing = [
        name
        for name, column in required.items()
        if column is None
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    result = pd.DataFrame({
        "Estimator": dataframe[estimator_column],
        "Coverage": dataframe[coverage_column],
        "CI Length": dataframe[ci_column],
        "Bias": dataframe[bias_column],
        "Variance": dataframe[variance_column],
    })

    if quality_column is not None:
        result["Quality Score"] = dataframe[
            quality_column
        ]

    else:
        result["Quality Score"] = (
            calculate_quality_score(
                pd.to_numeric(
                    result["Coverage"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    result["CI Length"],
                    errors="coerce",
                ),
            )
        )

    result["Source File"] = source_file
    result["k"] = k_value
    result["p Group"] = p_group

    numeric_columns = [
        "Coverage",
        "CI Length",
        "Quality Score",
        "Bias",
        "Variance",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result["Estimator"] = (
        result["Estimator"]
        .astype(str)
        .str.strip()
    )

    result = result.dropna(
        subset=[
            "Estimator",
            "Coverage",
            "CI Length",
        ]
    )

    return result


# ============================================================
# GROUP AVERAGING
# ============================================================

def calculate_group_average(
    data: pd.DataFrame,
    group_name: str,
) -> pd.DataFrame:
    """
    Calculates the final averages for one sampling proportion.
    """
    group_data = data[
        data["p Group"] == group_name
    ].copy()

    if group_data.empty:
        return pd.DataFrame()

    average = (
        group_data
        .groupby(
            "Estimator",
            as_index=False,
        )
        .agg(
            Average_Coverage=(
                "Coverage",
                "mean",
            ),
            Average_CI_Length=(
                "CI Length",
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

    average = average.sort_values(
        "Average_Quality_Score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    average.insert(
        0,
        "Rank",
        range(1, len(average) + 1),
    )

    return average


# ============================================================
# COMPARISON TABLE
# ============================================================

def create_comparison(
    p10: pd.DataFrame,
    p01: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates a side-by-side p=0.10 versus p=0.01 table.
    """
    p10_columns = p10[[
        "Estimator",
        "Rank",
        "Average_Coverage",
        "Average_CI_Length",
        "Average_Quality_Score",
        "Average_Bias",
        "Average_Variance",
        "Number_of_Files",
    ]].copy()

    p01_columns = p01[[
        "Estimator",
        "Rank",
        "Average_Coverage",
        "Average_CI_Length",
        "Average_Quality_Score",
        "Average_Bias",
        "Average_Variance",
        "Number_of_Files",
    ]].copy()

    p10_columns = p10_columns.rename(
        columns={
            "Rank": "Rank_p10",
            "Average_Coverage": "Coverage_p10",
            "Average_CI_Length": "CI_Length_p10",
            "Average_Quality_Score": "Quality_p10",
            "Average_Bias": "Bias_p10",
            "Average_Variance": "Variance_p10",
            "Number_of_Files": "Files_p10",
        }
    )

    p01_columns = p01_columns.rename(
        columns={
            "Rank": "Rank_p01",
            "Average_Coverage": "Coverage_p01",
            "Average_CI_Length": "CI_Length_p01",
            "Average_Quality_Score": "Quality_p01",
            "Average_Bias": "Bias_p01",
            "Average_Variance": "Variance_p01",
            "Number_of_Files": "Files_p01",
        }
    )

    comparison = p10_columns.merge(
        p01_columns,
        on="Estimator",
        how="outer",
    )

    comparison["Coverage_Change_p01_minus_p10"] = (
        comparison["Coverage_p01"]
        - comparison["Coverage_p10"]
    )

    comparison["CI_Length_Change_p01_minus_p10"] = (
        comparison["CI_Length_p01"]
        - comparison["CI_Length_p10"]
    )

    comparison["Quality_Change_p01_minus_p10"] = (
        comparison["Quality_p01"]
        - comparison["Quality_p10"]
    )

    comparison["Bias_Change_p01_minus_p10"] = (
        comparison["Bias_p01"]
        - comparison["Bias_p10"]
    )

    comparison["Variance_Change_p01_minus_p10"] = (
        comparison["Variance_p01"]
        - comparison["Variance_p10"]
    )

    comparison["Rank_Change_p01_minus_p10"] = (
        comparison["Rank_p01"]
        - comparison["Rank_p10"]
    )

    comparison = comparison.sort_values(
        [
            "Rank_p10",
            "Rank_p01",
        ],
        na_position="last",
    ).reset_index(drop=True)

    return comparison


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Folder not found: "
            f"{INPUT_FOLDER.resolve()}"
        )

    workbook_files = sorted(
        INPUT_FOLDER.glob(
            "*_estimator_averages.xlsx"
        )
    )

    if not workbook_files:
        raise FileNotFoundError(
            "No estimator-average workbooks were found in "
            f"{INPUT_FOLDER.resolve()}"
        )

    collected_data: list[pd.DataFrame] = []

    selected_file_log: list[dict] = []

    for workbook in workbook_files:
        k_value = extract_k(workbook.name)
        p_group = extract_p_group(workbook.name)

        if k_value is None:
            print(
                f"Skipped — no k found: "
                f"{workbook.name}"
            )
            continue

        if k_value < MINIMUM_K:
            print(
                f"Skipped — k={k_value} is below "
                f"{MINIMUM_K}: {workbook.name}"
            )
            continue

        if p_group is None:
            print(
                f"Skipped — not p=0.10 or p=0.01: "
                f"{workbook.name}"
            )
            continue

        try:
            dataframe = pd.read_excel(
                workbook,
                sheet_name="Estimator Averages",
            )

        except ValueError:
            # Fall back to the first sheet if the expected
            # sheet name is not present.
            dataframe = pd.read_excel(
                workbook,
                sheet_name=0,
            )

        try:
            standardized = standardize_summary(
                dataframe=dataframe,
                source_file=workbook.name,
                k_value=k_value,
                p_group=p_group,
            )

        except ValueError as error:
            print(
                f"Skipped {workbook.name}: {error}"
            )
            continue

        collected_data.append(standardized)

        selected_file_log.append({
            "Source File": workbook.name,
            "p Group": p_group,
            "k": k_value,
        })

        print(
            f"Included: {workbook.name} "
            f"({p_group}, k={k_value})"
        )

    if not collected_data:
        raise RuntimeError(
            "No matching p=0.10 or p=0.01 files "
            "with k >= 100 were processed."
        )

    combined = pd.concat(
        collected_data,
        ignore_index=True,
    )

    p10_average = calculate_group_average(
        combined,
        "p10",
    )

    p01_average = calculate_group_average(
        combined,
        "p01",
    )

    if p10_average.empty:
        raise RuntimeError(
            "No p=0.10 files were detected. "
            "Check the p convention in the filenames."
        )

    if p01_average.empty:
        raise RuntimeError(
            "No p=0.01 files were detected. "
            "Check the p convention in the filenames."
        )

    comparison = create_comparison(
        p10_average,
        p01_average,
    )

    # Save the main comparison CSV.
    comparison.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # Save all tables in one Excel workbook.
    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl",
    ) as writer:
        p10_average.to_excel(
            writer,
            sheet_name="p10 Overall",
            index=False,
        )

        p01_average.to_excel(
            writer,
            sheet_name="p01 Overall",
            index=False,
        )

        comparison.to_excel(
            writer,
            sheet_name="p10 vs p01",
            index=False,
        )

        pd.DataFrame(
            selected_file_log
        ).to_excel(
            writer,
            sheet_name="Files Used",
            index=False,
        )

    print()
    print("=" * 100)
    print("p = 0.10 OVERALL RESULTS")
    print("=" * 100)
    print(
        p10_average.to_string(
            index=False,
        )
    )

    print()
    print("=" * 100)
    print("p = 0.01 OVERALL RESULTS")
    print("=" * 100)
    print(
        p01_average.to_string(
            index=False,
        )
    )

    print()
    print("=" * 100)
    print("p = 0.10 VERSUS p = 0.01")
    print("=" * 100)
    print(
        comparison.to_string(
            index=False,
        )
    )

    print()
    print(f"CSV saved to:\n{OUTPUT_CSV.resolve()}")
    print()
    print(f"Excel saved to:\n{OUTPUT_EXCEL.resolve()}")


if __name__ == "__main__":
    main()