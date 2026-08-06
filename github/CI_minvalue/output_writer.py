from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def safe_sheet_name(name: str, used: set[str]) -> str:
    invalid = set(r'[]:*?/\\')
    base = "".join("_" if c in invalid else c for c in name)[:31] or "Results"
    candidate = base
    number = 2

    while candidate in used:
        suffix = f"_{number}"
        candidate = base[: 31 - len(suffix)] + suffix
        number += 1

    used.add(candidate)
    return candidate


def _extract_estimators(results: pd.DataFrame) -> list[str]:
    """
    Find estimator names from columns such as:

        alpha (Hill)
        P (1 or 0) (Hill)
        (Upper - Lower) (Hill)

    True Alpha is handled separately.
    """
    estimators: list[str] = []

    for column in results.columns:
        match = re.fullmatch(r"alpha \((.+)\)", str(column))
        if match:
            estimator = match.group(1)
            if estimator not in estimators:
                estimators.append(estimator)

    if "True Alpha" in results.columns:
        estimators.append("True Alpha")

    return estimators


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _summary_for_group(
    group: pd.DataFrame,
    distribution: str,
    k_value: object,
) -> list[dict[str, object]]:
    """
    Produce one row per alpha estimator.

    p:
        Mean of the 0/1 coverage indicator.

    Average Length of CI:
        Mean of CI Upper - CI Lower.

    Bias:
        Mean(alpha_hat - true_alpha).

    Variance:
        Sample variance of alpha_hat (ddof=1).
    """
    rows: list[dict[str, object]] = []
    estimators = _extract_estimators(group)

    if "True Alpha" not in group.columns:
        raise ValueError(
            f"{distribution!r} is missing the 'True Alpha' column, "
            "so alpha bias cannot be calculated."
        )

    true_alpha = _clean_numeric(group["True Alpha"])

    for estimator in estimators:
        if estimator == "True Alpha":
            alpha_column = "True Alpha"
        else:
            alpha_column = f"alpha ({estimator})"

        coverage_column = f"P (1 or 0) ({estimator})"
        length_column = f"(Upper - Lower) ({estimator})"

        missing = [
            column
            for column in (
                alpha_column,
                coverage_column,
                length_column,
            )
            if column not in group.columns
        ]

        if missing:
            # Skip an incomplete estimator instead of crashing the entire
            # workbook-writing step.
            continue

        alpha_estimates = _clean_numeric(group[alpha_column])
        coverage = _clean_numeric(group[coverage_column])
        ci_lengths = _clean_numeric(group[length_column])

        valid_bias = alpha_estimates.notna() & true_alpha.notna()
        alpha_errors = (
            alpha_estimates[valid_bias]
            - true_alpha[valid_bias]
        )

        rows.append(
            {
                "Distribution": distribution,
                "k": k_value,
                "Estimator": estimator,
                "p": coverage.mean(),
                "Average Length of CI": ci_lengths.mean(),
                "Bias": alpha_errors.mean(),
                "Variance": alpha_estimates.var(ddof=1),
            }
        )

    return rows


def build_final_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the estimator-specific summary produced by summary.py into the
    clean workbook Summary format.

    This preserves the separately selected k for every estimator. It avoids
    rebuilding the summary from the plain ``k`` column in the wide results
    table, because that column belongs only to the first estimator.
    """
    required_columns = {
        "Distribution",
        "k",
        "Method",
        "Coverage Proportion",
        "Average CI Length",
        "Alpha Bias",
        "Alpha Variance",
    }
    missing = sorted(required_columns.difference(summary.columns))
    if missing:
        raise ValueError(
            "Summary table is missing required columns: "
            + ", ".join(missing)
        )

    final_summary = summary.rename(
        columns={
            "Method": "Estimator",
            "Coverage Proportion": "p",
            "Average CI Length": "Average Length of CI",
            "Alpha Bias": "Bias",
            "Alpha Variance": "Variance",
        }
    ).copy()

    final_summary = final_summary[
        [
            "Distribution",
            "k",
            "Estimator",
            "p",
            "Average Length of CI",
            "Bias",
            "Variance",
        ]
    ]

    if final_summary.empty:
        return final_summary

    # With one distribution, remove only the redundant Distribution column.
    # Keep k even when all estimators happen to select the same value, because
    # k is part of the requested output format.
    if final_summary["Distribution"].nunique(dropna=False) == 1:
        final_summary = final_summary.drop(columns="Distribution")

    preferred_order = [
        "True Alpha Benchmark",
        "Hill",
        "Mixed Kernel",
        "GPD MLE",
        "Grimshaw MLE",
        "L-Moment",
        "Moment",
        "MPS GPD",
        "Pickands",
        "PWM",
        "QQ",
        "Reduced Bias Hill",
        "UH",
        "Zhang LME",
    ]
    order_lookup = {
        estimator: index
        for index, estimator in enumerate(preferred_order)
    }

    final_summary["_order"] = final_summary["Estimator"].map(
        order_lookup
    ).fillna(len(preferred_order))

    sort_columns = [
        column
        for column in ("Distribution", "_order")
        if column in final_summary.columns
    ]

    final_summary = (
        final_summary.sort_values(sort_columns, kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return final_summary

def write_workbook(
    path: Path,
    settings: pd.DataFrame,
    results_by_distribution: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    errors: pd.DataFrame,
) -> None:
    """
    Write the simulation workbook.

    The supplied summary already contains the separately selected k for each
    estimator, so it is used directly when building the Summary sheet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = {
        "Settings",
        "Summary",
        "Errors",
    }
    result_sheets: list[str] = []

    final_summary = build_final_summary(summary)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        settings.to_excel(
            writer,
            sheet_name="Settings",
            index=False,
        )

        for distribution, results in results_by_distribution.items():
            sheet = safe_sheet_name(distribution, used)
            result_sheets.append(sheet)
            results.to_excel(
                writer,
                sheet_name=sheet,
                index=False,
            )

        final_summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        errors.to_excel(
            writer,
            sheet_name="Errors",
            index=False,
        )

    format_workbook(path, result_sheets)


def format_workbook(
    path: Path,
    result_sheets: list[str],
) -> None:
    workbook = load_workbook(path)

    pale = PatternFill(
        "solid",
        fgColor="D9EAF7",
    )
    dark = PatternFill(
        "solid",
        fgColor="1F4E78",
    )
    yes = PatternFill(
        "solid",
        fgColor="D9EAD3",
    )
    no = PatternFill(
        "solid",
        fgColor="F4CCCC",
    )

    for sheet_name in workbook.sheetnames:
        ws = workbook[sheet_name]
        ws.freeze_panes = "A2"

        if ws.max_row >= 1 and ws.max_column >= 1:
            ws.auto_filter.ref = ws.dimensions

        for cell in ws[1]:
            cell.font = Font(
                bold=True,
                color=(
                    "000000"
                    if sheet_name in result_sheets
                    else "FFFFFF"
                ),
            )
            cell.fill = (
                pale
                if sheet_name in result_sheets
                else dark
            )
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )

        ws.row_dimensions[1].height = 38

        for index in range(1, ws.max_column + 1):
            header = str(
                ws.cell(1, index).value or ""
            )
            letter = get_column_letter(index)

            if header == "Estimator":
                width = 24
            elif header == "Distribution":
                width = 24
            elif header in {
                "p",
                "Bias",
                "Variance",
            }:
                width = 18
            elif header == "Average Length of CI":
                width = 24
            elif "Upper - Lower" in header:
                width = 24
            elif "P (1 or 0)" in header:
                width = 20
            elif "alpha" in header.lower():
                width = 18
            else:
                width = min(
                    max(len(header) + 3, 14),
                    30,
                )

            ws.column_dimensions[letter].width = width

            if (
                sheet_name in result_sheets
                and "P (1 or 0)" in header
                and ws.max_row >= 2
            ):
                cell_range = (
                    f"{letter}2:{letter}{ws.max_row}"
                )
                ws.conditional_formatting.add(
                    cell_range,
                    CellIsRule(
                        operator="equal",
                        formula=["1"],
                        fill=yes,
                    ),
                )
                ws.conditional_formatting.add(
                    cell_range,
                    CellIsRule(
                        operator="equal",
                        formula=["0"],
                        fill=no,
                    ),
                )

        header_lookup = {
            str(cell.value): cell.column
            for cell in ws[1]
        }

        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if not isinstance(
                    cell.value,
                    (float, np.floating),
                ):
                    continue

                header = str(
                    ws.cell(
                        row=1,
                        column=cell.column,
                    ).value
                    or ""
                )

                if sheet_name == "Summary" and header == "p":
                    cell.number_format = "0.000"
                elif sheet_name == "Summary":
                    cell.number_format = "0.0000000000"
                else:
                    cell.number_format = "0.0000000000"

        if sheet_name == "Summary":
            for row in ws.iter_rows(
                min_row=2,
                max_row=ws.max_row,
            ):
                for cell in row:
                    cell.alignment = Alignment(
                        horizontal=(
                            "left"
                            if ws.cell(
                                1,
                                cell.column,
                            ).value
                            in {
                                "Estimator",
                                "Distribution",
                            }
                            else "center"
                        ),
                        vertical="center",
                    )

    workbook.save(path)