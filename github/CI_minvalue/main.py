from __future__ import annotations

import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from config import (
    CONFIDENCE_LEVEL,
    FULL_SAMPLE_SIZE,
    K_VALUES,
    MASTER_SEED,
    N_REPETITIONS,
    OUTPUT_DIRECTORY,
    OUTPUT_WORKBOOK,
    RAW_CSV_DIRECTORY,
    SUBSAMPLE_SIZE,
)
from distributions import DISTRIBUTIONS
from evt.estimator_adapter import load_estimators
from evt.estimator_registry import ESTIMATORS
from output_writer import write_workbook
from simulation import run_distribution
from summary import build_summary


def build_settings_table() -> pd.DataFrame:
    return pd.DataFrame([
        ["Master Seed", MASTER_SEED],
        ["Repetitions", N_REPETITIONS],
        ["Full Sample Size", FULL_SAMPLE_SIZE],
        ["Subsample Size", SUBSAMPLE_SIZE],
        ["Sampling With Replacement", False],
        ["k Values Tested", ", ".join(map(str, K_VALUES))],
        ["k Selection", "Separate optimal k selected for each estimator"],
        ["Confidence Level", CONFIDENCE_LEVEL],
        ["Distributions", ", ".join(d.name for d in DISTRIBUTIONS)],
        ["Estimators", ", ".join(e.display_name for e in ESTIMATORS)],
        ["Known Distribution Endpoint Used", False],
    ], columns=["Setting", "Value"])


def main() -> None:
    if SUBSAMPLE_SIZE > FULL_SAMPLE_SIZE:
        raise ValueError("SUBSAMPLE_SIZE cannot exceed FULL_SAMPLE_SIZE.")
    if any(not 4 <= k < SUBSAMPLE_SIZE for k in K_VALUES):
        raise ValueError("Every k must satisfy 4 <= k < SUBSAMPLE_SIZE.")
    if len(set(K_VALUES)) < 2:
        raise ValueError(
            "Optimal-k selection requires at least two distinct values in K_VALUES."
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RAW_CSV_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EVT MINIMUM-CI SIMULATION WITH PER-ESTIMATOR OPTIMAL k")
    print("=" * 70)
    print(f"Master seed      : {MASTER_SEED}")
    print(f"Repetitions      : {N_REPETITIONS:,}")
    print(f"Full sample size : {FULL_SAMPLE_SIZE:,}")
    print(f"Subsample size   : {SUBSAMPLE_SIZE:,}")
    print(f"k values tested  : {K_VALUES}")
    print(f"Confidence level : {CONFIDENCE_LEVEL:.2%}")
    print("Known endpoint   : not supplied")
    print("=" * 70)

    estimator_functions = load_estimators(ESTIMATORS)
    distribution_seeds = np.random.SeedSequence(MASTER_SEED).spawn(
        len(DISTRIBUTIONS)
    )

    distribution_results: dict[str, pd.DataFrame] = {}
    summaries: list[pd.DataFrame] = []
    all_errors: list[dict[str, object]] = []
    all_k_diagnostics: list[pd.DataFrame] = []

    for distribution, seed_sequence in zip(DISTRIBUTIONS, distribution_seeds):
        print(f"\nRunning {distribution.name}")
        results, errors, k_diagnostics = run_distribution(
            distribution=distribution,
            estimators=ESTIMATORS,
            estimator_functions=estimator_functions,
            seed_sequence=seed_sequence,
        )
        distribution_results[distribution.name] = results
        summaries.append(build_summary(distribution, ESTIMATORS, results))
        all_errors.extend(errors)

        k_diagnostics = k_diagnostics.copy()
        k_diagnostics.insert(0, "Distribution", distribution.name)
        all_k_diagnostics.append(k_diagnostics)

        safe_name = "".join(
            c if c.isalnum() else "_" for c in distribution.name
        )
        results.to_csv(RAW_CSV_DIRECTORY / f"{safe_name}.csv", index=False)
        k_diagnostics.to_csv(
            RAW_CSV_DIRECTORY / f"{safe_name}_k_selection.csv",
            index=False,
        )

    summary_table = pd.concat(summaries, ignore_index=True)
    errors_table = pd.DataFrame(all_errors, columns=[
        "Distribution", "Repetition", "k", "Estimator", "Error"
    ])

    write_workbook(
        path=OUTPUT_WORKBOOK,
        settings=build_settings_table(),
        results_by_distribution=distribution_results,
        summary=summary_table,
        errors=errors_table,
    )

    combined_k_diagnostics = pd.concat(all_k_diagnostics, ignore_index=True)
    combined_k_diagnostics.to_csv(
        OUTPUT_DIRECTORY / "optimal_k_diagnostics.csv",
        index=False,
    )

    configuration = {
        "master_seed": MASTER_SEED,
        "n_repetitions": N_REPETITIONS,
        "full_sample_size": FULL_SAMPLE_SIZE,
        "subsample_size": SUBSAMPLE_SIZE,
        "sampling_with_replacement": False,
        "k_values_tested": K_VALUES,
        "k_selection": {
            "separate_for_each_estimator": True,
            "primary_rule": "most valid CIs",
            "secondary_rule": "weighted distance from target coverage",
            "undercoverage_weight": 2.0,
            "overcoverage_weight": 1.0,
            "remaining_tiebreakers": [
                "shortest average CI",
                "smallest absolute alpha bias",
                "smallest alpha variance",
                "smallest k",
            ],
        },
        "confidence_level": CONFIDENCE_LEVEL,
        "known_distribution_endpoint_used": False,
    }
    (OUTPUT_DIRECTORY / "simulation_configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    print("\nSimulation complete.")
    print(f"Workbook: {OUTPUT_WORKBOOK.resolve()}")
    print(
        "k diagnostics: "
        f"{(OUTPUT_DIRECTORY / 'optimal_k_diagnostics.csv').resolve()}"
    )
    print(f"Estimator errors recorded: {len(all_errors)}")


if __name__ == "__main__":
    mp.freeze_support()
    main()