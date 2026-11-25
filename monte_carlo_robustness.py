import argparse
import csv
import json
from pathlib import Path

import numpy as np

import config as cfg
from transitions import create_mileage_grid, build_transition_matrices
from limited_foresight import solve_limited_foresight
from simulation import simulate_data
from estimation import estimate_transition_probabilities
from utility_estimation import estimate_utility_parameters

# Reuse the summary calculations from Variant 2.
# Importing this module does not run its Monte Carlo experiment.
from monte_carlo_variant2 import (
    HORIZON_VALUES,
    PARAMETER_NAMES,
    summarize_estimates,
    write_csv,
)


OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "results" / "robustness"
)

# Explicit thesis baseline: each scenario changes only the listed values.
BASELINE = {
    "replacement_cost": 15.0750,
    "maintenance_cost": 0.0005293,
    "p_zero": 0.3919,
    "p_one_step": 0.5953,
    "beta": 0.8,
}

SCENARIOS = {
    "maintenance_cost": {
        "label": "Higher maintenance cost",
        "changes": {"maintenance_cost": 0.001},
        "table_a2": (
            12.47, 18.06, 18.04, 18.05,
            18.04, 18.05, 18.05, 18.05,
        ),
    },
    "replacement_cost": {
        "label": "Higher replacement cost",
        "changes": {"replacement_cost": 50.0},
        "table_a2": (
            1.38, 3.92, 5.12, 5.20,
            5.06, 4.99, 4.99, 4.99,
        ),
    },
    "transitions": {
        "label": "Higher actual mileage growth",
        "changes": {
            "p_zero": 0.1,
            "p_one_step": 0.3,
        },
        "table_a2": (
            13.23, 20.61, 19.81, 19.63,
            19.69, 19.69, 19.69, 19.69,
        ),
    },
    "discount_factor": {
        "label": "Higher discount factor for agent and researcher",
        "changes": {"beta": 0.99},
        "table_a2": (
            5.75, 14.40, 13.13, 13.23,
            13.24, 13.25, 13.25, 13.24,
        ),
    },
}


def save_settings(path, settings):
    path.write_text(
        json.dumps(settings, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def compare_replacement_counts(replacement_rows, reference_values):
    """Compare simulated means with the rounded entries in Table A2."""
    reference_by_horizon = dict(
        zip((0, 2, 4, 6, 8, 25, 40, 60), reference_values)
    )

    comparison_rows = []

    for row in replacement_rows:
        horizon = row["h"]
        python_mean = row["mean_replacements_per_bus"]
        thesis_mean = reference_by_horizon[horizon]
        difference = python_mean - thesis_mean

        comparison_rows.append(
            {
                "h": horizon,
                "python_mean": python_mean,
                "python_mc_standard_error": row[
                    "se_mean_replacements_per_bus"
                ],
                "thesis_table_a2": thesis_mean,
                "difference": difference,
                # Table A2 reports two decimal places.
                "within_printed_rounding_interval": (
                    abs(difference) <= 0.005 + 1e-12
                ),
            }
        )

    return comparison_rows


def run_scenario(scenario_name, n_repetitions):
    specification = SCENARIOS[scenario_name]
    parameters = {**BASELINE, **specification["changes"]}

    replacement_cost = parameters["replacement_cost"]
    maintenance_cost = parameters["maintenance_cost"]
    beta = parameters["beta"]

    p_zero = parameters["p_zero"]
    p_one_step = parameters["p_one_step"]
    p_two_steps = 1.0 - p_zero - p_one_step

    true_probabilities = np.array(
        [p_zero, p_one_step, p_two_steps],
        dtype=float,
    )

    # Biases must be calculated relative to THIS scenario's true values.
    true_parameters = np.array(
        [
            replacement_cost,
            maintenance_cost,
            *true_probabilities,
        ],
        dtype=float,
    )

    if (
        not np.all(np.isfinite(true_parameters))
        or np.any(true_parameters <= 0)
    ):
        raise ValueError(
            "True parameters must be positive and finite "
            "to calculate percentage biases."
        )

    mileage_grid = create_mileage_grid()

    keep_matrix, replace_matrix = build_transition_matrices(
        len(mileage_grid),
        *true_probabilities,
    )

    scenario_directory = OUTPUT_DIRECTORY / scenario_name
    scenario_directory.mkdir(parents=True, exist_ok=True)

    raw_path = (
        scenario_directory / f"raw_estimates_R{n_repetitions}.csv"
    )
    summary_path = (
        scenario_directory / f"summary_R{n_repetitions}.csv"
    )
    replacement_path = (
        scenario_directory / f"replacement_counts_R{n_repetitions}.csv"
    )
    settings_path = (
        scenario_directory / f"settings_R{n_repetitions}.json"
    )

    settings = {
        "scenario": scenario_name,
        "description": specification["label"],
        "n_buses": cfg.N_BUSES,
        "n_periods": cfg.N_PERIODS,
        "n_repetitions": n_repetitions,
        "mileage_step": cfg.MILEAGE_STEP,
        "max_mileage": cfg.MAX_MILEAGE,
        "horizons": list(HORIZON_VALUES),
        "true_parameters": dict(
            zip(PARAMETER_NAMES, true_parameters.tolist())
        ),
        "beta_agent": beta,
        "beta_model": beta,
        "seed_rule": (
            "repetition + 1000 * horizon_index; "
            "both indices start at 1"
        ),
        "run_completed": False,
    }

    save_settings(settings_path, settings)

    raw_columns = [
        "scenario",
        "h",
        "repetition",
        "seed",
        *PARAMETER_NAMES,
        "mean_replacements_per_bus",
        "initial_negative_log_likelihood",
        "negative_log_likelihood",
        "converged",
        "optimizer",
        "function_evaluations",
    ]

    summary_rows = []
    replacement_rows = []

    print(f"\nScenario: {specification['label']}", flush=True)
    print(
        f"True RC = {replacement_cost}; "
        f"true theta = {maintenance_cost}; "
        f"beta_agent = beta_model = {beta}",
        flush=True,
    )
    print(f"True transitions: {true_probabilities}", flush=True)

    with raw_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=raw_columns)
        writer.writeheader()
        file.flush()

        for horizon_index, horizon in enumerate(HORIZON_VALUES, start=1):
            print(f"\n  Agent's foresight: h = {horizon}", flush=True)

            solution = solve_limited_foresight(
                mileage_grid=mileage_grid,
                keep_matrix=keep_matrix,
                replace_matrix=replace_matrix,
                replacement_cost=replacement_cost,
                theta_c=maintenance_cost,
                beta=beta,
                n_periods=cfg.N_PERIODS,
                horizon=horizon,
            )

            estimates_for_horizon = []
            replacements_for_horizon = []

            for repetition in range(1, n_repetitions + 1):
                seed = repetition + 1000 * horizon_index

                # Each scenario uses its own actual transition probabilities.
                data = simulate_data(
                    replace_probabilities=solution["replace_probability"],
                    n_buses=cfg.N_BUSES,
                    p_zero=p_zero,
                    p_one_step=p_one_step,
                    p_two_steps=p_two_steps,
                    seed=seed,
                )

                p_hat = estimate_transition_probabilities(data)

                # The researcher assumes full foresight.
                # In the discount-factor check, beta=0.99 here too.
                fit = estimate_utility_parameters(
                    data=data,
                    transition_probabilities=p_hat,
                    beta=beta,
                )

                if not fit["converged"]:
                    raise RuntimeError(
                        f"Estimation failed: scenario={scenario_name}, "
                        f"h={horizon}, repetition={repetition}."
                    )

                estimates = [
                    fit["replacement_cost"],
                    fit["maintenance_cost"],
                    float(p_hat[0]),
                    float(p_hat[1]),
                    float(p_hat[2]),
                ]

                if not np.all(np.isfinite(estimates)):
                    raise RuntimeError(
                        f"Non-finite estimates: scenario={scenario_name}, "
                        f"h={horizon}, repetition={repetition}."
                    )

                mean_replacements = float(
                    data[:, 0].sum() / cfg.N_BUSES
                )

                estimates_for_horizon.append(estimates)
                replacements_for_horizon.append(mean_replacements)

                writer.writerow(
                    {
                        "scenario": scenario_name,
                        "h": horizon,
                        "repetition": repetition,
                        "seed": seed,
                        **dict(zip(PARAMETER_NAMES, estimates)),
                        "mean_replacements_per_bus": mean_replacements,
                        "initial_negative_log_likelihood": fit[
                            "initial_negative_log_likelihood"
                        ],
                        "negative_log_likelihood": fit[
                            "negative_log_likelihood"
                        ],
                        "converged": fit["converged"],
                        "optimizer": fit["optimizer"],
                        "function_evaluations": fit["function_evaluations"],
                    }
                )
                file.flush()

                if (
                    repetition == 1
                    or repetition % 5 == 0
                    or repetition == n_repetitions
                ):
                    print(
                        f"    Repetition {repetition:02d}/{n_repetitions} | "
                        f"RC = {estimates[0]:.6f} | "
                        f"theta = {estimates[1]:.9f}",
                        flush=True,
                    )

            summary = summarize_estimates(
                horizon=horizon,
                estimates=estimates_for_horizon,
                true_parameters=true_parameters,
            )
            summary_rows.append(summary)

            replacements = np.asarray(replacements_for_horizon)

            replacement_rows.append(
                {
                    "h": horizon,
                    "mean_replacements_per_bus": float(
                        replacements.mean()
                    ),
                    "se_mean_replacements_per_bus": float(
                        replacements.std(ddof=1)
                        / np.sqrt(n_repetitions)
                    ),
                }
            )

            print(
                f"  Average bias: RC = {summary['bias_rc']:.3f}%, "
                f"theta = {summary['bias_theta']:.3f}%",
                flush=True,
            )

    write_csv(summary_path, summary_rows)
    write_csv(replacement_path, replacement_rows)

    # The thesis table uses 60 repetitions.
    if n_repetitions == 60:
        comparison_rows = compare_replacement_counts(
            replacement_rows,
            specification["table_a2"],
        )

        comparison_path = (
            scenario_directory / "table_A2_comparison_R60.csv"
        )
        write_csv(comparison_path, comparison_rows)

        matches = sum(
            row["within_printed_rounding_interval"]
            for row in comparison_rows
        )

        print(
            f"\nTable A2 comparison: {matches}/{len(comparison_rows)} "
            "entries lie within the printed rounding intervals."
        )
        print(f"Comparison saved: {comparison_path}")

    settings["run_completed"] = True
    save_settings(settings_path, settings)

    print(f"\nCompleted scenario: {scenario_name}")
    print(f"Individual estimates: {raw_path}")
    print(f"Biases and standard errors: {summary_path}")
    print(f"Replacement counts: {replacement_path}")
    print(f"Run settings: {settings_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the thesis robustness checks for Variant 2."
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS],
        default="all",
        help="Run all four scenarios or select one.",
    )
    args = parser.parse_args()

    n_repetitions = cfg.N_REPETITIONS

    if not isinstance(n_repetitions, int) or n_repetitions < 2:
        raise ValueError("N_REPETITIONS must be an integer of at least 2.")

    # Keep the experiment dimensions consistent with Appendix A.1.
    if (
        cfg.N_BUSES,
        cfg.N_PERIODS,
        cfg.MILEAGE_STEP,
        cfg.MAX_MILEAGE,
    ) != (1000, 60, 5000, 350000):
        raise ValueError(
            "For the thesis robustness checks, use N_BUSES=1000, "
            "N_PERIODS=60, MILEAGE_STEP=5000 and MAX_MILEAGE=350000."
        )

    if tuple(HORIZON_VALUES) != (0, 2, 4, 6, 8, 25, 40, 60):
        raise ValueError("The horizon grid must match Appendix A.1.")

    selected_scenarios = (
        list(SCENARIOS)
        if args.scenario == "all"
        else [args.scenario]
    )

    total_fits = (
        len(selected_scenarios) * len(HORIZON_VALUES) * n_repetitions
    )

    print(f"Robustness checks: {total_fits} estimations", flush=True)

    for scenario_name in selected_scenarios:
        run_scenario(scenario_name, n_repetitions)

    print("\nRequested robustness runs completed.")


if __name__ == "__main__":
    main()