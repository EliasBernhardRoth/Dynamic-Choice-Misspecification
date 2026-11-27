import csv
from pathlib import Path

import numpy as np

import config as cfg
from transitions import create_mileage_grid, build_transition_matrices
from limited_foresight import solve_limited_foresight
from simulation import simulate_data
from estimation import estimate_transition_probabilities
from utility_estimation import estimate_utility_parameters


HORIZON_VALUES = (0, 2, 4, 6, 8, 25, 40, 60)

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "results" / "variant2"

PARAMETER_NAMES = ("rc", "theta", "px0", "px1", "px2")


def write_csv(path, rows):
    """Save results as a CSV file."""

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_estimates(horizon, estimates, true_parameters):
    """
    Calculate relative biases and Monte Carlo standard errors.

    Both are expressed as percentages of the true parameters.
    """

    estimates = np.asarray(estimates, dtype=float)
    n_repetitions = estimates.shape[0]

    if n_repetitions < 2:
        raise ValueError("At least two repetitions are required.")

    means = estimates.mean(axis=0)

    # Sample standard deviation divided by the square root of R.
    standard_errors = (
        estimates.std(axis=0, ddof=1) / np.sqrt(n_repetitions)
    )

    relative_bias = 100.0 * (means - true_parameters) / true_parameters
    relative_se = 100.0 * standard_errors / true_parameters

    row = {"h": horizon}

    for i in range(len(PARAMETER_NAMES)):
        name = PARAMETER_NAMES[i]

        row[f"bias_{name}"] = float(relative_bias[i])
        row[f"se_{name}"] = float(relative_se[i])

    return row


def run_variant2():

    n_repetitions = cfg.N_REPETITIONS

    if n_repetitions < 2:
        raise ValueError("At least two repetitions are required.")

    # Both agent and researcher use a discount factor of 0.8.
    beta_agent = 0.8
    beta_model = 0.8

    if not np.isclose(cfg.BETA_AGENT, beta_agent, rtol=0, atol=1e-12):
        raise ValueError("Set BETA_AGENT = 0.8 in config.py.")

    if not np.isclose(cfg.BETA_MODEL, beta_model, rtol=0, atol=1e-12):
        raise ValueError("Set BETA_MODEL = 0.8 in config.py.")

    true_probabilities = np.array([
        cfg.P_ZERO,
        cfg.P_ONE_STEP,
        cfg.P_TWO_STEPS,
    ], dtype=float)

    true_parameters = np.array([
        cfg.REPLACEMENT_COST,
        cfg.MAINTENANCE_COST,
        true_probabilities[0],
        true_probabilities[1],
        true_probabilities[2],
    ], dtype=float)

    if not np.all(np.isfinite(true_parameters)):
        raise ValueError("True parameters must be finite.")

    if np.any(true_parameters <= 0):
        raise ValueError("True parameters must be positive.")

    mileage_grid = create_mileage_grid()

    keep_matrix, replace_matrix = build_transition_matrices(
        len(mileage_grid),
        true_probabilities[0],
        true_probabilities[1],
        true_probabilities[2],
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    raw_path = OUTPUT_DIRECTORY / f"raw_estimates_R{n_repetitions}.csv"
    summary_path = OUTPUT_DIRECTORY / f"summary_R{n_repetitions}.csv"
    replacement_path = OUTPUT_DIRECTORY / f"replacement_counts_R{n_repetitions}.csv"

    raw_columns = [
        "h",
        "repetition",
        "seed",
        "rc",
        "theta",
        "px0",
        "px1",
        "px2",
        "mean_replacements_per_bus",
        "negative_log_likelihood",
        "converged",
        "optimizer",
        "function_evaluations",
    ]

    summary_rows = []
    replacement_rows = []

    total_fits = len(HORIZON_VALUES) * n_repetitions

    print(f"Variant 2: {total_fits} simulated datasets", flush=True)
    print(f"Buses: {cfg.N_BUSES}; periods: {cfg.N_PERIODS}", flush=True)
    print(f"Agent's discount factor: {beta_agent}", flush=True)
    print(f"Researcher's discount factor: {beta_model}", flush=True)

    # Save each completed repetition immediately.
    with raw_path.open("w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=raw_columns)
        writer.writeheader()
        file.flush()

        # Loop over the agent's foresight horizons.
        for scenario_index, horizon in enumerate(HORIZON_VALUES, start=1):

            print(f"\nAgent's foresight: h = {horizon}", flush=True)

            # Solve the agent's limited-foresight model.
            solution = solve_limited_foresight(
                mileage_grid=mileage_grid,
                keep_matrix=keep_matrix,
                replace_matrix=replace_matrix,
                replacement_cost=cfg.REPLACEMENT_COST,
                theta_c=cfg.MAINTENANCE_COST,
                beta=beta_agent,
                n_periods=cfg.N_PERIODS,
                horizon=horizon,
            )

            scenario_estimates = []
            scenario_replacements = []

            # Monte Carlo repetitions.
            for repetition in range(1, n_repetitions + 1):

                seed = repetition + 1000 * scenario_index

                data = simulate_data(
                    replace_probabilities=solution["replace_probability"],
                    n_buses=cfg.N_BUSES,
                    p_zero=cfg.P_ZERO,
                    p_one_step=cfg.P_ONE_STEP,
                    p_two_steps=cfg.P_TWO_STEPS,
                    seed=seed,
                )

                # Stage 1: estimate transition probabilities.
                p_hat = estimate_transition_probabilities(data)

                # Stage 2: estimate utility parameters.
                # The researcher assumes full foresight.
                fit = estimate_utility_parameters(
                    data=data,
                    transition_probabilities=p_hat,
                    beta=beta_model,
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
                        f"Non-finite estimates for h={horizon}, "
                        f"repetition={repetition}."
                    )

                mean_replacements = float(
                    data[:, 0].sum() / cfg.N_BUSES
                )

                scenario_estimates.append(estimates)
                scenario_replacements.append(mean_replacements)

                row = {
                    "h": horizon,
                    "repetition": repetition,
                    "seed": seed,
                    "rc": estimates[0],
                    "theta": estimates[1],
                    "px0": estimates[2],
                    "px1": estimates[3],
                    "px2": estimates[4],
                    "mean_replacements_per_bus": mean_replacements,
                    "negative_log_likelihood": fit["negative_log_likelihood"],
                    "converged": fit["converged"],
                    "optimizer": fit["optimizer"],
                    "function_evaluations": fit["function_evaluations"],
                }

                writer.writerow(row)
                file.flush()

                print(
                    f"  Repetition {repetition:02d}/{n_repetitions} | "
                    f"RC = {estimates[0]:.6f} | "
                    f"theta = {estimates[1]:.9f}",
                    flush=True,
                )

            # Summarise estimates across repetitions.
            summary = summarize_estimates(
                horizon=horizon,
                estimates=scenario_estimates,
                true_parameters=true_parameters,
            )

            summary_rows.append(summary)

            # Mean replacement counts and Monte Carlo standard errors.
            replacement_values = np.asarray(scenario_replacements)

            mean_replacements = replacement_values.mean()

            se_replacements = (
                replacement_values.std(ddof=1)
                / np.sqrt(n_repetitions)
            )

            replacement_rows.append({
                "h": horizon,
                "mean_replacements_per_bus": float(mean_replacements),
                "se_mean_replacements_per_bus": float(se_replacements),
            })

            print(
                f"  Average bias: RC = {summary['bias_rc']:.3f}%, "
                f"theta = {summary['bias_theta']:.3f}%",
                flush=True,
            )

    # Save summary results.
    write_csv(summary_path, summary_rows)
    write_csv(replacement_path, replacement_rows)

    print("\nVariant 2 completed successfully.")
    print(f"Individual estimates: {raw_path}")
    print(f"Biases and standard errors: {summary_path}")
    print(f"Replacement counts: {replacement_path}")


if __name__ == "__main__":
    run_variant2()