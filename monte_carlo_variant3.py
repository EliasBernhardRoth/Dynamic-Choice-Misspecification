import csv
from pathlib import Path

import numpy as np

import config as cfg
from transitions import create_mileage_grid, build_transition_matrices
from solver import solve_model
from simulation import simulate_data
from estimation import estimate_transition_probabilities
from utility_estimation import estimate_utility_parameters


ALPHA_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

BELIEF_SCENARIOS = (
    ("001", "pessimistic", (0.0, 0.0, 1.0)),
    ("100", "optimistic", (1.0, 0.0, 0.0)),
)

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "results" / "variant3"

PARAMETER_NAMES = ("rc", "theta", "px0", "px1", "px2")


def write_csv(path, rows):
    """Save a list of dictionaries as a CSV file."""
    columns = list(rows[0])

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_estimates(alpha, estimates, true_parameters):
    """
    Calculate relative bias and Monte Carlo standard errors.

    Both are expressed as percentages of the true parameter.
    The standard error measures uncertainty in the Monte Carlo mean.
    """
    estimates = np.asarray(estimates, dtype=float)

    n_repetitions = estimates.shape[0]

    if n_repetitions < 2:
        raise ValueError("At least two repetitions are required.")

    means = estimates.mean(axis=0)

    standard_errors = (
        estimates.std(axis=0, ddof=1) / np.sqrt(n_repetitions)
    )

    relative_bias = 100.0 * (means - true_parameters) / true_parameters
    relative_se = 100.0 * standard_errors / true_parameters

    row = {"alpha": alpha}

    for i in range(len(PARAMETER_NAMES)):
        name = PARAMETER_NAMES[i]
        row[f"bias_{name}"] = float(relative_bias[i])
        row[f"se_{name}"] = float(relative_se[i])

    return row


def run_variant3():
    n_repetitions = cfg.N_REPETITIONS

    if n_repetitions < 2:
        raise ValueError("At least two repetitions are required.")

    beta_agent = 0.8
    beta_model = 0.8

    if not np.isclose(cfg.BETA_AGENT, beta_agent, rtol=0, atol=1e-12):
        raise ValueError("Set BETA_AGENT = 0.8 in config.py for Variant 3.")

    if not np.isclose(cfg.BETA_MODEL, beta_model, rtol=0, atol=1e-12):
        raise ValueError("Set BETA_MODEL = 0.8 in config.py for Variant 3.")

    true_probabilities = np.array(
        [cfg.P_ZERO, cfg.P_ONE_STEP, cfg.P_TWO_STEPS],
        dtype=float,
    )

    true_parameters = np.array(
        [
            cfg.REPLACEMENT_COST,
            cfg.MAINTENANCE_COST,
            true_probabilities[0],
            true_probabilities[1],
            true_probabilities[2],
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(true_parameters)):
        raise ValueError("True parameters must be finite.")

    if np.any(true_parameters <= 0):
        raise ValueError("True parameters must be positive.")

    mileage_grid = create_mileage_grid()

    raw_columns = [
        "scenario",
        "alpha",
        "repetition",
        "seed",
        "belief_px0",
        "belief_px1",
        "belief_px2",
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

    total_fits = (
        len(BELIEF_SCENARIOS) * len(ALPHA_VALUES) * n_repetitions
    )

    print(f"Variant 3: {total_fits} simulated datasets", flush=True)
    print(f"Buses: {cfg.N_BUSES}; periods: {cfg.N_PERIODS}", flush=True)
    print(f"Agent's discount factor: {beta_agent}", flush=True)
    print(f"Researcher's discount factor: {beta_model}", flush=True)

    baseline_estimates = None

    for scenario_code, scenario_name, extreme in BELIEF_SCENARIOS:
        print(f"\nBelief scenario: {scenario_name}", flush=True)

        extreme_probabilities = np.asarray(extreme, dtype=float)

        scenario_directory = OUTPUT_DIRECTORY / scenario_code
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

        summary_rows = []
        replacement_rows = []

        # Save each completed repetition immediately.
        with raw_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=raw_columns)
            writer.writeheader()
            file.flush()

            for alpha_index, alpha in enumerate(ALPHA_VALUES, start=1):
                print(f"\n  Belief distortion: alpha = {alpha:.1f}", flush=True)

                subjective_probabilities = (
                    (1.0 - alpha) * true_probabilities
                    + alpha * extreme_probabilities
                )

                # Beliefs determine how the agent evaluates future states.
                subjective_keep, subjective_replace = build_transition_matrices(
                    len(mileage_grid),
                    subjective_probabilities[0],
                    subjective_probabilities[1],
                    subjective_probabilities[2],
                )

                solution = solve_model(
                    mileage_grid=mileage_grid,
                    keep_matrix=subjective_keep,
                    replace_matrix=subjective_replace,
                    replacement_cost=cfg.REPLACEMENT_COST,
                    theta_c=cfg.MAINTENANCE_COST,
                    beta=beta_agent,
                    n_periods=cfg.N_PERIODS,
                )

                scenario_estimates = []
                scenario_replacements = []

                for repetition in range(1, n_repetitions + 1):
                    # Match the original seed convention.
                    # The same seeds are used for both belief scenarios.
                    seed = repetition + 1000 * alpha_index

                    # Actual mileage follows TRUE transitions.
                    # Only the agent's decision rule reflects their beliefs.
                    data = simulate_data(
                        replace_probabilities=solution["replace_probability"],
                        n_buses=cfg.N_BUSES,
                        p_zero=cfg.P_ZERO,
                        p_one_step=cfg.P_ONE_STEP,
                        p_two_steps=cfg.P_TWO_STEPS,
                        seed=seed,
                    )

                    # Stage 1: estimate actual transitions from observed data.
                    p_hat = estimate_transition_probabilities(data)

                    # Stage 2: the researcher uses the estimated transitions,
                    # assuming that the agent shares these beliefs.
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
                            f"Non-finite estimates for {scenario_name}, "
                            f"alpha={alpha}, repetition={repetition}."
                        )

                    mean_replacements = float(
                        data[:, 0].sum() / cfg.N_BUSES
                    )

                    scenario_estimates.append(estimates)
                    scenario_replacements.append(mean_replacements)

                    row = {
                        "scenario": scenario_name,
                        "alpha": alpha,
                        "repetition": repetition,
                        "seed": seed,
                        "belief_px0": float(subjective_probabilities[0]),
                        "belief_px1": float(subjective_probabilities[1]),
                        "belief_px2": float(subjective_probabilities[2]),
                        "rc": estimates[0],
                        "theta": estimates[1],
                        "px0": estimates[2],
                        "px1": estimates[3],
                        "px2": estimates[4],
                        "mean_replacements_per_bus": mean_replacements,
                        "negative_log_likelihood": fit[
                            "negative_log_likelihood"
                        ],
                        "converged": fit["converged"],
                        "optimizer": fit["optimizer"],
                        "function_evaluations": fit["function_evaluations"],
                    }

                    writer.writerow(row)
                    file.flush()

                    print(
                        f"    Repetition {repetition:02d}/{n_repetitions} | "
                        f"RC = {estimates[0]:.6f} | "
                        f"theta = {estimates[1]:.9f}",
                        flush=True,
                    )

                # At alpha=0, beliefs and seeds coincide across scenarios.
                # Therefore, the individual estimates must also agree.
                if alpha == 0.0:
                    if baseline_estimates is None:
                        baseline_estimates = np.array(scenario_estimates)
                    else:
                        np.testing.assert_allclose(
                            np.asarray(scenario_estimates),
                            baseline_estimates,
                            rtol=1e-10,
                            atol=1e-10,
                            err_msg=(
                                "The two alpha=0 baselines do not agree."
                            ),
                        )

                summary = summarize_estimates(
                    alpha=alpha,
                    estimates=scenario_estimates,
                    true_parameters=true_parameters,
                )
                summary_rows.append(summary)

                replacement_values = np.asarray(scenario_replacements)

                mean_replacements = replacement_values.mean()
                se_replacements = (
                    replacement_values.std(ddof=1) / np.sqrt(n_repetitions)
                )

                replacement_rows.append({
                    "alpha": alpha,
                    "mean_replacements_per_bus": float(mean_replacements),
                    "se_mean_replacements_per_bus": float(se_replacements),
                })

                print(
                    f"  Average bias: RC = {summary['bias_rc']:.3f}%, "
                    f"theta = {summary['bias_theta']:.3f}%",
                    flush=True,
                )

        write_csv(summary_path, summary_rows)
        write_csv(replacement_path, replacement_rows)

        print(f"\nCompleted: {scenario_name}")
        print(f"Individual estimates: {raw_path}")
        print(f"Biases and standard errors: {summary_path}")
        print(f"Replacement counts: {replacement_path}")

    print("\nVariant 3 completed successfully.")
    print("The two alpha=0 baselines agree.")


if __name__ == "__main__":
    run_variant3()