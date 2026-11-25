import numpy as np
from scipy.optimize import minimize

import config as cfg

from transitions import (
    create_mileage_grid,
    build_transition_matrices,
)
from solver import solve_model
from simulation import simulate_data
from estimation import estimate_transition_probabilities


def prepare_choice_counts(data, mileage_grid):
    """
    Count observations by action, mileage state and period.

    This avoids repeatedly processing every individual observation
    during optimisation.
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 2 or data.shape[1] != 5 or len(data) == 0:
        raise ValueError("Data must be a non-empty array with five columns.")

    if not np.all(np.isfinite(data)):
        raise ValueError("Data must contain only finite values.")

    if not np.all(np.isin(data[:, 0], [0, 1])):
        raise ValueError("Actions must be 0 or 1.")

    periods = data[:, 4]

    if np.any(periods < 1) or np.any(periods != np.floor(periods)):
        raise ValueError("Periods must be positive integers.")

    # Locate each observed mileage in the model's grid.
    state_indices = np.searchsorted(mileage_grid, data[:, 1])

    if np.any(state_indices >= len(mileage_grid)):
        raise ValueError("Observed mileage exceeds the model's grid.")

    if not np.array_equal(
        mileage_grid[state_indices], data[:, 1]
    ):
        raise ValueError("Observed mileage must lie on the model's grid.")

    actions = data[:, 0].astype(int)
    time_indices = periods.astype(int) - 1

    n_states = len(mileage_grid)
    n_periods = int(periods.max())

    counts = np.zeros(
        (2, n_states, n_periods),
        dtype=np.int64,
    )

    np.add.at(
        counts,
        (actions, state_indices, time_indices),
        1,
    )

    return counts


def negative_log_likelihood(
    parameters,
    choice_counts,
    mileage_grid,
    keep_matrix,
    replace_matrix,
    beta,
):
    """
    Calculate the negative choice log-likelihood.

    parameters[0] = replacement cost
    parameters[1] = maintenance-cost coefficient
    """
    replacement_cost, theta_c = parameters

    if (
        not np.all(np.isfinite(parameters))
        or replacement_cost < 0
        or theta_c < 0
    ):
        return np.inf

    n_periods = choice_counts.shape[2]

    solution = solve_model(
        mileage_grid=mileage_grid,
        keep_matrix=keep_matrix,
        replace_matrix=replace_matrix,
        replacement_cost=replacement_cost,
        theta_c=theta_c,
        beta=beta,
        n_periods=n_periods,
    )

    log_likelihood = np.sum(
        choice_counts[0] * solution["log_keep_probability"]
        + choice_counts[1] * solution["log_replace_probability"]
    )

    return -float(log_likelihood)


def estimate_utility_parameters(
    data,
    transition_probabilities,
    beta=cfg.BETA_MODEL,
):
    """
    Estimate replacement and maintenance-cost parameters.

    Transition probabilities are held fixed at their stage-one
    estimates. The discount factor is the researcher's assumption.
    """
    mileage_grid = create_mileage_grid()
    choice_counts = prepare_choice_counts(data, mileage_grid)

    p_hat = np.asarray(transition_probabilities, dtype=float)

    if p_hat.shape != (3,):
        raise ValueError("Exactly three transition probabilities are required.")

    keep_matrix, replace_matrix = build_transition_matrices(
        len(mileage_grid),
        p_hat[0],
        p_hat[1],
        p_hat[2],
    )

    n_observations = int(choice_counts.sum())

    # Same starting values and bounds as the MATLAB code.
    initial_parameters = np.array([14.0, 0.0001])

    # Optimise the maintenance coefficient in cost-per-mileage-step
    # units. This improves numerical scaling.
    scale = np.array([1.0, float(cfg.MILEAGE_STEP)])

    initial_scaled = initial_parameters * scale

    scaled_bounds = [
        (0.0, 10000.0),
        (0.0, float(cfg.MILEAGE_STEP)),
    ]

    def objective(scaled_parameters):
        parameters = scaled_parameters / scale

        nll = negative_log_likelihood(
            parameters,
            choice_counts,
            mileage_grid,
            keep_matrix,
            replace_matrix,
            beta,
        )

        # Averaging changes the scale, but not the minimiser.
        return nll / n_observations

    initial_nll = objective(initial_scaled) * n_observations

    method = "L-BFGS-B"

    result = minimize(
        objective,
        initial_scaled,
        method=method,
        bounds=scaled_bounds,
        options={
            "ftol": 1e-15,
            "gtol": 1e-9,
            "maxiter": 1000,
            "maxfun": 3000,
        },
    )

    total_evaluations = result.nfev

    # Retry with a different method if the first solver fails.
    if not result.success or not np.isfinite(result.fun):
        method = "Nelder-Mead"

        result = minimize(
            objective,
            initial_scaled,
            method=method,
            bounds=scaled_bounds,
            options={
                "xatol": 1e-9,
                "fatol": 1e-15,
                "maxiter": 3000,
                "maxfev": 5000,
            },
        )

        total_evaluations += result.nfev

    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(
            f"Utility estimation did not converge: {result.message}"
        )

    estimated_parameters = result.x / scale
    final_nll = float(result.fun * n_observations)

    if final_nll > initial_nll + 1e-6:
        raise RuntimeError(
            "The fitted likelihood is worse than the starting likelihood."
        )

    return {
        "replacement_cost": float(estimated_parameters[0]),
        "maintenance_cost": float(estimated_parameters[1]),
        "negative_log_likelihood": final_nll,
        "initial_negative_log_likelihood": float(initial_nll),
        "converged": bool(result.success),
        "optimizer": method,
        "function_evaluations": int(total_evaluations),
        "message": str(result.message),
    }


if __name__ == "__main__":
    # Generate one baseline dataset.
    mileage_grid = create_mileage_grid()

    keep_matrix, replace_matrix = build_transition_matrices(
        len(mileage_grid),
        cfg.P_ZERO,
        cfg.P_ONE_STEP,
        cfg.P_TWO_STEPS,
    )

    solution = solve_model(
        mileage_grid=mileage_grid,
        keep_matrix=keep_matrix,
        replace_matrix=replace_matrix,
        replacement_cost=cfg.REPLACEMENT_COST,
        theta_c=cfg.MAINTENANCE_COST,
        beta=cfg.BETA_AGENT,
        n_periods=cfg.N_PERIODS,
    )

    data = simulate_data(
        replace_probabilities=solution["replace_probability"],
        n_buses=cfg.N_BUSES,
        p_zero=cfg.P_ZERO,
        p_one_step=cfg.P_ONE_STEP,
        p_two_steps=cfg.P_TWO_STEPS,
        seed=1001,
    )

    # Stage one: estimate transition probabilities.
    p_hat = estimate_transition_probabilities(data)

    # Stage two: estimate utility parameters.
    fit = estimate_utility_parameters(
        data,
        transition_probabilities=p_hat,
        beta=cfg.BETA_MODEL,
    )

    print("Two-stage estimation completed")

    print(
        f"\nReplacement cost: "
        f"true = {cfg.REPLACEMENT_COST:.6f}, "
        f"estimated = {fit['replacement_cost']:.6f}"
    )

    print(
        f"Maintenance coefficient: "
        f"true = {cfg.MAINTENANCE_COST:.9f}, "
        f"estimated = {fit['maintenance_cost']:.9f}"
    )

    print(
        "\nStarting negative log-likelihood:",
        f"{fit['initial_negative_log_likelihood']:.6f}",
    )

    print(
        "Final negative log-likelihood:",
        f"{fit['negative_log_likelihood']:.6f}",
    )

    print("Converged:", fit["converged"])
    print("Optimiser:", fit["optimizer"])
    print("Function evaluations:", fit["function_evaluations"])