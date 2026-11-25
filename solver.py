import numpy as np

from config import (
    REPLACEMENT_COST,
    MAINTENANCE_COST,
    BETA_AGENT,
    N_PERIODS,
    MILEAGE_STEP,
    P_ZERO,
    P_ONE_STEP,
    P_TWO_STEPS,
)

from model import utility
from transitions import (
    create_mileage_grid,
    build_transition_matrices,
)


def solve_model(
    mileage_grid,
    keep_matrix,
    replace_matrix,
    replacement_cost,
    theta_c,
    beta,
    n_periods,
):
    """
    Solve the finite-horizon model by backward induction.

    Returned arrays have:
        rows = mileage states
        columns = periods
    """
    n_states = len(mileage_grid)

    if n_periods < 1:
        raise ValueError("n_periods must be at least 1.")

    if not 0 <= beta <= 1:
        raise ValueError("beta must lie between 0 and 1.")

    expected_shape = (n_states, n_states)

    if (
        keep_matrix.shape != expected_shape
        or replace_matrix.shape != expected_shape
    ):
        raise ValueError(
            "Transition matrices must match the mileage grid."
        )

    # Current utility from each action.
    u_keep = utility(
        0, mileage_grid, replacement_cost, theta_c
    )

    u_replace = utility(
        1, 0, replacement_cost, theta_c
    )

    # The extra column represents life after the final period:
    # there is no remaining utility.
    value = np.zeros((n_states, n_periods + 1))

    log_keep = np.zeros((n_states, n_periods))
    log_replace = np.zeros((n_states, n_periods))

    # Work backwards from the final period to the first.
    for t in range(n_periods - 1, -1, -1):

        # Current utility plus discounted expected future value.
        q_keep = (
            u_keep
            + beta * (keep_matrix @ value[:, t + 1])
        )

        q_replace = (
            u_replace
            + beta * (replace_matrix @ value[:, t + 1])
        )

        # Numerically stable version of:
        # log(exp(q_keep) + exp(q_replace))
        value[:, t] = np.logaddexp(q_keep, q_replace)

        # Log choice probabilities, also calculated stably.
        log_keep[:, t] = -np.logaddexp(
            0.0, q_replace - q_keep
        )

        log_replace[:, t] = -np.logaddexp(
            0.0, q_keep - q_replace
        )

    return {
        "value": value[:, :n_periods],
        "replace_probability": np.exp(log_replace),
        "log_keep_probability": log_keep,
        "log_replace_probability": log_replace,
    }


if __name__ == "__main__":
    mileage_grid = create_mileage_grid()

    keep_matrix, replace_matrix = build_transition_matrices(
        len(mileage_grid),
        P_ZERO,
        P_ONE_STEP,
        P_TWO_STEPS,
    )

    solution = solve_model(
        mileage_grid=mileage_grid,
        keep_matrix=keep_matrix,
        replace_matrix=replace_matrix,
        replacement_cost=REPLACEMENT_COST,
        theta_c=MAINTENANCE_COST,
        beta=BETA_AGENT,
        n_periods=N_PERIODS,
    )

    probabilities = solution["replace_probability"]

    # Check that values are finite and probabilities are valid.
    assert np.all(np.isfinite(solution["value"]))
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0) & (probabilities <= 1))

    keep_probabilities = np.exp(
        solution["log_keep_probability"]
    )

    assert np.allclose(
        keep_probabilities + probabilities, 1.0
    )

    print("Value function shape:", solution["value"].shape)
    print("Probability matrix shape:", probabilities.shape)

    # Inspect the state corresponding to 10,000 km.
    state_index = 10000 // MILEAGE_STEP

    print(
        "Replacement probability at 10,000 km, first period:",
        f"{probabilities[state_index, 0]:.6f}",
    )

    print(
        "Replacement probability at 10,000 km, last period:",
        f"{probabilities[state_index, -1]:.6f}",
    )

    print("All checks passed.")