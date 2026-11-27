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

    Rows represent mileage states.
    Columns represent periods.
    """

    if n_periods < 1:
        raise ValueError("At least one period is required.")

    if not 0 <= beta <= 1:
        raise ValueError("Beta must lie between 0 and 1.")

    n_states = len(mileage_grid)

    # Current utility from each action.
    u_keep = utility(0, mileage_grid, replacement_cost, theta_c)
    u_replace = utility(1, 0, replacement_cost, theta_c)

    # Initialise value function and log choice probabilities.
    value = np.zeros((n_states, n_periods + 1))
    log_keep = np.zeros((n_states, n_periods))
    log_replace = np.zeros((n_states, n_periods))

    # Backward induction.
    for t in range(n_periods - 1, -1, -1):

        # Value function in the next period.
        next_value = value[:, t + 1]

        # Choice-specific value functions.
        q_keep = u_keep + beta * (keep_matrix @ next_value)
        q_replace = u_replace + beta * (replace_matrix @ next_value)

        # Expected value function.
        value[:, t] = np.logaddexp(q_keep, q_replace)

        # Log choice probabilities.
        log_keep[:, t] = -np.logaddexp(0, q_replace - q_keep)
        log_replace[:, t] = -np.logaddexp(0, q_keep - q_replace)

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
    keep_probabilities = np.exp(solution["log_keep_probability"])

    # Check numerical and mathematical validity.
    assert np.all(np.isfinite(solution["value"]))
    assert np.allclose(keep_probabilities + probabilities, 1)

    print("Value function shape:", solution["value"].shape)
    print("Probability matrix shape:", probabilities.shape)

    # Inspect replacement probabilities at 10,000 km.
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