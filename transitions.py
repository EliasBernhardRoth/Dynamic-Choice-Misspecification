import numpy as np

from config import (
    MILEAGE_STEP, MAX_MILEAGE, P_ZERO, P_ONE_STEP, P_TWO_STEPS
)


def create_mileage_grid(max_mileage=MAX_MILEAGE, step=MILEAGE_STEP):
    """Create the possible mileage states."""

    if max_mileage % step != 0:
        raise ValueError("Maximum mileage must be a multiple of the step.")

    return np.arange(0, max_mileage + step, step)


def build_transition_matrices(
    n_states, p_zero, p_one_step, p_two_steps
):
    """
    Create transition matrices for keeping and replacing the engine.

    Rows: current mileage state.
    Columns: next-period mileage state.
    """

    probabilities = np.array([p_zero, p_one_step, p_two_steps])

    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative.")

    if not np.isclose(np.sum(probabilities), 1, rtol=0, atol=1e-12):
        raise ValueError("Probabilities must sum to one.")

    # Action 0: keep the engine.
    keep_matrix = np.zeros((n_states, n_states))

    for i in range(n_states - 2):
        keep_matrix[i, i] = p_zero
        keep_matrix[i, i + 1] = p_one_step
        keep_matrix[i, i + 2] = p_two_steps

    # Second-highest mileage state.
    keep_matrix[-2, -2] = p_zero
    keep_matrix[-2, -1] = 1 - p_zero

    # Highest mileage state.
    keep_matrix[-1, -1] = 1

    # Action 1: replace the engine.
    replace_matrix = np.zeros((n_states, n_states))

    for i in range(n_states):
        replace_matrix[i, :3] = probabilities

    return keep_matrix, replace_matrix


if __name__ == "__main__":

    mileage_grid = create_mileage_grid()

    keep_matrix, replace_matrix = build_transition_matrices(
        len(mileage_grid),
        P_ZERO,
        P_ONE_STEP,
        P_TWO_STEPS,
    )

    # Check that every row sums to one.
    assert np.allclose(keep_matrix.sum(axis=1), 1)
    assert np.allclose(replace_matrix.sum(axis=1), 1)

    print("Number of states:", len(mileage_grid))
    print("First five states:", mileage_grid[:5])
    print("Keep matrix shape:", keep_matrix.shape)
    print("Replace matrix shape:", replace_matrix.shape)
    print("Keep from zero mileage:", keep_matrix[0, :3])
    print("Replace from any mileage:", replace_matrix[0, :3])
    print("All checks passed.")