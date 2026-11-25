import numpy as np

from config import ( 
    MILEAGE_STEP, MAX_MILEAGE, P_ZERO, P_ONE_STEP, P_TWO_STEPS,
)


def create_mileage_grid(max_mileage=MAX_MILEAGE, step=MILEAGE_STEP):
    """Create the possible mileage states, including both endpoints."""
    if step <= 0 or max_mileage < 0 or max_mileage % step != 0:
        raise ValueError(
            "Mileage step must be positive, and maximum mileage "
            "must be a non-negative multiple of the step."
        )

    return np.arange(0, max_mileage + step, step)


def build_transition_matrices(
    n_states, p_zero, p_one_step, p_two_steps
):
    """
    Return transition matrices for keeping and replacing the engine.

    Rows represent the current state.
    Columns represent the next-period state.
    """
    if n_states < 3:
        raise ValueError("At least three mileage states are required.")

    probabilities = np.array(
        [p_zero, p_one_step, p_two_steps], dtype=float
    )

    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or not np.isclose(
            probabilities.sum(), 1.0, rtol=0, atol=1e-12
        )
    ):
        raise ValueError(
            "Probabilities must be finite, non-negative and sum to one."
        )

    # Action 0: keep the engine.
    keep_matrix = np.zeros((n_states, n_states))

    for i in range(n_states):
        keep_matrix[i, i] = p_zero

        if i + 1 < n_states:
            keep_matrix[i, i + 1] = p_one_step

        if i + 2 < n_states:
            keep_matrix[i, i + 2] = p_two_steps

    # Preserve the upper-bound treatment in the MATLAB model.
    keep_matrix[-2, -1] = 1 - p_zero
    keep_matrix[-1, -1] = 1.0

    # Action 1: replace, then draw the next mileage increase.
    replacement_row = np.zeros(n_states)
    replacement_row[:3] = probabilities

    replace_matrix = np.tile(replacement_row, (n_states, 1))

    return keep_matrix, replace_matrix


if __name__ == "__main__":
    mileage_grid = create_mileage_grid()

    keep_matrix, replace_matrix = build_transition_matrices(
        len(mileage_grid),
        P_ZERO,
        P_ONE_STEP,
        P_TWO_STEPS,
    )

    # Every row must describe a probability distribution.
    assert np.allclose(keep_matrix.sum(axis=1), 1.0)
    assert np.allclose(replace_matrix.sum(axis=1), 1.0)

    assert np.all(keep_matrix >= 0)
    assert np.all(replace_matrix >= 0)

    print("Number of states:", len(mileage_grid))
    print("First five states:", mileage_grid[:5])
    print("Keep matrix shape:", keep_matrix.shape)
    print("Replace matrix shape:", replace_matrix.shape)
    print("Keep from zero mileage:", keep_matrix[0, :3])
    print("Replace from any mileage:", replace_matrix[0, :3])
    print("All checks passed.")