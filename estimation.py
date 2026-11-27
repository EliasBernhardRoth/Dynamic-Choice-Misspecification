import numpy as np
import config as cfg

from transitions import (
    create_mileage_grid,
    build_transition_matrices,
)
from solver import solve_model
from simulation import simulate_data


def estimate_transition_probabilities(
    data,
    mileage_step=cfg.MILEAGE_STEP,
):
    """
    Estimate transition probabilities using observations
    without engine replacement.

    Returns: [p_zero, p_one_step, p_two_steps]
    """
    data = np.asarray(data)

    # Select observations without replacement.
    kept = data[:, 0] == 0

    if not np.any(kept):
        raise ValueError("No non-replacement observations.")

    # Calculate observed mileage increases.
    mileage = data[kept, 1]
    next_mileage = data[kept, 2]

    increments = next_mileage - mileage

    # Check that all mileage increases are valid.
    allowed_increments = [0, mileage_step, 2 * mileage_step]

    if not np.all(np.isin(increments, allowed_increments)):
        raise ValueError("Unexpected mileage increases.")

    # Estimate probabilities using relative frequencies.
    p_zero_hat = np.mean(increments == 0)
    p_one_step_hat = np.mean(increments == mileage_step)
    p_two_steps_hat = 1 - p_zero_hat - p_one_step_hat

    return np.array([
        p_zero_hat,
        p_one_step_hat,
        p_two_steps_hat,
    ])


if __name__ == "__main__":

    # Test the estimator on a small example.
    step = cfg.MILEAGE_STEP

    example_data = np.array([
        [0, 0, 0, 1, 1],
        [0, 0, step, 1, 2],
        [0, step, 3 * step, 1, 3],
        [1, 3 * step, 0, 1, 4],
    ])

    example_estimates = estimate_transition_probabilities(example_data)

    assert np.allclose(example_estimates, [1/3, 1/3, 1/3])

    # Generate a baseline dataset.
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

    # Estimate transition probabilities.
    p_hat = estimate_transition_probabilities(data)

    assert np.all(p_hat >= 0)

    print("Stage-one estimates")

    print(
        f"P(0 km):      true = {cfg.P_ZERO:.6f}, "
        f"estimated = {p_hat[0]:.6f}"
    )

    print(
        f"P(5,000 km):  true = {cfg.P_ONE_STEP:.6f}, "
        f"estimated = {p_hat[1]:.6f}"
    )

    print(
        f"P(10,000 km): true = {cfg.P_TWO_STEPS:.6f}, "
        f"estimated = {p_hat[2]:.6f}"
    )

    print(
        "\nNon-replacement observations:",
        np.sum(data[:, 0] == 0),
    )

    print("All checks passed.")