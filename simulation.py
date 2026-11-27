import numpy as np

from config import (
    N_BUSES,
    N_PERIODS,
    REPLACEMENT_COST,
    MAINTENANCE_COST,
    BETA_AGENT,
    MILEAGE_STEP,
    P_ZERO,
    P_ONE_STEP,
    P_TWO_STEPS,
)

from transitions import (
    create_mileage_grid,
    build_transition_matrices,
)

from solver import solve_model


def simulate_data(
    replace_probabilities,
    n_buses,
    p_zero,
    p_one_step,
    p_two_steps,
    seed=1001,
    mileage_step=MILEAGE_STEP,
):
    """
    Simulate choices and mileage using actual transition probabilities.

    Columns: action, mileage, next_mileage, bus_id, period.

    Action 0: keep.
    Action 1: replace.
    """
    replace_probabilities = np.asarray(
        replace_probabilities, dtype=float
    )

    n_states, n_periods = replace_probabilities.shape

    if not np.all(
        (replace_probabilities >= 0) & (replace_probabilities <= 1)
    ):
        raise ValueError("Invalid replacement probabilities.")

    probabilities = np.array([p_zero, p_one_step, p_two_steps])

    if np.any(probabilities < 0):
        raise ValueError("Transition probabilities cannot be negative.")

    if not np.isclose(np.sum(probabilities), 1, rtol=0, atol=1e-12):
        raise ValueError("Transition probabilities must sum to one.")

    if n_buses < 1 or n_periods < 1 or mileage_step <= 0:
        raise ValueError("Invalid simulation settings.")

    # Generate random draws in the same order as the original code.
    rng = np.random.RandomState(seed)

    choice_draws = rng.random_sample(
        n_buses * n_periods
    ).reshape((n_buses, n_periods), order="F")

    transition_draws = rng.random_sample(
        n_buses * n_periods
    ).reshape((n_buses, n_periods), order="F")

    # Assign mileage increases based on transition probabilities.
    increments = np.where(
        transition_draws <= p_zero + p_one_step,
        mileage_step,
        2 * mileage_step,
    )

    increments[transition_draws <= p_zero] = 0

    # Initialise arrays.
    mileage = np.zeros((n_buses, n_periods), dtype=np.int64)
    actions = np.zeros((n_buses, n_periods), dtype=np.int64)
    next_mileage = np.zeros((n_buses, n_periods), dtype=np.int64)

    # Simulate choices and mileage period by period.
    for t in range(n_periods):

        if t > 0:
            mileage[:, t] = next_mileage[:, t - 1]

        states = mileage[:, t] // mileage_step

        if np.any(states >= n_states):
            raise ValueError("Simulated mileage exceeds grid.")

        # Determine which buses replace their engines.
        actions[:, t] = (
            choice_draws[:, t] <= replace_probabilities[states, t]
        )

        # Replacement resets mileage to zero.
        starting_mileage = mileage[:, t].copy()
        starting_mileage[actions[:, t] == 1] = 0

        # Add the mileage increase.
        next_mileage[:, t] = starting_mileage + increments[:, t]

    # Create bus IDs and period numbers.
    bus_ids = np.repeat(np.arange(1, n_buses + 1), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_buses)

    # Combine all observations into one dataset.
    data = np.column_stack([
        actions.ravel(),
        mileage.ravel(),
        next_mileage.ravel(),
        bus_ids,
        periods,
    ])

    return data


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

    data = simulate_data(
        replace_probabilities=solution["replace_probability"],
        n_buses=N_BUSES,
        p_zero=P_ZERO,
        p_one_step=P_ONE_STEP,
        p_two_steps=P_TWO_STEPS,
        seed=1001,
    )

    # Check dataset dimensions.
    assert data.shape == (N_BUSES * N_PERIODS, 5)

    panel = data.reshape(N_BUSES, N_PERIODS, 5)

    # Every bus starts at zero mileage.
    assert np.all(panel[:, 0, 1] == 0)

    # Check consistency of mileage across periods.
    assert np.array_equal(panel[:, 1:, 1], panel[:, :-1, 2])

    mean_replacements = data[:, 0].sum() / N_BUSES

    print("Dataset shape:", data.shape)
    print("Columns: action, mileage, next_mileage, bus_id, period")
    print("\nFirst 10 observations:")
    print(data[:10])
    print(f"\nMean replacements per bus: {mean_replacements:.3f}")
    print("All checks passed.")