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

    Return one row per bus and period, with columns:
        action, mileage, next_mileage, bus_id, period

    action = 0: keep
    action = 1: replace

    Bus IDs and periods start at 1, matching the MATLAB data.
    """
    replace_probabilities = np.asarray(
        replace_probabilities, dtype=float
    )

    if replace_probabilities.ndim != 2:
        raise ValueError("Replacement probabilities must be a matrix.")

    if (
        not np.all(np.isfinite(replace_probabilities))
        or np.any(replace_probabilities < 0)
        or np.any(replace_probabilities > 1)
    ):
        raise ValueError("Replacement probabilities must lie in [0, 1].")

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
            "Transition probabilities must be non-negative and sum to one."
        )

    n_states, n_periods = replace_probabilities.shape

    if n_buses < 1 or n_periods < 1 or mileage_step <= 0:
        raise ValueError(
            "Bus count, period count and mileage step must be positive."
        )

    # Legacy random generator for compatibility with the original runs.
    rng = np.random.RandomState(seed)

    # MATLAB fills random matrices column by column.
    # order="F" preserves that arrangement.
    choice_draws = rng.random_sample(
        n_buses * n_periods
    ).reshape((n_buses, n_periods), order="F")

    transition_draws = rng.random_sample(
        n_buses * n_periods
    ).reshape((n_buses, n_periods), order="F")

    # Actual mileage increases: 0, one step or two steps.
    increments = np.where(
        transition_draws <= p_zero,
        0,
        np.where(
            transition_draws <= p_zero + p_one_step,
            mileage_step,
            2 * mileage_step,
        ),
    )

    mileage = np.zeros((n_buses, n_periods), dtype=np.int64)
    actions = np.zeros((n_buses, n_periods), dtype=np.int64)
    next_mileage = np.zeros((n_buses, n_periods), dtype=np.int64)

    for t in range(n_periods):

        # Carry forward the previous period's resulting mileage.
        if t > 0:
            mileage[:, t] = next_mileage[:, t - 1]

        state_indices = mileage[:, t] // mileage_step

        if np.any(state_indices >= n_states):
            raise ValueError(
                "Simulated mileage exceeds the model's state grid."
            )

        # Replace if the random draw is below the choice probability.
        actions[:, t] = (
            choice_draws[:, t]
            <= replace_probabilities[state_indices, t]
        ).astype(np.int64)

        # Replacement resets mileage before the next increase.
        starting_mileage = np.where(
            actions[:, t] == 1,
            0,
            mileage[:, t],
        )

        next_mileage[:, t] = (
            starting_mileage + increments[:, t]
        )

    # Arrange observations bus by bus, then period by period.
    bus_ids = np.repeat(
        np.arange(1, n_buses + 1), n_periods
    )

    periods = np.tile(
        np.arange(1, n_periods + 1), n_buses
    )

    data = np.column_stack(
        [
            actions.ravel(order="C"),
            mileage.ravel(order="C"),
            next_mileage.ravel(order="C"),
            bus_ids,
            periods,
        ]
    )

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

    # Check the dataset's structure and timing.
    assert data.shape == (N_BUSES * N_PERIODS, 5)
    assert np.all(np.isin(data[:, 0], [0, 1]))

    panel = data.reshape(N_BUSES, N_PERIODS, 5)

    # Every bus starts at zero mileage.
    assert np.all(panel[:, 0, 1] == 0)

    # Next-period mileage becomes the following observed state.
    assert np.array_equal(
        panel[:, 1:, 1],
        panel[:, :-1, 2],
    )

    mean_replacements = data[:, 0].sum() / N_BUSES

    print("Dataset shape:", data.shape)
    print("Columns: action, mileage, next_mileage, bus_id, period")
    print("\nFirst 10 observations:")
    print(data[:10])
    print(f"\nMean replacements per bus: {mean_replacements:.3f}")
    print("All checks passed.")