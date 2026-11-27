import numpy as np

import config as cfg
from transitions import create_mileage_grid, build_transition_matrices
from solver import solve_model


def solve_limited_foresight(
    mileage_grid,
    keep_matrix,
    replace_matrix,
    replacement_cost,
    theta_c,
    beta,
    n_periods,
    horizon,
):
    """
    Solve the agent's model with limited foresight.

    horizon = 0: current period only.
    horizon = 6: current period plus six future periods.
    """

    if horizon < 0:
        raise ValueError("Horizon cannot be negative.")

    if n_periods < 1:
        raise ValueError("At least one period is required.")

    # Include the current period in the planning horizon.
    planning_periods = min(horizon + 1, n_periods)

    short_solution = solve_model(
        mileage_grid=mileage_grid,
        keep_matrix=keep_matrix,
        replace_matrix=replace_matrix,
        replacement_cost=replacement_cost,
        theta_c=theta_c,
        beta=beta,
        n_periods=planning_periods,
    )

    # Determine the relevant solution column for each period.
    column_indices = []

    for t in range(n_periods):
        remaining_periods = n_periods - t
        effective_periods = min(planning_periods, remaining_periods)

        column = planning_periods - effective_periods
        column_indices.append(column)

    # Extend the shorter solution to the full simulation period.
    solution = {}

    for name in short_solution:
        solution[name] = short_solution[name][:, column_indices]

    return solution


def main():

    mileage_grid = create_mileage_grid()

    keep_matrix, replace_matrix = build_transition_matrices(
        len(mileage_grid),
        cfg.P_ZERO,
        cfg.P_ONE_STEP,
        cfg.P_TWO_STEPS,
    )

    model_arguments = {
        "mileage_grid": mileage_grid,
        "keep_matrix": keep_matrix,
        "replace_matrix": replace_matrix,
        "replacement_cost": cfg.REPLACEMENT_COST,
        "theta_c": cfg.MAINTENANCE_COST,
        "beta": cfg.BETA_AGENT,
        "n_periods": cfg.N_PERIODS,
    }

    full_solution = solve_model(**model_arguments)

    # Myopic replacement probability based on current costs.
    myopic_probability = np.exp(
        -np.logaddexp(
            0,
            cfg.REPLACEMENT_COST - cfg.MAINTENANCE_COST * mileage_grid,
        )
    )

    horizons = (0, 2, 4, 6, 8, 25, 40, 60)

    example_index = np.argmin(np.abs(mileage_grid - 10000))
    example_mileage = int(mileage_grid[example_index])

    print(f"Replacement probabilities at {example_mileage:,} km")
    print(f"{'Horizon':>8} {'First period':>15} {'Last period':>15}")

    for horizon in horizons:

        limited_solution = solve_limited_foresight(
            horizon=horizon,
            **model_arguments,
        )

        probabilities = limited_solution["replace_probability"]

        assert probabilities.shape == (len(mileage_grid), cfg.N_PERIODS)
        assert np.all(np.isfinite(probabilities))

        # At h=0, all periods must use the myopic decision rule.
        if horizon == 0:

            for t in range(cfg.N_PERIODS):
                assert np.allclose(
                    probabilities[:, t],
                    myopic_probability,
                    rtol=1e-12,
                    atol=1e-12,
                )

        # Limited and full foresight coincide when the agent
        # considers all remaining periods.
        tail_length = min(horizon + 1, cfg.N_PERIODS)

        for name in full_solution:

            assert np.allclose(
                limited_solution[name][:, -tail_length:],
                full_solution[name][:, -tail_length:],
                rtol=1e-12,
                atol=1e-12,
            )

        print(
            f"{horizon:>8d}"
            f"{probabilities[example_index, 0]:>15.6f}"
            f"{probabilities[example_index, -1]:>15.6f}"
        )

    print("\nAll limited-foresight checks passed.")


if __name__ == "__main__":
    main()