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

    horizon = 0: consider only the current period.
    horizon = 6: consider today plus six future periods.

    Planning never extends beyond the final simulation period.
    """
    if not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer.")

    if not isinstance(n_periods, (int, np.integer)) or n_periods < 1:
        raise ValueError("n_periods must be a positive integer.")

    horizon = int(horizon)
    n_periods = int(n_periods)

    # Include the current period in the planning window.
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

    # Number of actual periods remaining, including today.
    remaining_periods = n_periods - np.arange(n_periods)

    # Near the end, the agent cannot plan beyond the final period.
    effective_periods = np.minimum(planning_periods, remaining_periods)

    # Costs and transitions do not depend on calendar time.
    # Therefore, we can reuse the corresponding columns of the
    # shorter model instead of solving every planning window again.
    column_indices = planning_periods - effective_periods

    return {
        name: values[:, column_indices]
        for name, values in short_solution.items()
    }


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

    # With no foresight, replacement depends only on current costs.
    myopic_probability = np.exp(
        -np.logaddexp(
            0.0,
            cfg.REPLACEMENT_COST - cfg.MAINTENANCE_COST * mileage_grid,
        )
    )

    horizons = (0, 2, 4, 6, 8, 25, 40, 60)

    example_index = int(np.argmin(np.abs(mileage_grid - 10000)))
    example_mileage = int(mileage_grid[example_index])

    print(f"Replacement probabilities at {example_mileage:,} km")
    print(f"{'Horizon':>8} {'First period':>15} {'Last period':>15}")

    for horizon in horizons:
        limited_solution = solve_limited_foresight(
            horizon=horizon,
            **model_arguments,
        )

        probabilities = limited_solution["replace_probability"]

        assert probabilities.shape == (
            len(mileage_grid),
            cfg.N_PERIODS,
        )
        assert np.all(np.isfinite(probabilities))
        assert np.all((probabilities >= 0) & (probabilities <= 1))

        # At h=0, every period must use the myopic decision rule.
        if horizon == 0:
            expected = np.repeat(
                myopic_probability[:, None],
                cfg.N_PERIODS,
                axis=1,
            )

            np.testing.assert_allclose(
                probabilities,
                expected,
                rtol=1e-12,
                atol=1e-12,
            )

        # Once the planning window covers all remaining periods,
        # limited foresight must match the full-horizon solution.
        # For h >= T-1, this checks the entire solution.
        tail_length = min(horizon + 1, cfg.N_PERIODS)

        for name in full_solution:
            np.testing.assert_allclose(
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