from config import MAINTENANCE_COST, REPLACEMENT_COST


def maintenance_cost(mileage, theta_c):
    """Calculate maintenance cost at the current mileage."""
    return theta_c * mileage


def utility(action, mileage, replacement_cost, theta_c):
    """
    Calculate utility before the random utility shock.

    action = 0: keep the engine
    action = 1: replace the engine
    """
    if action == 0:
        return -maintenance_cost(mileage, theta_c)

    elif action == 1:
        return -(replacement_cost + maintenance_cost(0, theta_c))

    else:
        raise ValueError("action must be 0 (keep) or 1 (replace)")


# These checks run only when you run this file directly.
if __name__ == "__main__":
    mileage = 10000

    cost = maintenance_cost(mileage, MAINTENANCE_COST)

    keep_utility = utility(
        0, mileage, REPLACEMENT_COST, MAINTENANCE_COST
    )

    replace_utility = utility(
        1, mileage, REPLACEMENT_COST, MAINTENANCE_COST
    )

    print(f"Maintenance cost: {cost:.3f}")
    print(f"Utility of keeping: {keep_utility:.3f}")
    print(f"Utility of replacing: {replace_utility:.3f}")