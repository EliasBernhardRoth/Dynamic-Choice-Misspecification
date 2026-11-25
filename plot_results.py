import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import config as cfg


PROJECT_DIRECTORY = Path(__file__).resolve().parent
RESULTS_DIRECTORY = PROJECT_DIRECTORY / "results"
OUTPUT_DIRECTORY = (
    PROJECT_DIRECTORY / "figures" / f"R{cfg.N_REPETITIONS}"
)

PARAMETER_NAMES = ("rc", "theta", "px0", "px1", "px2")

UTILITY_PARAMETERS = (
    ("rc", r"Replacement cost $RC$"),
    ("theta", r"Maintenance cost coefficient $\theta_c$"),
)

TRANSITION_PARAMETERS = (
    ("px0", r"$p_0$: 0 km", "#F8766D", "o"),
    ("px1", r"$p_1$: 5,000 km", "#00BA38", "s"),
    ("px2", r"$p_2$: 10,000 km", "#619CFF", "^"),
)


def load_summary(folder, x_column, expected_values):
    """Read a summary CSV and check that all scenarios are present."""
    path = (
        RESULTS_DIRECTORY
        / folder
        / f"summary_R{cfg.N_REPETITIONS}.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing results file:\n{path}\n"
            "Run the corresponding Monte Carlo script first. "
            "N_REPETITIONS in config.py must match the saved results."
        )

    required_columns = [x_column]

    for parameter in PARAMETER_NAMES:
        required_columns.extend(
            [f"bias_{parameter}", f"se_{parameter}"]
        )

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        missing = set(required_columns) - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"The results file is empty: {path}")

    try:
        data = {
            column: np.array(
                [float(row[column]) for row in rows],
                dtype=float,
            )
            for column in required_columns
        }
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid numerical values in {path}") from error

    if any(not np.all(np.isfinite(values)) for values in data.values()):
        raise ValueError(f"Non-finite values found in {path}")

    for parameter in PARAMETER_NAMES:
        if np.any(data[f"se_{parameter}"] < 0):
            raise ValueError(f"Negative standard errors found in {path}")

    # Sort numerically so lines connect scenarios in the correct order.
    order = np.argsort(data[x_column])
    data = {column: values[order] for column, values in data.items()}

    expected_values = np.asarray(expected_values, dtype=float)

    if (
        len(data[x_column]) != len(expected_values)
        or not np.allclose(
            data[x_column],
            expected_values,
            rtol=0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            f"Unexpected or incomplete scenario grid in {path}.\n"
            f"Expected: {expected_values}\n"
            f"Found: {data[x_column]}"
        )

    return data


def configure_style():
    """Use a consistent style that works across computers."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.fontsize": 11,
            "legend.framealpha": 1.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def format_axis(ax, data, x_column, x_label, title):
    ax.set_title(title, pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Relative bias (%)")

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        zorder=0,
    )

    if x_column == "h":
        # Preserve numerical spacing between horizons.
        # All eight simulated horizons are still plotted.
        ax.set_xticks([0, 6, 20, 40, 60])
    else:
        ax.set_xticks(data[x_column])

    ax.margins(x=0.04)


def draw_bias(ax, data, x_column, parameter, color, marker="o", label=None):
    """Draw the bias and its Monte Carlo uncertainty."""
    ax.errorbar(
        data[x_column],
        data[f"bias_{parameter}"],
        yerr=1.96 * data[f"se_{parameter}"],
        fmt=f"{marker}-",
        color=color,
        linewidth=1.8,
        markersize=5,
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        label=label,
        zorder=3,
    )


def save_figure(fig, filename, title):
    fig.suptitle(title, fontsize=18)

    fig.text(
        0.5,
        0.015,
        (
            "Error bars: ±1.96 Monte Carlo SE; "
            f"R = {cfg.N_REPETITIONS} repetitions."
        ),
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    for extension in ("png", "pdf"):
        path = OUTPUT_DIRECTORY / f"{filename}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}", flush=True)

    plt.close(fig)


def plot_utility(
    datasets,
    scenario_labels,
    x_column,
    x_label,
    color,
    filename,
    title,
):
    """
    Variants 1 and 2: two panels side by side.
    Variant 3: two rows, one for each belief scenario.
    """
    n_rows = len(datasets)

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, 5 * n_rows),
        squeeze=False,
    )

    for row_index, (data, scenario_label) in enumerate(
        zip(datasets, scenario_labels)
    ):
        for column_index, (parameter, parameter_title) in enumerate(
            UTILITY_PARAMETERS
        ):
            ax = axes[row_index, column_index]

            panel_title = parameter_title
            if scenario_label:
                panel_title = f"{scenario_label}\n{parameter_title}"

            draw_bias(
                ax,
                data,
                x_column,
                parameter,
                color,
            )

            format_axis(
                ax,
                data,
                x_column,
                x_label,
                panel_title,
            )

    save_figure(fig, filename, title)


def plot_transitions(
    datasets,
    scenario_labels,
    x_column,
    x_label,
    filename,
    title,
):
    """
    Plot all three transition parameters together.
    Variant 3 has two vertically stacked panels.
    """
    n_rows = len(datasets)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(11 if n_rows > 1 else 10, 5 * n_rows),
        squeeze=False,
    )

    for row_index, (data, scenario_label) in enumerate(
        zip(datasets, scenario_labels)
    ):
        ax = axes[row_index, 0]

        # Match the original R plots' default vertical range.
        # Expand it if necessary to keep every error bar visible.
        lower_limit = -10.0
        upper_limit = 30.0

        for parameter, label, color, marker in TRANSITION_PARAMETERS:
            draw_bias(
                ax,
                data,
                x_column,
                parameter,
                color,
                marker=marker,
                label=label,
            )

            bias = data[f"bias_{parameter}"]
            error = 1.96 * data[f"se_{parameter}"]

            lower_limit = min(
                lower_limit,
                float(np.min(bias - error)) - 1.0,
            )
            upper_limit = max(
                upper_limit,
                float(np.max(bias + error)) + 1.0,
            )

        format_axis(
            ax,
            data,
            x_column,
            x_label,
            scenario_label,
        )

        ax.set_ylim(lower_limit, upper_limit)

        ax.legend(
            loc="upper right",
            frameon=True,
            edgecolor="black",
        )

    save_figure(fig, filename, title)


def main():
    unit_grid = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    horizon_grid = (0, 2, 4, 6, 8, 25, 40, 60)

    # Read and validate every input before producing figures.
    variant1 = load_summary("variant1", "beta", unit_grid)
    variant2 = load_summary("variant2", "h", horizon_grid)

    variant3_pessimistic = load_summary(
        "variant3/001", "alpha", unit_grid
    )
    variant3_optimistic = load_summary(
        "variant3/100", "alpha", unit_grid
    )

    configure_style()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    single_scenarios = (
        (
            1,
            variant1,
            "beta",
            r"Agent discount factor $\beta^{\mathrm{agent}}$",
            "#009ACD",
        ),
        (
            2,
            variant2,
            "h",
            r"Agent foresight $h$ (future periods)",
            "#FFA500",
        ),
    )

    for number, data, x_column, x_label, color in single_scenarios:
        plot_utility(
            datasets=[data],
            scenario_labels=[""],
            x_column=x_column,
            x_label=x_label,
            color=color,
            filename=f"variant{number}_utility",
            title=f"Variant {number}: utility parameter biases",
        )

        plot_transitions(
            datasets=[data],
            scenario_labels=[""],
            x_column=x_column,
            x_label=x_label,
            filename=f"variant{number}_transitions",
            title=f"Variant {number}: transition probability biases",
        )

    belief_datasets = [
        variant3_pessimistic,
        variant3_optimistic,
    ]

    belief_labels = [
        "Pessimistic beliefs: (0, 0, 1)",
        "Optimistic beliefs: (1, 0, 0)",
    ]

    plot_utility(
        datasets=belief_datasets,
        scenario_labels=belief_labels,
        x_column="alpha",
        x_label=r"Belief distortion $\alpha$",
        color="#228B22",
        filename="variant3_utility",
        title="Variant 3: utility parameter biases",
    )

    plot_transitions(
        datasets=belief_datasets,
        scenario_labels=belief_labels,
        x_column="alpha",
        x_label=r"Belief distortion $\alpha$",
        filename="variant3_transitions",
        title="Variant 3: transition probability biases",
    )

    print("\nAll six figures saved as PNG and PDF.")
    print(f"Open this folder to view them:\n{OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()