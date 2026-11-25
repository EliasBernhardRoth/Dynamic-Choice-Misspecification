import csv
import json
from pathlib import Path

import numpy as np

import config as cfg
from monte_carlo_variant2 import (
    HORIZON_VALUES,
    PARAMETER_NAMES,
    write_csv,
)
from monte_carlo_robustness import (
    BASELINE,
    SCENARIOS,
    compare_replacement_counts,
)
from plot_results import (
    OUTPUT_DIRECTORY as FIGURE_DIRECTORY,
    configure_style,
    load_summary,
    plot_utility,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent
RESULTS_DIRECTORY = PROJECT_DIRECTORY / "results"
TABLE_DIRECTORY = (
    PROJECT_DIRECTORY / "tables" / f"R{cfg.N_REPETITIONS}"
)

MEAN_COLUMN = "mean_replacements_per_bus"
SE_COLUMN = "se_mean_replacements_per_bus"

ROBUSTNESS_LABELS = {
    "maintenance_cost": r"(a) Maintenance cost $\theta_c = 0.001$",
    "replacement_cost": r"(b) Replacement cost $RC = 50$",
    "transitions": "(c) Actual transitions: (0.1, 0.3, 0.6)",
    "discount_factor": "(d) Agent and researcher discount factors: 0.99",
}

TABLE_LABELS = {
    "maintenance_cost": "theta_c = 0.001",
    "replacement_cost": "RC = 50",
    "transitions": "p = (0.1, 0.3, 0.6)",
    "discount_factor": "beta = 0.99",
}


def load_replacement_counts(folder, x_column, expected_values):
    """Read replacement means and their Monte Carlo standard errors."""
    path = (
        RESULTS_DIRECTORY
        / folder
        / f"replacement_counts_R{cfg.N_REPETITIONS}.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing results:\n{path}\n"
            "Complete the corresponding Monte Carlo run first."
        )

    columns = [x_column, MEAN_COLUMN, SE_COLUMN]

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty results file: {path}")

    data = {
        column: np.array(
            [float(row[column]) for row in rows],
            dtype=float,
        )
        for column in columns
    }

    if any(not np.all(np.isfinite(values)) for values in data.values()):
        raise ValueError(f"Non-finite values in {path}")

    if np.any(data[MEAN_COLUMN] < 0) or np.any(data[SE_COLUMN] < 0):
        raise ValueError(f"Negative replacement counts or standard errors: {path}")

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
        raise ValueError(f"Unexpected or incomplete scenario grid in {path}")

    return data


def check_robustness_settings(scenario_name):
    """Check completion and the parameter settings behind each figure."""
    path = (
        RESULTS_DIRECTORY
        / "robustness"
        / scenario_name
        / f"settings_R{cfg.N_REPETITIONS}.json"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing run settings:\n{path}\n"
            f"Run monte_carlo_robustness.py --scenario {scenario_name}"
        )

    settings = json.loads(path.read_text(encoding="utf-8"))

    if (
        settings.get("run_completed") is not True
        or settings.get("n_repetitions") != cfg.N_REPETITIONS
        or settings.get("horizons") != list(HORIZON_VALUES)
    ):
        raise ValueError(
            f"The {scenario_name} run is incomplete or uses different settings."
        )

    dimensions = (
        settings.get("n_buses"),
        settings.get("n_periods"),
        settings.get("mileage_step"),
        settings.get("max_mileage"),
    )

    if dimensions != (1000, 60, 5000, 350000):
        raise ValueError(f"Unexpected experiment dimensions in {path}")

    parameters = {
        **BASELINE,
        **SCENARIOS[scenario_name]["changes"],
    }

    expected = [
        parameters["replacement_cost"],
        parameters["maintenance_cost"],
        parameters["p_zero"],
        parameters["p_one_step"],
        1.0 - parameters["p_zero"] - parameters["p_one_step"],
    ]

    stored_parameters = settings.get("true_parameters", {})
    actual = [
        stored_parameters.get(name, np.nan)
        for name in PARAMETER_NAMES
    ]

    if not np.allclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError(
            f"Saved true parameters do not match the scenario definition: {path}"
        )

    if not np.allclose(
        [
            settings.get("beta_agent", np.nan),
            settings.get("beta_model", np.nan),
        ],
        [parameters["beta"], parameters["beta"]],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(f"Unexpected discount factors in {path}")


def write_mean_table(filename, title, x_column, x_label, series):
    """
    Save full-precision means as CSV and return a Markdown table.

    Each series is:
        (CSV column name, displayed column name, replacement data)
    """
    x_values = series[0][2][x_column]
    rows = []

    for index, value in enumerate(x_values):
        row = {
            x_column: int(value) if x_column == "h" else float(value)
        }

        for column, label, data in series:
            row[column] = float(data[MEAN_COLUMN][index])

        rows.append(row)

    write_csv(TABLE_DIRECTORY / f"{filename}.csv", rows)

    headers = [x_label] + [label for column, label, data in series]

    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        values = [f"{row[x_column]:g}"]
        values.extend(
            f"{row[column]:.2f}"
            for column, label, data in series
        )
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            f"[Full-precision CSV]({filename}.csv)",
            "",
        ]
    )

    return "\n".join(lines)


def write_table_a2_comparison(robustness_counts):
    """Keep the published values and Python results separately identifiable."""
    comparison_rows = []
    lines = [
        "## Comparison with thesis Table A2",
        "",
        "The comparison uses the thesis's rounded replacement counts.",
        "An entry matches the printed rounding interval when its absolute",
        "difference is at most 0.005 replacements per bus.",
        "",
        "| Scenario | Entries within rounding interval | Largest absolute difference |",
        "| --- | --- | --- |",
    ]

    for name, specification in SCENARIOS.items():
        data = robustness_counts[name]

        replacement_rows = [
            {
                "h": int(horizon),
                MEAN_COLUMN: float(data[MEAN_COLUMN][index]),
                SE_COLUMN: float(data[SE_COLUMN][index]),
            }
            for index, horizon in enumerate(data["h"])
        ]

        comparison = compare_replacement_counts(
            replacement_rows,
            specification["table_a2"],
        )

        comparison_rows.extend(
            {"scenario": name, **row}
            for row in comparison
        )

        matches = sum(
            row["within_printed_rounding_interval"]
            for row in comparison
        )
        largest_difference = max(
            abs(row["difference"])
            for row in comparison
        )

        lines.append(
            f"| {TABLE_LABELS[name]} | {matches}/{len(comparison)} | "
            f"{largest_difference:.4f} |"
        )

    write_csv(
        TABLE_DIRECTORY / "table_A2_comparison.csv",
        comparison_rows,
    )

    lines.extend(
        [
            "",
            "[Detailed comparison](table_A2_comparison.csv)",
            "",
            "Differences remain under review. Matching replacement counts",
            "does not by itself establish agreement of the utility estimates.",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    unit_grid = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    # Load the replacement tables from the three main experiments.
    variant1 = load_replacement_counts(
        "variant1", "beta", unit_grid
    )
    variant2 = load_replacement_counts(
        "variant2", "h", HORIZON_VALUES
    )
    pessimistic = load_replacement_counts(
        "variant3/001", "alpha", unit_grid
    )
    optimistic = load_replacement_counts(
        "variant3/100", "alpha", unit_grid
    )

    robustness_summaries = []
    robustness_counts = {}

    # Validate all robustness runs before creating output.
    for name in SCENARIOS:
        check_robustness_settings(name)

        robustness_summaries.append(
            load_summary(
                f"robustness/{name}",
                "h",
                HORIZON_VALUES,
            )
        )

        robustness_counts[name] = load_replacement_counts(
            f"robustness/{name}",
            "h",
            HORIZON_VALUES,
        )

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    configure_style()

    # Four rows: one per robustness scenario.
    # Two columns: replacement cost and maintenance cost.
    plot_utility(
        datasets=robustness_summaries,
        scenario_labels=[
            ROBUSTNESS_LABELS[name] for name in SCENARIOS
        ],
        x_column="h",
        x_label=r"Agent foresight $h$ (future periods)",
        color="#FFA500",
        filename="robustness_utility",
        title="Robustness checks: limited foresight",
    )

    document = [
        "# Appendix replacement tables",
        "",
        f"Monte Carlo repetitions per setting: **{cfg.N_REPETITIONS}**.",
        "",
        "Entries are mean replacements per bus over the simulated period.",
        "Markdown tables show two decimal places; CSV files preserve full precision.",
        "These are Python results; exact agreement with all thesis results",
        "has not yet been established.",
        "",
    ]

    document.append(
        write_mean_table(
            "variant1_replacements",
            "Variant 1: agent discount factor",
            "beta",
            "Agent beta",
            [("mean_replacements", "Mean replacements", variant1)],
        )
    )

    document.append(
        write_mean_table(
            "variant2_replacements",
            "Variant 2: limited foresight",
            "h",
            "h",
            [("mean_replacements", "Mean replacements", variant2)],
        )
    )

    document.append(
        write_mean_table(
            "variant3_replacements",
            "Variant 3: subjective transition beliefs",
            "alpha",
            "alpha",
            [
                ("pessimistic", "Pessimistic (0, 0, 1)", pessimistic),
                ("optimistic", "Optimistic (1, 0, 0)", optimistic),
            ],
        )
    )

    document.append(
        write_mean_table(
            "robustness_replacements",
            "Robustness checks: counterpart to Table A2",
            "h",
            "h",
            [
                (name, TABLE_LABELS[name], robustness_counts[name])
                for name in SCENARIOS
            ],
        )
    )

    # Also retain standard errors in one combined, full-precision file.
    all_datasets = [
        ("variant1", "beta", variant1),
        ("variant2", "h", variant2),
        ("variant3_pessimistic", "alpha", pessimistic),
        ("variant3_optimistic", "alpha", optimistic),
    ]

    all_datasets.extend(
        (f"robustness_{name}", "h", data)
        for name, data in robustness_counts.items()
    )

    long_rows = []

    for experiment, x_column, data in all_datasets:
        for index, value in enumerate(data[x_column]):
            long_rows.append(
                {
                    "experiment": experiment,
                    "varied_parameter": x_column,
                    "value": float(value),
                    MEAN_COLUMN: float(data[MEAN_COLUMN][index]),
                    SE_COLUMN: float(data[SE_COLUMN][index]),
                }
            )

    write_csv(
        TABLE_DIRECTORY / "all_replacement_counts.csv",
        long_rows,
    )

    document.extend(
        [
            "## Monte Carlo uncertainty",
            "",
            "[All replacement means and standard errors]"
            "(all_replacement_counts.csv)",
            "",
        ]
    )

    if cfg.N_REPETITIONS == 60:
        document.append(
            write_table_a2_comparison(robustness_counts)
        )
    else:
        document.extend(
            [
                "## Reference comparison",
                "",
                "The Table A2 comparison is generated for R = 60,",
                "matching the thesis's number of repetitions.",
                "",
            ]
        )

    report_path = TABLE_DIRECTORY / "appendix_replacements.md"
    report_path.write_text(
        "\n".join(document),
        encoding="utf-8",
    )

    print("\nAppendix outputs completed.")
    print(f"Robustness figure: {FIGURE_DIRECTORY}")
    print(f"CSV tables: {TABLE_DIRECTORY}")
    print(f"Readable tables and comparison: {report_path}")


if __name__ == "__main__":
    main()