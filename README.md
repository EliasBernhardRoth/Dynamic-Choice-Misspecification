# Dynamic Discrete Choice Models under Misspecification

**Stochastic Optimisation · Statistical Modelling · Structural Estimation · Monte Carlo Simulation**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Research](https://img.shields.io/badge/Research-Structural_Econometrics-555555)

## Overview

When does a misspecified model still produce reliable parameter estimates?

This project develops a Python framework to investigate how incorrect assumptions about sequential decision-making under uncertainty affect statistical inference. Building on Rust's (1987) dynamic discrete choice model, it combines dynamic programming, stochastic simulation and two-stage maximum likelihood estimation in controlled Monte Carlo experiments.

**Key finding:** In the baseline calibration, allowing agents to consider just six future periods reduces the relative bias of both estimated cost parameters to below 1%, despite the estimation model assuming fully forward-looking behaviour.

The project builds on original undergraduate research recognised with the **IHK Award for Best Thesis**.

[Technical methodology](METHODOLOGY.md) · [Simulation outputs](results/) · [Figures](figures/) · [Statistical tables](tables/)

<br>

## Project Highlights

| Experimental scale | Value |
| :--- | ---: |
| Simulated decision observations per specification | **3,600,000** |
| Monte Carlo repetitions per specification | 60 |
| Simulated agents per repetition | 1,000 |
| Decision periods per agent | 60 |
| Discrete states | 71 |
| Dimensions of misspecification | 3 |
| Additional robustness calibrations | 4 |

**Computational methods:** finite-horizon dynamic programming, backward induction, Markov transition matrices, likelihood maximisation, constrained numerical optimisation and Monte Carlo error quantification.

<br>

## Selected Results

All figures and numerical results in this section are from the Python implementation (60 Monte Carlo repetitions per specification).

### Limited Foresight

![Relative cost parameter biases across planning horizons](figures/R60/variant2_utility.png)

Short planning horizons produce substantial estimation errors. With six future periods, however, the biases become small in magnitude, and average replacement behaviour is close to the full-foresight benchmark.

| Future periods considered | Replacement cost bias | Maintenance coefficient bias | Replacements per bus |
| ---: | ---: | ---: | ---: |
| 0 | +6.55% | -64.37% | 5.75 |
| 2 | +30.45% | +39.59% | 11.77 |
| 4 | -4.65% | -4.10% | 11.56 |
| **6** | **+0.53%** | **+0.10%** | **11.47** |
| 8 | -0.06% | +0.05% | 11.50 |
| 60 (full foresight) | +0.08% | +0.06% | 11.50 |

The six-period result concerns practical accuracy in this calibration, not exact unbiasedness: the replacement-cost bias at six periods remains statistically distinguishable from zero using the plotted Monte Carlo interval. The horizon required for accurate estimation is model-dependent.

[Complete horizon results](results/variant2/summary_R60.csv) · [Replacement frequencies](results/variant2/replacement_counts_R60.csv)

<br>

### Discount Factor Misspecification

![Relative cost parameter biases under discount factor misspecification](figures/R60/variant1_utility.png)

When the agent discounts the future more heavily than assumed by the researcher (e.g. 0.6 instead of 0.8), the replacement cost is overestimated by **15.25%**, while the maintenance coefficient is underestimated by **11.31%**. Incorrect assumptions about intertemporal preferences can therefore be absorbed into estimated cost parameters.

[Underlying estimates](results/variant1/summary_R60.csv)

<br>

### Subjective Transition Beliefs

![Relative cost parameter biases under subjective transition beliefs](figures/R60/variant3_utility.png)

Even when objective transition probabilities are accurately estimated, decisions based on different subjective beliefs can generate substantial utility parameter bias. In the extreme optimistic-belief scenario, the estimated replacement cost is approximately **45% below** its true value.

[Optimistic beliefs](results/variant3/100/summary_R60.csv) · [Pessimistic beliefs](results/variant3/001/summary_R60.csv)

<br>

### Robustness Across Calibrations

![Limited-foresight robustness experiments under alternative calibrations](figures/R60/robustness_utility.png)

Four additional experiments vary the replacement cost, maintenance cost, mileage transition probabilities and discount factor to assess the sensitivity of the baseline findings.

**Central takeaway:** Across the main experiments, objective transition estimates are considerably less sensitive than utility parameters. The consequences of misspecification depend on how closely the assumed model represents the actual decision-making process, not simply on whether its assumptions are formally correct.

*Error bars in the figures represent ±1.96 Monte Carlo standard errors of the estimated mean bias.*

<br>

## Model and Experimental Design

A fleet manager repeatedly chooses whether to maintain a bus engine or replace it. Maintenance avoids an upfront replacement cost but becomes more expensive as mileage accumulates; replacement resets the mileage state. Future mileage growth is stochastic.

The researcher observes actions and realised mileage and estimates the underlying cost and transition parameters. The experiments vary the agent's behaviour while imposing a different behavioural model during estimation.

| Experiment | Assumption varied |
| :--- | :--- |
| Discount factor | How heavily agents discount future payoffs |
| Limited foresight | How many future periods agents consider |
| Subjective transition beliefs | How agents perceive future mileage growth |

Each experiment solves the decision problem, simulates a panel of choices and states, estimates the model under potentially incorrect assumptions, and measures relative bias against the known data-generating parameters. Repeated simulation separates systematic estimation errors from sampling variability.

<br>

## Python Implementation

The code separates model specification, numerical solution, data generation, parameter estimation and statistical evaluation.

| Component | Implementation |
| :--- | :--- |
| Dynamic programming | [solver.py](solver.py) · [limited_foresight.py](limited_foresight.py) |
| State transition matrices | [transitions.py](transitions.py) |
| Stochastic simulation | [simulation.py](simulation.py) |
| Two-stage maximum likelihood | [estimation.py](estimation.py) · [utility_estimation.py](utility_estimation.py) |
| Monte Carlo experiments | [Variant 1](monte_carlo_variant1.py) · [Variant 2](monte_carlo_variant2.py) · [Variant 3](monte_carlo_variant3.py) |
| Robustness and visualisation | [monte_carlo_robustness.py](monte_carlo_robustness.py) · [plot_results.py](plot_results.py) · [plot_appendix.py](plot_appendix.py) |

### Numerical considerations

- Aggregate observations by action, state and period to avoid repeatedly processing the full panel during likelihood evaluation.
- Rescale the maintenance coefficient to improve numerical conditioning during optimisation.
- Use bounded L-BFGS-B optimisation with a Nelder–Mead fallback and convergence checks.
- Use deterministic simulation seeds and retain repetition-level estimates, likelihood values and optimiser diagnostics.
- Validate scenario grids and numerical outputs before generating figures.

The methods and implementation address problems relevant to quantitative research: parameter recovery, stochastic optimisation, estimation under model risk and computational experimentation.

<br>

## Quick Start

Clone the repository and install the numerical dependencies in a Python 3 environment:

```bash
git clone https://github.com/EliasBernhardRoth/Dynamic-Choice-Misspecification.git
cd Dynamic-Choice-Misspecification
python3 -m pip install numpy scipy matplotlib
```

**Run a single baseline estimation:**

```bash
python3 utility_estimation.py
```

This simulates one dataset and prints the estimated and true cost parameters, initial and final negative log-likelihoods, and optimiser diagnostics.

**Regenerate the figures and appendix tables from the included simulation outputs:**

```bash
python3 plot_results.py
python3 plot_appendix.py
```

**Reproduce the full Monte Carlo experiments:**

```bash
python3 monte_carlo_variant1.py
python3 monte_carlo_variant2.py
python3 monte_carlo_variant3.py
python3 monte_carlo_robustness.py --scenario all
python3 plot_results.py
python3 plot_appendix.py
```

The default configuration uses 60 repetitions per specification. Full experiments are computationally more demanding than the single-estimation demonstration and rewrite the corresponding output files. Experiment settings are defined in [config.py](config.py).

<br>

## Repository Structure

```text
.
├── README.md
├── METHODOLOGY.md
├── config.py
├── model.py
├── solver.py
├── transitions.py
├── limited_foresight.py
├── simulation.py
├── estimation.py
├── utility_estimation.py
├── monte_carlo_variant1.py
├── monte_carlo_variant2.py
├── monte_carlo_variant3.py
├── monte_carlo_robustness.py
├── plot_results.py
├── plot_appendix.py
├── figures/
├── results/
└── tables/
```

**Technology stack:** Python · NumPy · SciPy · Matplotlib · Git

<br>

## Research Provenance and Limitations

This repository is a Python implementation of original research conducted at the Düsseldorf Institute for Competition Economics (DICE), Heinrich Heine University Düsseldorf. The original research was recognised with the **IHK Award for Best Thesis**.

The numerical results reported above come from the Python experiments. The implementation uses the documented model calibrations; exact numerical agreement with the original experiments is not assumed. In particular, the higher-maintenance-cost robustness experiment does not reproduce the replacement frequencies reported in the original Table A2. The discrepancy is retained in the [comparison output](results/robustness/maintenance_cost/table_A2_comparison_R60.csv).

For mathematical derivations, parameter values and estimation procedures, see the [Technical Methodology](METHODOLOGY.md).

<br>

## Reference

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. *Econometrica*, 55(5), 999–1033.

---

**Elias Roth**  
MPhil Economics | University of Oxford  
[GitHub Profile](https://github.com/EliasBernhardRoth)
