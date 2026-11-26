# Dynamic Discrete Choice Models under Misspecification

**Stochastic Optimisation · Statistical Modelling · Structural Estimation · Monte Carlo Simulation**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Research](https://img.shields.io/badge/Research-Structural_Econometrics-555555)

## Overview

When does a misspecified model still produce reliable parameter estimates?

I developed a Python simulation and structural estimation framework to investigate how incorrect assumptions about dynamic decision-making under uncertainty affect parameter recovery. Based on Rust's (1987) dynamic discrete choice model, the framework combines finite-horizon dynamic programming, stochastic simulation and two-stage maximum likelihood estimation to isolate the effects of misspecification in planning horizons, discount factors and expectations about future states.

**Key finding:** In the baseline calibration, assuming full foresight produces less than **1% relative bias** in both cost estimates when agents consider only six future periods. By contrast, misspecified transition beliefs can lead to replacement cost estimates that are approximately **45% too low**.

This research formed my bachelor's thesis at the Düsseldorf Institute for Competition Economics (DICE), Heinrich Heine University Düsseldorf, and received the **IHK Düsseldorf Award for Best Thesis**.

[Technical Methodology](METHODOLOGY.md) · [Simulation outputs](results/) · [Figures](figures/) · [Statistical tables](tables/)

<br>

## Project Highlights

| Experimental scale | Value |
| :--- | ---: |
| Simulated decision observations per specification | **3,600,000** |
| Monte Carlo repetitions per specification | 60 |
| Simulated buses × decision periods per repetition | 1,000 × 60 |
| Discrete states | 71 |
| Dimensions of misspecification | 3 |
| Additional robustness calibrations | 4 |

**Computational methods:** finite-horizon dynamic programming, backward induction, Markov transition matrices, two-stage maximum likelihood estimation, constrained numerical optimisation and Monte Carlo error quantification.

<br>

## The Decision Problem

A fleet manager decides each month whether to maintain or replace a bus engine over a 60-month operating horizon:

- **Maintain:** Avoid the upfront replacement cost but incur maintenance costs that increase with accumulated mileage.
- **Replace:** Pay the replacement cost, reset mileage and avoid maintenance costs in the current month, with potential savings in subsequent months.

Mileage growth is stochastic: a bus travels either 0, 5,000 or 10,000 km each month. The value of replacement therefore depends on how the agent evaluates future costs and mileage growth.

The researcher observes choices and realised mileage but not the underlying costs. The framework estimates these in two stages:

1. Estimate the objective mileage transition probabilities from observed state transitions.
2. Estimate the replacement cost (**RC**) and maintenance cost coefficient (**θ<sub>c</sub>**) by maximising the conditional-choice likelihood under the researcher's behavioural assumptions.

If these assumptions are incorrect, the model may explain differences in behaviour by distorting the estimated costs rather than recovering the true structural parameters.

<br>

## Selected Results

In the three main experiments, the researcher assumes full foresight, a discount factor of 0.8 and correct expectations about future mileage growth. Each experiment varies one aspect of the agent's behaviour while holding the remaining assumptions fixed.

The figures show relative parameter bias across 60 Monte Carlo repetitions per specification. Positive values indicate overestimation; negative values indicate underestimation. Error bars represent ±1.96 Monte Carlo standard errors of the estimated mean bias.

### 1. Limited Foresight

**Researcher's assumption:** The agent considers all remaining months of the operating horizon.

**Agent's actual behaviour:** The agent considers at most the next *h* months at each decision. With *h* = 0, only immediate costs matter; with *h* = 60, the agent has full foresight.

![Relative cost parameter biases across planning horizons](figures/R60/variant2_utility.png)

**Intuition:** A myopic agent ignores the future maintenance savings from replacement, making replacement less attractive than the researcher predicts. The estimator compensates by raising the replacement cost and lowering the maintenance cost coefficient. At intermediate horizons, biases need not move monotonically: the likelihood depends on how choices vary with mileage and time, not merely on the overall frequency of replacement.

| Future periods considered | Replacement cost bias | Maintenance coefficient bias |
| ---: | ---: | ---: |
| 0 | +6.55% | −64.37% |
| 2 | +30.45% | +39.59% |
| 4 | −4.65% | −4.10% |
| **6** | **+0.53%** | **+0.10%** |
| 8 | −0.06% | +0.05% |
| 60 (full foresight) | +0.08% | +0.06% |

**Key result:** Six future periods are sufficient for both relative biases to fall below 1% in the baseline calibration. A sufficiently far-sighted agent already accounts for the future consequences that matter most for the current decision. The small replacement cost bias at six periods remains statistically distinguishable from zero using the plotted interval; the result demonstrates practical robustness, not exact unbiasedness.

[Complete horizon results](results/variant2/summary_R60.csv) · [Replacement frequencies](results/variant2/replacement_counts_R60.csv)

<br>

### 2. Discount Factor Misspecification

**Researcher's assumption:** The agent discounts future utility using a factor of 0.8.

**Agent's actual behaviour:** The discount factor varies from 0 to 1. Both the agent and researcher use full foresight and correct mileage transition probabilities but attach different weights to future utility.

![Relative cost parameter biases under discount factor misspecification](figures/R60/variant1_utility.png)

**Intuition:** When the agent discounts the future more heavily than the researcher assumes, future maintenance savings are less valuable and replacement is less attractive. The estimator raises the replacement cost and lowers the maintenance cost coefficient to explain the observed decisions. The reverse mechanism applies when the agent places more weight on the future than the researcher assumes.

**Key result:** At an actual discount factor of 0.6, the replacement cost is overestimated by **15.25%** and the maintenance cost coefficient is underestimated by **11.31%**. Differences in time preferences are mistaken for differences in underlying costs.

[Underlying estimates](results/variant1/summary_R60.csv)

<br>

### 3. Subjective Transition Beliefs

**Researcher's assumption:** The agent evaluates future mileage growth using the objective transition probabilities estimated from realised mileage changes.

**Agent's actual behaviour:** The agent instead uses subjective beliefs that become increasingly pessimistic (favouring 10,000 km of additional mileage) or optimistic (favouring zero additional mileage). The deviation α ranges from 0 (correct beliefs) to 1 (extreme beliefs), while actual mileage growth continues to follow the objective distribution.

![Relative cost parameter biases under subjective transition beliefs](figures/R60/variant3_utility.png)

**Intuition:**

- **Pessimistic beliefs:** The agent expects even a new engine to accumulate mileage quickly, reducing the value of replacement. The estimator generally compensates by raising the replacement cost and lowering the maintenance cost coefficient.
- **Optimistic beliefs:** The agent expects a new engine to remain at low mileage for longer, increasing the value of replacement. The estimator lowers the replacement cost and tends to raise the maintenance cost coefficient.

**Key result:** Under extreme optimistic beliefs, the replacement cost estimate is approximately **45% too low**. Even accurately estimated objective transition probabilities do not reveal the subjective expectations that determine the agent's choices.

[Optimistic beliefs](results/variant3/100/summary_R60.csv) · [Pessimistic beliefs](results/variant3/001/summary_R60.csv)

<br>

### 4. Robustness Across Calibrations

**Researcher's assumption:** Full foresight, with the other behavioural assumptions correctly specified.

**Agent's actual behaviour:** The planning horizon varies again under four alternative calibrations, changing one component at a time: the maintenance coefficient (0.001), replacement cost (50), objective transition probabilities (0.1, 0.3, 0.6) or discount factor (0.99 for both agent and researcher).

![Limited-foresight robustness experiments under alternative calibrations](figures/R60/robustness_utility.png)

**Intuition:** Changing costs, mileage growth or discounting changes the value of future maintenance savings and hence how much foresight matters for current decisions. The experiments examine how parameter recovery responds to these changes, rather than treating six periods as a universal threshold.

**Scope:** The higher-maintenance-cost experiment has a documented discrepancy with the replacement frequencies reported in the thesis; see the [comparison output](results/robustness/maintenance_cost/table_A2_comparison_R60.csv).

<br>

### Why Transition Estimates Remain Robust

First-stage estimation uses eligible realised mileage increments without engine replacement. Across the main experiments, these increments are generated from the same objective distribution, independently of the agent's behavioural assumptions. The second-stage estimator, however, uses these probabilities together with the researcher's assumptions to explain observed choices.

**Central takeaway:** Structural parameter recovery depends on whether the researcher's model generates choice probabilities compatible with actual behaviour at the true parameters. An assumption can be formally incorrect yet have little practical effect on estimation. Conversely, accurately estimating one component of a model does not guarantee accurate estimation of the remaining components.

<br>

## Computational Methods

- **Dynamic programming:** Solve the agent's finite-horizon decision problem by backward induction, allowing distinct behavioural assumptions in simulation and estimation.
- **Efficient likelihood evaluation:** Aggregate observations by action, state and period to avoid repeatedly processing the full simulated panel during optimisation.
- **Numerical optimisation:** Rescale the maintenance coefficient to improve conditioning; use bounded L-BFGS-B optimisation with a Nelder–Mead fallback and convergence checks.
- **Controlled Monte Carlo experiments:** Use deterministic seeds and retain repetition-level estimates, likelihood values and optimiser diagnostics.
- **Statistical evaluation:** Quantify relative bias and Monte Carlo uncertainty, and check scenario grids and numerical outputs before generating figures.

For formal model specification, parameter calibration, likelihood derivations and numerical algorithms, see the [Technical Methodology](METHODOLOGY.md).

<br>

## Quick Start

Clone the repository and install the dependencies in a Python 3 environment:

```bash
git clone https://github.com/EliasBernhardRoth/Dynamic-Choice-Misspecification.git
cd Dynamic-Choice-Misspecification
python3 -m pip install numpy scipy matplotlib
```

**Run a single baseline estimation:**

```bash
python3 utility_estimation.py
```

Simulates a dataset and reports parameter estimates, likelihood values and optimiser diagnostics.

**Regenerate figures and appendix tables from saved simulation outputs:**

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

The default configuration uses 60 repetitions per specification. Full experiments rewrite the corresponding output files. Experiment settings are defined in [config.py](config.py).

<br>

## Repository Structure

| Component | Files |
| :--- | :--- |
| Technical documentation | [METHODOLOGY.md](METHODOLOGY.md) |
| Model specification and configuration | [model.py](model.py) · [config.py](config.py) |
| Dynamic programming | [solver.py](solver.py) · [limited_foresight.py](limited_foresight.py) |
| State transitions and simulation | [transitions.py](transitions.py) · [simulation.py](simulation.py) |
| Two-stage maximum likelihood | [estimation.py](estimation.py) · [utility_estimation.py](utility_estimation.py) |
| Monte Carlo experiments | [Variant 1](monte_carlo_variant1.py) · [Variant 2](monte_carlo_variant2.py) · [Variant 3](monte_carlo_variant3.py) |
| Robustness experiments | [monte_carlo_robustness.py](monte_carlo_robustness.py) |
| Figures and appendix tables | [plot_results.py](plot_results.py) · [plot_appendix.py](plot_appendix.py) |
| Research outputs | [figures/](figures/) · [results/](results/) · [tables/](tables/) |

**Technology stack:** Python · NumPy · SciPy · Matplotlib · Git

<br>

## Reference

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. *Econometrica*, 55(5), 999–1033.

---

**Elias Roth**  
MPhil Economics | University of Oxford  
[GitHub Profile](https://github.com/EliasBernhardRoth)
