# Dynamic Discrete Choice Models under Misspecification

**Stochastic Optimisation · Statistical Modelling · Structural Estimation · Monte Carlo Simulation**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Research](https://img.shields.io/badge/Research-Structural_Econometrics-555555)

## Overview

When does a misspecified model still produce reliable parameter estimates?

This project develops a Python framework to investigate how incorrect assumptions about dynamic decision-making under uncertainty affect statistical inference. Building on Rust's (1987) dynamic discrete choice model, it combines dynamic programming, stochastic simulation and two-stage maximum likelihood estimation in controlled Monte Carlo experiments.

**Key finding:** In the baseline calibration, a researcher who assumes fully forward-looking behaviour recovers both cost parameters with less than 1% relative bias even when agents consider only six future periods.

This repository is my bachelor’s thesis, conducted at the Düsseldorf Institute for Competition Economics (DICE), Heinrich Heine University Düsseldorf. The thesis received the **IHK Düsseldorf Award for Best Thesis**.

[Technical methodology](METHODOLOGY.md) · [Simulation outputs](results/) · [Figures](figures/) · [Statistical tables](tables/)

<br>

## Project Highlights

| Experimental scale | Value |
| :--- | ---: |
| Simulated decision observations per specification | **3,600,000** |
| Monte Carlo repetitions per specification | 60 |
| Simulated buses per repetition | 1,000 |
| Decision periods per bus | 60 |
| Discrete states | 71 |
| Dimensions of misspecification | 3 |
| Additional robustness calibrations | 4 |

**Computational methods:** finite-horizon dynamic programming, backward induction, Markov transition matrices, likelihood maximisation, constrained numerical optimisation and Monte Carlo error quantification.

<br>

## The Decision Problem

A fleet manager decides each month whether to maintain or replace a bus engine over a 60-month operating horizon. The decision involves a trade-off:

- **Maintain:** Avoid the upfront replacement cost but incur maintenance costs that increase with accumulated mileage.
- **Replace:** Pay the replacement cost, reset accumulated mileage and avoid maintenance costs in the current month, with potential maintenance savings in subsequent months.

Future mileage growth is stochastic. Each month, a bus travels either 0, 5,000 or 10,000 km. The attractiveness of replacement therefore depends on how the manager evaluates future costs and mileage growth.

The researcher observes replacement decisions and realised mileage, but not the underlying structural parameters. These are recovered in two stages:

1. Estimate the objective probabilities of the three mileage increments from observed state transitions.
2. Estimate the replacement cost (**RC**) and the maintenance cost coefficient (**$θ_c$**) from observed decisions using maximum likelihood.

The central difficulty is that the second stage requires assumptions about how the agent evaluates future consequences. If these assumptions generate incorrect choice probabilities at the true cost parameters, the estimator may compensate by distorting the costs to better explain the observed decisions.

<br>

## Selected Results

Across the three main experiments, the researcher assumes a discount factor of 0.8, full foresight over the remaining operating horizon, and correct expectations about future mileage growth. Each experiment varies exactly one aspect of the agent's behaviour while keeping the researcher's assumptions unchanged.

The figures show the relative bias of the estimated replacement cost and maintenance cost coefficient, calculated from 60 Monte Carlo repetitions per specification. Positive values indicate overestimation; negative values indicate underestimation. The error bars represent ±1.96 Monte Carlo standard errors of the estimated mean bias.

### 1. Limited Foresight

**Researcher's assumption:** The agent considers all remaining months until the end of the 60-month operating horizon.

**Agent's actual behaviour:** At each decision, the agent considers at most the next h months. With h = 0, only immediate costs matter; with h = 60, the agent has full foresight. The discount factor and expectations about mileage growth are correctly specified.

![Relative cost parameter biases across planning horizons](figures/R60/variant2_utility.png)

**Intuition:** An agent who ignores the future entirely does not consider the future maintenance savings of replacing an engine. Such an agent replaces much less frequently than the researcher expects at the true cost parameters. The estimator compensates by increasing the replacement cost and decreasing the maintenance cost coefficient, making replacement less attractive in the model.

For other short planning horizons, the pattern is not monotonic: at h = 2, for example, the agent replaces slightly more frequently than under full foresight, yet both cost parameters are overestimated. This illustrates why the average number of replacements alone cannot explain the two estimates: the likelihood also depends on how replacement probabilities vary with mileage and time.

| Future periods considered | Replacement cost bias | Maintenance coefficient bias | Average replacements per bus (original thesis) |
| ---: | ---: | ---: | ---: |
| 0 | +6.55% | -64.37% | 5.75 |
| 2 | +30.45% | +39.59% | 11.77 |
| 4 | -4.65% | -4.10% | 11.56 |
| **6** | **+0.53%** | **+0.10%** | **11.47** |
| 8 | -0.06% | +0.05% | 11.50 |
| 60 (full foresight) | +0.08% | +0.06% | 11.50 |

**Key result:** Once the planning horizon is sufficiently long, extending it has little additional effect on replacement behaviour. At six future periods, replacement frequencies in the original thesis are already close to those under full foresight, and both cost biases in the Python implementation are below 1%.

The intuition is that a sufficiently far-sighted agent already accounts for the future consequences that matter most for the current decision. The researcher may therefore assume full foresight without materially distorting the cost estimates, even though that assumption is formally incorrect.

The original thesis reports statistically insignificant biases from six periods onwards. In the Python implementation, replacement cost bias at six periods remains statistically distinguishable from zero using the plotted interval, despite its small magnitude. The result therefore demonstrates practical robustness in this calibration, not exact unbiasedness or a universal six-period threshold.

[Complete horizon results](results/variant2/summary_R60.csv) · [Replacement frequencies](results/variant2/replacement_counts_R60.csv)

<br>

### 2. Discount Factor Misspecification

**Researcher's assumption:** The agent discounts future utility using a factor of 0.8.

**Agent's actual behaviour:** The agent's discount factor varies from 0 to 1. Both the agent and the researcher consider the full remaining horizon and use the correct mileage transition probabilities, but they attach different weights to future utility.

![Relative cost parameter biases under discount factor misspecification](figures/R60/variant1_utility.png)

**Intuition:** When the agent's discount factor is below 0.8, future maintenance savings matter less to the agent than the researcher assumes. The researcher consequently overvalues the benefits of replacement at the true cost parameters and predicts too many replacements. In the original thesis, an agent with a discount factor of 0.6 replaces an engine 9.82 times per bus on average, compared with 11.50 under the assumed discount factor of 0.8.

To reconcile the model with these decisions, the estimator increases the estimated replacement cost and decreases the estimated maintenance cost coefficient. Replacement becomes less attractive, while maintenance becomes more attractive. When the agent's discount factor exceeds 0.8, the reverse mechanism applies: the agent replaces more often than expected, and the estimator lowers replacement costs and raises the maintenance cost coefficient.

**Key result:** With an actual discount factor of 0.6, the Python estimates overstate the replacement cost by **15.25%** and understate the maintenance cost coefficient by **11.31%**. The estimator confuses differences in time preferences with differences in underlying costs.

[Underlying estimates](results/variant1/summary_R60.csv)

<br>

### 3. Subjective Transition Beliefs

**Researcher's assumption:** The agent correctly anticipates future mileage growth. The researcher therefore uses the objective transition probabilities estimated from realised mileage changes to model the agent's decisions.

**Agent's actual behaviour:** The agent uses subjective transition probabilities instead. Two scenarios are considered: increasingly pessimistic beliefs favouring 10,000 km of additional mileage each month, and increasingly optimistic beliefs favouring zero additional mileage. The degree of deviation α ranges from 0 (correct beliefs) to 1 (the extreme belief). Actual mileage growth continues to follow the same objective distribution in both scenarios.

![Relative cost parameter biases under subjective transition beliefs](figures/R60/variant3_utility.png)

**Intuition:**

- **Pessimistic beliefs:** The agent expects even a newly installed engine to accumulate mileage quickly and incur high maintenance costs again soon. In this model, paying the upfront replacement cost is therefore less attractive. The agent replaces less frequently than the researcher expects. The estimator generally compensates by raising the replacement cost and lowering the maintenance cost coefficient.
- **Optimistic beliefs:** The agent expects a new engine to remain at low mileage for longer, so replacing the existing engine promises more persistent maintenance savings. The agent replaces more frequently than the researcher expects. The estimator lowers the replacement cost and tends to raise the maintenance cost coefficient to make frequent replacement more plausible.

The original thesis documents these contrasting patterns: under the extreme pessimistic belief, the average number of replacements falls to 8.26 per bus; under the extreme optimistic belief, it rises to 17.69, relative to 11.50 under correct expectations.

**Key result:** In the extreme optimistic scenario, the Python estimate of the replacement cost is approximately **45% too low**. The key problem is not inaccurate estimation of the objective mileage process: the first-stage estimates capture actual mileage growth, but not the subjective probabilities that determine the agent's choices.

The second stage consequently uses the wrong probabilities to model expected future utility, even when the objective probabilities have been accurately estimated.

[Optimistic beliefs](results/variant3/100/summary_R60.csv) · [Pessimistic beliefs](results/variant3/001/summary_R60.csv)

<br>

### 4. Robustness Across Calibrations

**Researcher's assumption:** Full foresight, as in the baseline planning-horizon experiment.

**Agent's actual behaviour:** The agent's planning horizon varies again, but the experiment is repeated under four alternative calibrations. In each case, the researcher continues to assume full foresight; the other behavioural assumptions are correctly specified.

![Limited-foresight robustness experiments under alternative calibrations](figures/R60/robustness_utility.png)

The original thesis varies four components separately: the maintenance cost coefficient (0.001), the replacement cost (50), the objective mileage transition probabilities (0.1, 0.3, 0.6), and the discount factor (0.99 for both the agent and the researcher).

**Intuition:** Changing the calibration affects the value of replacement relative to maintenance and can therefore change how far ahead an agent must look before additional foresight has little effect on decisions. The robustness experiments investigate whether the relationship between stabilising replacement behaviour and declining estimation bias persists under these alternative settings, rather than whether exactly six periods always suffice.

The higher-maintenance-cost experiment in the Python implementation does not reproduce the original thesis's replacement frequencies exactly; see [Research Provenance and Limitations](#research-provenance-and-limitations).

<br>

### Why Transition Estimates Remain Robust

In the first estimation stage, the researcher estimates the objective mileage transition probabilities from the relative frequencies of observed mileage increments in eligible periods without engine replacement. The actual mileage increments are generated from the same objective distribution across the main experiments, independently of the agent's discount factor, planning horizon or subjective beliefs.

The second stage is different: it uses these transition probabilities together with assumptions about dynamic behaviour to estimate the cost parameters from observed choices. In particular, when subjective beliefs differ from objective probabilities, correctly recovering the latter does not recover the agent's actual expectations.

**Central takeaway:** What matters for structural parameter recovery is whether the researcher's model generates choice probabilities compatible with actual behaviour at the true parameters. An assumption can be formally incorrect yet have little practical effect on estimation, as the limited-foresight experiment illustrates. Conversely, accurately estimating one component of a model does not guarantee accurate estimation of the remaining components.

<br>

## Model and Experimental Design

Each specification uses the same underlying stochastic decision model and known structural parameters, except for the explicitly varied calibration in the robustness experiments. The simulation and estimation stages use separate behavioural assumptions, allowing the effect of misspecification to be isolated.

1. Solve the agent's finite-horizon dynamic problem by backward induction under the specified behavioural assumptions.
2. Simulate a panel of choices and objective state transitions.
3. Estimate objective transition probabilities from eligible realised mileage increments.
4. Estimate cost parameters by maximising the conditional-choice likelihood under the researcher's maintained behavioural assumptions, solving the dynamic model for candidate cost parameters.
5. Compare the estimates against the known true values and repeat the experiment to quantify bias and sampling variability.

For the mathematical framework, parameter calibration and numerical algorithms, see the [Technical Methodology](METHODOLOGY.md).

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

The figures and percentage biases reported above are from the Python implementation; the replacement frequencies explicitly attributed to the original thesis are from its Table 1 and Appendix A.2. The original simulations were conducted in MATLAB, with figures prepared in R. The implementation uses the documented model calibrations; exact numerical agreement with the original experiments is not assumed. In particular, the higher-maintenance-cost robustness experiment does not reproduce the replacement frequencies reported in the original Table A2. The discrepancy is retained in the [comparison output](results/robustness/maintenance_cost/table_A2_comparison_R60.csv).

The interpretation follows Sections 3–5 and Appendix A of the original thesis. For mathematical derivations, parameter values and estimation procedures, see the [Technical Methodology](METHODOLOGY.md).

<br>

## Reference

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. *Econometrica*, 55(5), 999–1033.

---

**Elias Roth**  
MPhil Economics | University of Oxford  
[GitHub Profile](https://github.com/EliasBernhardRoth)
