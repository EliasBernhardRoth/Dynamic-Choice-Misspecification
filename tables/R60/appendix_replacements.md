# Appendix replacement tables

Monte Carlo repetitions per setting: **60**.

Entries are mean replacements per bus over the simulated period.
Markdown tables show two decimal places; CSV files preserve full precision.
These are Python results; exact agreement with all thesis results
has not yet been established.

## Variant 1: agent discount factor

| Agent beta | Mean replacements |
| --- | --- |
| 0 | 5.75 |
| 0.2 | 6.80 |
| 0.4 | 8.19 |
| 0.6 | 9.83 |
| 0.8 | 11.50 |
| 1 | 13.35 |

[Full-precision CSV](variant1_replacements.csv)

## Variant 2: limited foresight

| h | Mean replacements |
| --- | --- |
| 0 | 5.75 |
| 2 | 11.77 |
| 4 | 11.56 |
| 6 | 11.47 |
| 8 | 11.50 |
| 25 | 11.50 |
| 40 | 11.50 |
| 60 | 11.50 |

[Full-precision CSV](variant2_replacements.csv)

## Variant 3: subjective transition beliefs

| alpha | Pessimistic (0, 0, 1) | Optimistic (1, 0, 0) |
| --- | --- | --- |
| 0 | 11.50 | 11.50 |
| 0.2 | 10.59 | 12.13 |
| 0.4 | 9.82 | 13.10 |
| 0.6 | 9.18 | 14.46 |
| 0.8 | 8.68 | 15.80 |
| 1 | 8.26 | 17.69 |

[Full-precision CSV](variant3_replacements.csv)

## Robustness checks: counterpart to Table A2

| h | theta_c = 0.001 | RC = 50 | p = (0.1, 0.3, 0.6) | beta = 0.99 |
| --- | --- | --- | --- | --- |
| 0 | 10.42 | 1.39 | 13.23 | 5.75 |
| 2 | 17.36 | 3.93 | 20.60 | 14.40 |
| 4 | 16.98 | 5.12 | 19.81 | 13.12 |
| 6 | 16.98 | 5.21 | 19.63 | 13.23 |
| 8 | 16.98 | 5.06 | 19.69 | 13.24 |
| 25 | 16.99 | 5.00 | 19.69 | 13.25 |
| 40 | 16.99 | 5.00 | 19.69 | 13.25 |
| 60 | 16.99 | 5.00 | 19.68 | 13.24 |

[Full-precision CSV](robustness_replacements.csv)

## Monte Carlo uncertainty

[All replacement means and standard errors](all_replacement_counts.csv)

## Comparison with thesis Table A2

The comparison uses the thesis's rounded replacement counts.
An entry matches the printed rounding interval when its absolute
difference is at most 0.005 replacements per bus.

| Scenario | Entries within rounding interval | Largest absolute difference |
| --- | --- | --- |
| theta_c = 0.001 | 0/8 | 2.0471 |
| RC = 50 | 2/8 | 0.0090 |
| p = (0.1, 0.3, 0.6) | 6/8 | 0.0053 |
| beta = 0.99 | 7/8 | 0.0055 |

[Detailed comparison](table_A2_comparison.csv)

Differences remain under review. Matching replacement counts
does not by itself establish agreement of the utility estimates.
