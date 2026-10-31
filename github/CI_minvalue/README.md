
# Confidence Intervals for Minimum Estimation Using Extreme Value Theory

**Research Project | Python | Extreme Value Theory | Monte Carlo Simulation | Statistical Inference**

This project investigates the use of Extreme Value Theory (EVT) to estimate the minimum of a large collection of values when only a subset is observed. The research focuses on comparing tail-parameter estimation methods, selecting an appropriate tail threshold, and constructing approximate confidence intervals for the minimum of the full sample.

The work is motivated by **Integrated Information Theory (IIT)**, where calculating integrated information can require evaluating a large number of possible system partitions. As the number of partitions increases, finding the minimum information partition becomes computationally expensive.

The goal is to investigate whether the minimum can be estimated from a smaller sample rather than evaluating every possible value.

## Research Paper

**Evaluating Tail Parameter Estimation Methods for Extreme Value Distributions: Toward Confidence Intervals for Integrated Information**

**Author:** Alaa Masaoud  
**Institution:** Brock University

[Read the Full Research Paper](docs/research_paper.pdf)

The accompanying research paper contains the mathematical methodology, experimental setup, simulation results, figures, discussion, and references.

## Research Objectives

The main objectives of this project are to:

1. Compare different estimators of the lower-tail parameter alpha.
2. Investigate how the number of tail observations, k, affects estimator performance.
3. Construct approximate confidence intervals for the minimum of a full sample using only a subsample.
4. Evaluate empirical coverage probability, confidence interval length, bias, and variance.
5. Compare estimator performance across different distributions and subsampling proportions.

## Methodology

### 1. Data Generation

The simulation generates a complete sample of size N from a specified probability distribution.

A random subsample of size n is selected without replacement, and the true minimum of the complete sample is retained for evaluation.

The research paper considers four distributions in the Weibull domain of attraction:

- Uniform(0,1)
- Beta(1,2)
- Beta(2,1)
- Perturbed Power distribution with alpha = 1

The experiments investigate subsampling proportions of 1%, 5%, and 10%.

The distributions and simulation settings are configurable, allowing different experimental conditions to be evaluated.

### 2. Weibull-to-Fréchet Transformation

Many EVT tail estimators are designed for heavy upper tails, while minimum estimation concerns the lower tail of a distribution.

To apply these estimators, the lower-tail observations are transformed into an upper-tail problem.

For a candidate lower endpoint w, the transformation is:

\[
Y_i = \frac{1}{X_i-w}
\]

As an observation approaches the candidate lower endpoint from above, its transformed value increases.

This allows upper-tail estimation methods to be applied to the original minimum-estimation problem.

### 3. Tail-Parameter Estimation

The framework implements an estimator registry and adapter system for comparing EVT estimators.

The research paper evaluates:

- Hill
- Moment
- L-Moment
- GPD Maximum Likelihood Estimation (GPD MLE)
- Grimshaw Maximum Likelihood Estimation
- Maximum Product of Spacings for GPD (MPS GPD)
- Zhang Likelihood Moment Estimation (Zhang LME)

The broader codebase also supports additional estimators, including Reduced-Bias Hill, Mixed Kernel, Pickands, Probability Weighted Moments (PWM), QQ, and UH.

A true-alpha benchmark is included to evaluate the performance of the confidence interval procedure when the tail parameter is known.

### 4. Endpoint Estimation

The implementation generates candidate lower endpoints using the observed subsample.

For each candidate endpoint:

1. The observations are transformed into an upper-tail representation.
2. The selected tail estimator produces an estimate of alpha.
3. A Kolmogorov-Smirnov statistic measures agreement with a fitted Pareto tail model.
4. The candidate endpoint and estimated alpha with the smallest KS statistic are selected.

This procedure does not require the theoretical distribution endpoint to be supplied.

The endpoint used during the tail transformation is distinct from the endpoint subsequently estimated during confidence interval construction.

### 5. Selection of k

The parameter k determines how many extreme observations are used when estimating the tail.

Selecting k involves a bias-variance tradeoff. Smaller values focus on more extreme observations but may produce unstable estimates, while larger values use more observations but may introduce bias.

The implementation uses a two-stage procedure:

**Pilot stage:** Candidate k values are evaluated for each estimator using pilot simulation repetitions.

**Evaluation stage:** The selected k for each estimator is held fixed during a separate set of simulation repetitions.

The selection procedure considers:

1. Number of valid confidence intervals
2. Weighted deviation from target coverage
3. Average confidence interval length
4. Absolute alpha bias
5. Alpha variance
6. Smaller k as a final tie-breaker

The selection is performed separately for each estimator rather than assigning the same k to every method.

### 6. Confidence Interval Construction

The lower-tail quantile function is approximated by:

\[
Q(p)=\ell+ap^{1/\alpha}
\]

where:

- \(\ell\) is the lower endpoint.
- \(a\) is a positive scale parameter.
- \(\alpha\) is the lower-tail parameter.

The first k order statistics are used to estimate the endpoint and scale parameters.

The implementation uses generalized least squares (GLS) and an approximate covariance structure for the order statistics.

The fitted lower-tail model is then used to construct an approximate interval for the minimum of the full sample, accounting for the unobserved observations.

The observed subsample minimum provides a natural upper bound because the minimum of the full sample cannot exceed the smallest observed value.

The intervals are model-based approximations, and their empirical coverage is evaluated through simulation.

## Simulation Design

The simulation framework supports configurable experimental settings.

An example configuration is:

| Parameter | Value |
|---|---|
| Full sample size | 10,000 |
| Subsample size | 1,000 |
| Pilot repetitions | 15 |
| Evaluation repetitions | 500 |
| Candidate k values | 10, 20, ..., 990 |
| Confidence level | 95% |
| Sampling with replacement | No |
| Random seed | 123 |

These values represent one configuration of the simulation framework. Different settings, including larger pilot studies and different subsample proportions, can be used for additional experiments.

The simulation uses independent random number streams and multiprocessing to support reproducible Monte Carlo experiments.

## Performance Metrics

Estimator performance is evaluated using the following statistical measures.

### Empirical Coverage Probability

The proportion of valid confidence intervals containing the true minimum of the complete sample.

The target coverage probability is 95%.

### Average Confidence Interval Length

The average difference between the upper and lower confidence interval bounds.

Shorter intervals indicate greater precision when coverage is comparable.

### Bias

The difference between the average estimated tail parameter and the true tail parameter:

\[
\operatorname{Bias}(\hat{\alpha}) =
E[\hat{\alpha}]-\alpha
\]

### Variance

The variability of the estimated tail parameter across simulation repetitions.

### Quality Score

A quality score combines average confidence interval length with deviation from the target coverage probability:

\[
Q =
\frac{1}{
\overline{L}(1+50|C-0.95|)
}
\]

where:

- \(\overline{L}\) is the average confidence interval length.
- \(C\) is the empirical coverage probability.

Higher scores indicate shorter intervals with coverage closer to the target under this scoring rule.

The quality score is used alongside the individual performance metrics rather than replacing them.

## Research Results

The accompanying research paper presents results across four distributions and three subsampling proportions.

### Effect of Subsample Proportion

Increasing the subsample proportion generally improved empirical coverage and reduced average confidence interval length.

The largest improvements in interval length were typically observed when increasing the subsample proportion from 1% to 5%.

These findings suggest that observing a larger portion of the full sample improves the precision of minimum estimation.

### Effect of k

The selected value of k varied across estimators, distributions, and subsampling proportions.

No single value of k consistently produced the best performance for every estimator.

This supports selecting k separately for each estimation method.

### Overall Estimator Performance

The true-alpha benchmark achieved the highest overall quality score.

Among the methods that estimated alpha:

- **Grimshaw MLE** achieved the highest average quality score.
- **GPD MLE** also demonstrated strong overall performance.
- **Zhang LME** performed competitively across the simulated conditions.

These results identify promising tail-parameter estimation methods for further investigation of minimum estimation from incomplete observations.

The conclusions are based on the simulated conditions described in the research paper and do not establish universal superiority across all distributions or applications.

## Technologies

- Python
- NumPy
- pandas
- SciPy
- openpyxl
- Python multiprocessing
- Monte Carlo simulation
- Extreme Value Theory
- Statistical inference

## Project Structure

```text
CI_min/
├── main.py
├── config.py
├── distributions.py
├── simulation.py
├── confidence_intervals.py
├── k_selection.py
├── summary.py
├── summary_statistics.py
├── output_writer.py
├── overall_estimator.py
├── overall_estimator_k100.py
├── compare_p10_vs_p01.py
├── evt/
│   ├── estimator_registry.py
│   ├── estimator_adapter.py
│   └── transformations.py
├── outputs/
└── docs/
    └── research_paper.pdf
```

The `evt/` directory contains the estimator integration and tail-transformation components.

The `outputs/` directory contains generated simulation results and additional analysis outputs.

## Installation

Clone the repository:

```bash
git clone https://github.com/alaa-masaoud/CI_min.git
cd CI_min
```

Install the main Python dependencies:

```bash
python -m pip install numpy pandas scipy openpyxl
```

Additional dependencies may be required by individual estimator implementations.

## Running the Simulation

From the directory containing `main.py`, run:

```bash
python main.py
```

The simulation settings can be modified in `config.py`.

To limit the number of worker processes, set the `EVT_MAX_WORKERS` environment variable.

For example, using PowerShell:

```powershell
$env:EVT_MAX_WORKERS = "4"
python main.py
```

Full simulations may require substantial computation time depending on the number of repetitions, candidate k values, estimators, and endpoint candidates.

## Simulation Outputs

The main simulation writes results to `outputs/report/`.

Generated files include:

- `evt_simulation_results.xlsx` — formatted simulation results and summaries
- `optimal_k_diagnostics.csv` — diagnostics for candidate k values
- `simulation_configuration.json` — configuration used for the experiment
- `raw_csv/` — distribution-specific simulation results

Additional analysis scripts calculate estimator summaries, quality scores, and comparisons across experimental settings.

## Limitations and Future Work

The current study uses controlled probability distributions with known lower-tail behavior.

Although the results demonstrate the potential of EVT-based minimum estimation, the same performance is not guaranteed for actual IIT partition values.

Additional considerations include:

- Sensitivity to the choice of k
- Accuracy of the lower-tail quantile approximation
- Uncertainty in estimated tail parameters
- Numerical stability and estimator failures
- Performance under different distributions and sample sizes
- Applicability to actual integrated information calculations

Future work includes evaluating the method on IIT partition values from small systems where the exact minimum can be computed, allowing direct comparison between the estimated interval and the true minimum.

## Author

**Alaa Masaoud**

Honours Computer Science — Artificial Intelligence Specialization  
Brock University

GitHub: [alaa-masaoud](https://github.com/alaa-masaoud)

