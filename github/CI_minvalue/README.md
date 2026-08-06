# EVT Minimum-CI Simulation

## Exact experiment

For every Monte Carlo repetition:

1. Generate 10,000 observations from `Uniform(0,1)`.
2. Save the observed minimum of those 10,000 values.
3. Draw one 1,000-value subsample from the same sample, without replacement.
4. For each `k` in `25, 50, 75, 100, 150, 200`, estimate alpha with every EVT estimator using the same subsample.
5. Use each alpha estimate and the lower order statistics of the subsample to fit the lower-tail model and construct a 95% interval for the minimum of the original 10,000 values.
6. Record whether the saved 10,000-point minimum lies in the interval, along with alpha and interval length.
7. Repeat 1,000 times and summarize performance separately for every estimator and every k.

The known lower endpoint of `Uniform(0,1)` is **not supplied** to alpha estimation or to the interval calculation.

## Interval model

For fixed alpha, the first k subsample order statistics fit

```text
Q(p) = m + a p^(1/alpha).
```

Because the 1,000 sampled observations already belong to the full sample, the interval predicts the minimum of the remaining 9,000 observations and caps its upper endpoint at the observed subsample minimum.

## Outputs

- one raw CSV for the uniform experiment;
- an Excel workbook containing settings, raw results, summary by estimator and k, and errors;
- CI lower and upper endpoints, coverage indicators, and interval lengths.

## Run

```bash
pip install -r requirements.txt
python main.py
```
