# Evaluation and reporting contract

The 22 membership files under splits/ are byte-preserved from the previous
release. evaluation_reporting.json is their companion for context joins,
training cohort sizes, held-out identities, and pair categories. Exported
observations carry Context_ID, Plan_Row_ID, stage, workload_id and the
zero-based stage_a_sample_index. Match by ID before computing errors.

## Utilization

For each of the five baseline models, evaluate only the 30 workloads selected
from its own test fold. Join U=.6/.7/.8 Stage A rows on workload_id and
stage_a_sample_index. Every matched tuple uses the same Pbar, R and full weight
vector. Arrivals remain specific to each utilization. Report paired Stage A
errors separately from Stage B results. Stage B targets each context's own
measured evidence and its rows do not have an across-utilization bid match.
Summarize the five baseline model results across folds and retain the per-fold
values. U=.7 and J=7–8 remain test-only.

## Leave one source out

For source X, report ΔX = Error(Leave X Out) − Error(Baseline) on the same
Plan_Row_IDs. Use baseline_test_contexts in the companion JSON to select the
per-fold intersections with the leave-one test cohort. Evaluate the associated
baseline model only on its holdout contexts. If reporting a cross-fold delta
for a metric, compute the two metrics over identical paired observation IDs
with the same denominator, then subtract. Retain per-fold results and sample
counts. Include only the common eligible observations for both models if an
evaluation is partial, and report any missing IDs.

Leave-one training contains 321 workload definitions and 642 contexts, with
128,400 planned simulator rows across both stages. The baseline models contain
498 or 501 workload definitions and 996 or 1,002 contexts, with 199,200 or
200,400 planned rows. These are exact manifest counts. For actual model
reporting, also record the number of observations used after any filtering.
The training cohort size differs, so ΔX cannot isolate the causal effect of
unseen job identity. Lower training data volume may contribute to any
performance degradation. The source exclusion includes all variants derived
from X. There is no matched-training-size control in this DOE.

Example schema:

| Held-out source | Baseline fold | Metric | Stage | Matched rows | Leave-out training rows used | Baseline training rows used | Leave-out error | Baseline error | ΔX |
|---|---|---|---|---:|---:|---:|---:|---:|---:|

## Leave two sources out

Keep the two four-pair sets and their exact source exclusions. The companion
JSON identifies each model's set and category. Report each metric separately
for neither (familiar-source test control), exactly one, and both excluded
sources. Compute the metric within each model and group first. For each pair
category, report the Set 1 and Set 2 pair-level results, their unweighted mean
and optionally their range. Preserve both denominators and both pair names.
If either result is unavailable, state that and leave the category mean
unavailable. Never treat a pair with more rows as a larger repetition.
The two repetitions are a modest sensitivity check for source-pair choice and
do not support a precise population-level uncertainty estimate.

| Pair category | Presence group | Stage | Metric | Set 1 pair | Set 1 N | Set 1 value | Set 2 pair | Set 2 N | Set 2 value | Unweighted mean | Optional range |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|---|

## Metrics

Report p90 MAE, per-job QoS MAE, normalized violation distance MAE and
false-feasible rate where applicable. Feasible means p90 ≤ .30 and every
Pj ≤ .10. False-feasible rate uses actual infeasible observations as its
denominator. Mark a zero denominator unavailable.

The evaluation distance is d = hypot(max(p90/.30 − 1, 0),
max(max(Pj)/.10 − 1, 0)). The sampling margin is
m = max(p90/.30, max(Pj)/.10) − 1. The SCC pilot boundary criterion uses
measured abs(m) ≤ .25. Report sample counts, strata and exclusions. Stage B's
targeted acquisition changes the distribution of observed outcomes.
