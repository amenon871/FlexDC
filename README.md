# FlexDC-Sim

FlexDC-Sim is an open-source data-center demand response simulator for evaluating the power, performance, QoS, and cost implications of demand response (DR) participation.

This repository extends the original FlexDC-Sim codebase with research infrastructure for generating large simulator datasets used to train and evaluate learned FlexDC behavior models.

The current research baseline is **Model V3**, whose training data is generated using the **Sweep V3** methodology described below. Additional experiments, such as the J=2 model-family holdout study, are used to evaluate generalization and model architecture behavior but are not replacements for the V3 training framework.

The learned behavior-model code, training notebooks, inference utilities, and differentiable optimization code are maintained separately in:

- [CONDOR-FLEXDC](https://github.com/NetherMoon/CONDOR-FLEXDC)

The original FlexDC-Sim project is available at:

- [peaclab/flexdc-sim](https://github.com/peaclab/flexdc-sim)

---

## Research Overview

The broader research workflow uses FlexDC simulations to train a neural surrogate of data-center behavior.

Instead of repeatedly invoking the simulator during optimization, the learned model predicts the simulator quantities needed to reconstruct the FlexDC objective and evaluate feasibility.

The high-level workflow is:

```text
FlexDC workload/configuration definitions
                |
                v
Exact simulation sweep
                |
                v
FlexDC simulations
                |
                v
Audited training dataset
                |
                v
Behavior model training in CONDOR-FLEXDC
                |
                v
Differentiable prediction and optimization
```

FlexDC remains the **ground-truth simulator**. The learned model is trained to approximate its behavior.

---

# Active Model V3 Dataset

## Sweep V3

Sweep V3 is the active dataset-generation baseline for Model V3 and the starting point for future full-scale behavior-model datasets.

The design combines:

- dense structured coverage of the physical \(P/R\) operating space;
- scrambled Sobol samples for additional interior coverage;
- explicit physical-boundary samples;
- broad scheduler-weight coverage;
- deterministic simulation-seed repeats;
- grouped train/validation/test assignments.

### Default V3 coverage per context

A context is a fixed combination of:

- workload;
- server count;
- utilization.

For every context, Sweep V3 generates:

| Component | Count |
|---|---:|
| Structured P/R grid | 160 |
| Sobol interior P/R points | 96 |
| Physical-boundary P/R points | 32 |
| **Total P/R points** | **288** |
| Scheduler weight vectors per P/R point | **40** |
| **Base configurations per context** | **11,520** |

The structured portion uses:

```text
16 Pbar values
×
10 legal R values per Pbar
=
160 structured P/R points
```

The same 40-vector scheduler-weight bank is evaluated at every P/R location.

By default, 5% of base configurations are also repeated using additional simulation seeds to measure simulator variability without allowing repeated versions of the same base configuration to cross data splits.

---

## Why V3 Uses Multiple Sampling Strategies

A simple rectangular grid does not provide equally useful coverage everywhere in the physical operating region.

Sweep V3 therefore combines three P/R sampling strategies.

### 1. Structured grid

The structured grid provides predictable, interpretable coverage of the response surface.

It is useful for:

- learning smooth trends in Pbar and reserve;
- comparing nearby operating points;
- visualizing response surfaces;
- ensuring broad coverage across every context.

### 2. Sobol interior sampling

Scrambled Sobol points provide additional space-filling coverage away from the regular grid.

They reduce dependence on a single grid structure and help expose behavior between structured locations.

### 3. Physical-boundary sampling

Dedicated samples are placed near important physical limits, including:

- minimum reserve;
- the configured \(R/P\) limit;
- lower and upper Pbar limits;
- the \(Pbar + R\) limit.

Physical-boundary sampling is distinct from **behavioral boundary coverage**.

Behavioral feasibility boundaries depend on the simulator outputs and can only be measured after simulations have been completed.

---

# Scheduler-Weight Sampling

FlexDC workloads may contain multiple job types competing for scheduler allocation.

The scheduler weights:

```text
w_1, w_2, ..., w_J
```

must satisfy:

```text
sum(w_j) = 1
```

along with the configured lower and upper bounds for each job.

Sweep V3 constructs a deterministic weight bank containing:

- equal-weight allocation;
- individual low/high job anchors;
- ordered high/low allocations;
- lattice candidates;
- softmax-generated candidates;
- Dirichlet-generated candidates;
- Sobol/simplex candidates;
- maximin-selected space-filling vectors.

This allows the model to learn both ordinary scheduler allocations and more asymmetric allocations near the allowed weight limits.

---

# V3 Data Splits and Repeated Seeds

Sweep V3 assigns base configurations to:

```text
70% training
20% validation
10% test
```

The assignment is deterministic and stratified across the sampling design.

Repeated simulation seeds remain grouped with their original base configuration.

This is important because multiple simulator runs of the same:

```text
workload + context + Pbar + R + scheduler weights
```

should not appear on opposite sides of the training/test boundary.

The resulting training dataset also records a normalization weight for repeated base groups so repeated simulations do not unintentionally receive more training importance simply because they were run more times.

---

# Feasibility Labels

The current behavior-model feasibility definition uses both tracking and per-job QoS.

A simulator configuration is feasible when:

```text
p90 normalized tracking error <= 0.30
```

and

```text
max per-job QoS delay probability <= 0.10
```

Both conditions must hold.

The V3 audit also measures simulator coverage near the corresponding behavioral decision regions:

```text
Tracking diagnostic band:
0.20 <= p90 tracking error <= 0.40

QoS diagnostic band:
0.05 <= max(P_j) <= 0.15
```

These post-simulation diagnostics are used to determine whether the generated dataset contains sufficient examples near the boundaries that matter for optimization.

---

# FlexDC Objective

The learned behavior model is intended to predict underlying simulator behavior rather than directly memorizing a single scalar objective.

The FlexDC objective is reconstructed from simulator-derived components including:

```text
M_RSR
tracking penalty
QoS penalty
```

Conceptually:

```text
Full Objective
    =
M_RSR
    +
C_track
    +
C_QoS
```

The exact constants and cost-function configuration used for a dataset must remain consistent between:

- dataset generation;
- model training;
- inference;
- optimization;
- simulator validation.

Dataset-generation scripts and audits should verify this lineage rather than repairing inconsistent objective constants after generation.

---

# Current V3 Dataset Workflow

The following files implement the active Sweep V3 workflow.

## 1. Generate the V3 simulation plan

```text
am_generate_flexdc_sweep_plan_v3.py
```

This creates the immutable V3 plan before running any new simulations.

Default outputs include:

```text
flexdc_sweep_plan_v3.csv
flexdc_sweep_v3_plan_artifacts/
```

The plan specifies every simulation configuration in advance, including:

- workload;
- experiment configuration;
- server count;
- utilization;
- Pbar;
- reserve R;
- scheduler weights;
- simulation seed;
- sampling stratum;
- split assignment;
- stable Plan_Row_ID.

Generate the default V3 plan from the repository root:

```bash
python -u am_generate_flexdc_sweep_plan_v3.py
```

---

## 2. Audit the plan before simulation

Before spending time running the simulator, validate the plan itself:

```bash
python -u am_vet_flexdc_sweep_dataset_v3.py \
  --plan-file flexdc_sweep_plan_v3.csv \
  --plan-only
```

The preflight audit verifies the expected:

- contexts;
- P/R coverage;
- scheduler-weight coverage;
- physical limits;
- split counts;
- replicate design;
- plan identifiers.

A failed plan audit should be fixed before simulation begins.

---

## 3. Run the exact plan

The parallel runner is:

```text
am_run_flexdc_dataset_parallel_combo.py
```

It supports exact-plan execution using:

```text
--sweep-plan-file
```

and can:

- split contexts into worker chunks;
- run workers concurrently;
- write individual worker logs;
- validate worker outputs;
- verify Plan_Row_ID coverage;
- resume interrupted runs;
- perform dry runs before launching simulations.

A typical exact-plan invocation has the form:

```bash
python -u am_run_flexdc_dataset_parallel_combo.py \
  --sweep-plan-file flexdc_sweep_plan_v3.csv \
  --gradient-config <gradient-config> \
  --cluster-config configs/cluster/cluster.ini \
  --policy-name AQA \
  --output-prefix <run-prefix> \
  --node-count-control true \
  --pbar-lower-factor 0.9 \
  --pbar-upper-factor 1.0 \
  --pr-upper-factor 1.2 \
  --r-lower-kw-per-server 0.01 \
  --r-over-p-max 0.6 \
  --plan-num-chunks-per-context <chunks-per-context> \
  --plan-max-workers <max-workers> \
  --plan-run-id <run-id> \
  --plan-dry-run
```

After checking the dry run, replace:

```text
--plan-dry-run
```

with:

```text
--plan-resume
```

to execute or resume the plan.

Worker count and chunk count affect execution parallelism only. They should not change the scientific contents of the exact plan.

---

## 4. Audit and combine the completed simulations

After generation:

```bash
python -u am_vet_flexdc_sweep_dataset_v3.py \
  --plan-file flexdc_sweep_plan_v3.csv \
  --launch-manifest <completed-launch-manifest.csv>
```

The post-generation audit verifies:

- every planned row was produced;
- no unplanned rows were produced;
- Plan_Row_ID uniqueness;
- simulator inputs match the plan;
- weight vectors are valid;
- QoS vectors are valid;
- objective components reconstruct correctly;
- repeated seeds remain grouped;
- tracking/QoS feasibility coverage;
- near-boundary behavioral coverage;
- context-level dataset statistics.

It then produces the training-ready V3 dataset and supporting audit summaries.

---

# Future Full-Scale Model Training

Future behavior-model datasets should treat Sweep V3 as the baseline design rather than starting from a substantially sparser sampling strategy.

Changes to sampling density should be deliberate and validated against:

- feasible-row coverage;
- near-feasibility-boundary coverage;
- P/R surface resolution;
- scheduler-weight coverage;
- context coverage;
- simulator variability.

The goal is not merely to generate more rows. The goal is to ensure that the simulator response surface is sufficiently resolved for both accurate prediction and downstream optimization.

For parameterized or targeted future experiments, the repository also contains:

```text
am_generate_flexdc_exact_sweep_plan.py
```

This is a generic exact-plan generator whose:

- workload list;
- experiment list;
- utilization values;
- P/R density;
- scheduler-weight count;
- physical constraints;
- seed policy;
- cost constants

can be configured from the command line.

It is useful for new experiments where the V3 experiment definition itself should not be hard-coded.

For a full-scale successor to Model V3, V3-level sampling density should generally be the starting point unless a new experiment provides evidence that a different design is sufficient.

---

# J=2 Model-Family Holdout Experiment

The repository also contains infrastructure for a separate **J=2 pairwise experiment**.

This experiment is not the primary Model V3 training dataset and should not be confused with the main V3 workflow.

Its purpose is to evaluate questions such as:

- whether the behavior-model architecture generalizes to unseen model families;
- whether a model can combine one familiar and one unseen job type;
- whether it can predict workloads composed entirely of held-out model families;
- how generalization changes as model-family coverage in training is reduced.

The workload generator is:

```text
am_generate_flexdc_j2_pair_workloads.py
```

It constructs pairwise workloads from training and inference job types.

The generic exact-plan generator:

```text
am_generate_flexdc_exact_sweep_plan.py
```

is used for the controlled J=2 sweep.

The J=2 experiment is an **architecture/generalization evaluation experiment**. Its sampling choices should not automatically become the sampling strategy for future production behavior-model datasets.

Model-family holdout splitting and neural-model evaluation belong in the CONDOR-FLEXDC repository.

---

# Important Research Scripts

| Script | Purpose |
|---|---|
| `am_generate_flexdc_sweep_plan_v3.py` | Generate the active Sweep V3 training-data plan |
| `am_vet_flexdc_sweep_dataset_v3.py` | Preflight and post-generation V3 audit |
| `am_run_flexdc_dataset_parallel_combo.py` | Parallel execution of FlexDC dataset plans |
| `am_generate_flexdc_exact_sweep_plan.py` | Generic configurable exact-plan generator for new experiments |
| `am_generate_flexdc_j2_pair_workloads.py` | Generate workloads for the isolated J=2 family-holdout experiment |
| `am_generate_paper_iso_experiment_configs.py` | Generate server-count-specific experiment INIs from a canonical experiment config |
| `am_plot_cost_surfaces.py` | Plot aggregated FlexDC P/R response surfaces |
| `am_plot_cost_surfaces_with_feasible_points.py` | Plot P/R surfaces with constraint/feasibility points |

---

# Repository Responsibilities

This repository should contain the **simulator and simulator-data-generation side** of the research.

## FlexDC repository

This repository owns:

- FlexDC simulator source;
- simulator configuration files;
- workload definitions;
- experiment definitions;
- exact dataset-plan generation;
- parallel simulator execution;
- simulation-output auditing;
- P/R and cost-surface visualization.

## CONDOR-FLEXDC repository

CONDOR-FLEXDC owns:

- behavior-model architecture;
- model training;
- checkpoint selection;
- model-specific train/validation/test construction;
- inference;
- differentiable optimization;
- surrogate evaluation;
- surrogate-versus-simulator comparison.

Keeping this boundary clear prevents model-training assumptions from becoming dependencies of the simulator repository.

---

# Generated Data and Repository Hygiene

Large generated artifacts should not be committed to this repository.

Examples include:

```text
worker logs
worker batch files
plan chunks
launch manifests
generated sweep-plan CSVs
combined simulator outputs
training datasets
vetted datasets
plots generated for one experiment
Colab upload ZIPs
temporary staging directories
```

The local development repository may use:

```text
local_archive/
```

for research artifacts and legacy files that should remain available locally but should not be pushed to Git.

The important reproducibility artifacts to keep in Git are the **code and canonical configuration files required to regenerate those outputs**.

---

# Base FlexDC-Sim

## Simulator Inputs

FlexDC-Sim uses configuration files to define experiments.

Important categories include:

### Experiment configuration

Contains experiment-level settings such as:

- simulation duration;
- server count;
- random seed;
- ISO/power signal configuration.

### Policy configuration

Defines power-management and DR-policy behavior.

### Workload/job configuration

Defines application/job metadata including:

- minimum and maximum power;
- minimum and maximum runtime;
- QoS constraints;
- job size.

### Cluster configuration

Defines cluster-level simulation settings.

---

## Simulation Outputs

FlexDC records detailed simulator outputs including power and job behavior.

The simulator maintains:

- a node table describing node state, power, caps, and running jobs;
- a job table containing scheduling and execution information.

Simulation outputs are used to calculate quantities such as:

- tracking error;
- QoS delay probabilities;
- power/reserve cost;
- objective components.

---

# Development Setup

## 1. Install Python 3.10

Example on Ubuntu:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev -y
```

## 2. Clone this research repository

```bash
git clone https://github.com/amenon871/FlexDC.git
cd FlexDC
```

## 3. Create a virtual environment

```bash
python3.10 -m venv venv
```

Activate it:

### Linux/macOS

```bash
source venv/bin/activate
```

### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

## 5. Run tests

```bash
pytest
```

---

# Running a Basic FlexDC Simulation

The base simulator entry point is:

```text
src/peacsim/run_simulator.py
```

A simulation takes experiment, cluster, policy, and workload configuration files as input.

For example:

```bash
cd src/peacsim

python run_simulator.py \
  --experiment-config ../../configs/experiment/exp_low_util.ini \
  --cluster-config ../../configs/cluster/cluster.ini \
  --policy-config ../../configs/policy/example_flexDC_simulation.ini \
  --job-config ../../configs/workload/W2-short-qos3445.ini \
  --output-dir example_run
```

Simulation results are written beneath the FlexDC output directories.

For research dataset generation, use the plan-based workflow described earlier rather than manually launching large collections of independent simulations.

---

# Original FlexDC-Sim Citation

FlexDC-Sim is licensed under the BSD 3-Clause License.

If you use FlexDC-Sim, please cite:

> Fatih Acun, Can Hankendi, Ethan Levine, Hudson Reynolds, Joshua Bardwick, and Ayse K. Coskun. 2026. **Investigating Power Consumption Flexibility of AI Data Centers for Demand Response Participation.** In *The 17th ACM International Conference on Future and Sustainable Energy Systems (E-Energy '26)*, June 22-25, 2026, Banff, AB, Canada. ACM, New York, NY, USA.

DOI:

https://doi.org/10.1145/3744255.3798112

---

# Research Notes

When extending this codebase:

1. Treat **Model V3 / Sweep V3 as the active baseline** for full behavior-model training.
2. Do not reduce simulator coverage substantially without auditing feasibility and behavioral-boundary density.
3. Keep exact simulation plans deterministic and auditable.
4. Keep repeated seeds grouped across dataset splits.
5. Keep cost-function constants consistent across simulation, training, inference, and optimization.
6. Treat FlexDC as ground truth when evaluating learned models.
7. Keep isolated architecture experiments, such as J=2 family holdout, separate from the primary training-data methodology.
8. Do not commit generated datasets or temporary experiment artifacts to the source repository.