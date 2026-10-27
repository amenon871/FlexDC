# Foundation successor models (PRE-TRAINING REVIEW)

Foundation Baseline vs Meta V4 vs FlexPFN, using the same Foundation labels, frozen Baseline prior, canonical objective, K values and fixed DEVELOPMENT episodes. No real Foundation training or FlexDC campaign is run by preparing this package. This review implementation is uncommitted.

## Environment

Install inside a dedicated training environment, from the existing FlexDC checkout:

```bash
python -m pip install -r foundation_successor_models/requirements.txt
```

Validated on CPU with Python3.12.4, PyTorch2.4.1+cpu, NumPy2.2.6, pandas2.2.2, SciPy1.14.0. This training environment is separate from the pinned Python3.10 collection environment; it does not generate scientific coordinates. Scikit-learn is used only to import the read-only historical V3 utilities in the parity test. W&B is optional at runtime. No CUDA wheel URL or BU scheduler resource names are assumed.

## Data -> cache

Wait for the COMPLETE canonical Foundation exports. The input root must contain the final TRAIN, DEVELOPMENT and LOCKED_SYNTHETIC_TEST individual exports. TRAIN_DEV combined exports are deliberately ignored to prevent duplicate rows. Input CSVs and source manifests remain authoritative. Cache files are disposable and contain all partitions, but training only indexes TRAIN labels and DEVELOPMENT validation labels.

```bash
python -u foundation_successor_models/prepare_cache.py \
  --input /path/to/completed/foundation_v1_runs \
  --output foundation_successor_models/cache/foundation_v1
```

This validates500000 rows,500 workloads,2000 tasks,400000/50000/50000 partition rows,250 rows/task,230 seed20 geometries and10 paired21/22 repeat groups/task, Phase-A/wave quotas, identity, dimensions, legal physical domains, evidence and canonical objective. It requires the completed export source/provenance sidecars. It never ingests an incomplete campaign database or worker directory. `--profiles` and `--workloads` default to the existing immutable Foundation artifacts; override when relocating them.

Features are the original13 per-job and12 global engineered V3 features plus2 global hardware one-hot indicators. TRAIN-only population normalization uses the proven feature std<1e-3 guard; tracking target std uses the original separate1e-8 guard. Padding is zeroed after normalization. Scheduling features are engineered proxies, not fixed server ownership: the simulator's dynamic AQA semantics are unchanged.

## Foundation Baseline

A NEW random initialization, not historical V3 learned weights. Two masked512-wide Set Attention Blocks,4 heads, PMA1 seed + masked mean,128-wide global MLP,512-wide residual trunk, shared256-wide per-job QoS head. Predict log(mean+1e-6),log(p90+1e-3),Pj. Loss is1*mean SmoothL1 +1*p90 SmoothL1 +4*masked QoS Brier, with original boundary emphasis. Canonical cost is analytic, never a learned cost head.

```bash
python -u foundation_successor_models/train_baseline.py \
  --config foundation_successor_models/configs/baseline.yaml \
  --cache foundation_successor_models/cache/foundation_v1 \
  --run-dir foundation_successor_models/runs/baseline_seed101 \
  --seed 101 --wandb online
```

Defaults: AdamW3e-4,min1e-6,decay1e-4,batch2048,250 epochs,5-epoch warmup+cosine,clip1,early-stop30 epochs/min_delta1e-5. Validation is natural/unweighted. Training sampler is50% uniform base geometries,25% Task-balanced,25% feasibility/boundary priority, with repeats sharing geometry mass. Baseline best_feasibility retains the original global F1 selection rule; Task-macro F1 is additionally prominent in reports.

After training and DEVELOPMENT-only selection, freeze the chosen checkpoint. Do not overwrite that prior file while training either successor.

## Freeze ONE DEVELOPMENT episode bank

```bash
python -u foundation_successor_models/evaluate.py \
  --cache foundation_successor_models/cache/foundation_v1 \
  --build-dev-bank foundation_successor_models/runs/development_bank.json \
  --episodes-per-task 10 --query-count 32 --selection random --seed 73
```

200 DEVELOPMENT tasks *8 K values *10 episodes =16000 episodes. K=0,1,2,4,8,16,32,64. Context/query indices are disjoint by Base_Geometry_ID and ordinary episodes use seed20 only. `--selection diverse` uses input-only farthest-point selection; `--selection nested` uses one maximin ordering and fixed queries per task/replicate, with exact prefix contexts across K. Models never select their own evaluation context points.

## Meta V4 and FlexPFN

Both REQUIRE the same selected frozen Foundation Baseline. Meta V4 learns small kernel projections and exact Matern5/2 ARD residual GPs, with independent tracking outputs and shared job-aware QoS GP parameters. Cholesky solves use float32 outside autocast, positive scales/noise and bounded jitter escalation. No explicit matrix inverse, BoTorch or GP framework.

FlexPFN builds global/per-job observation tokens from frozen latents and observed residuals. A256-wide,4-layer,8-head position-free context Transformer feeds independent query cross-attention. Queries do not attend to other queries. Shared job heads preserve permutation behavior. Both models correct tracking in the original log space and QoS in logit space. QoS epsilon1e-4 is provisional. Both hard-bypass all adaptation at K=0, returning the exact Baseline predictions; no uncertainty is claimed at K=0.

```bash
python -u foundation_successor_models/train_meta_v4.py \
  --config foundation_successor_models/configs/meta_v4.yaml \
  --cache foundation_successor_models/cache/foundation_v1 \
  --baseline-checkpoint foundation_successor_models/runs/baseline_seed101/best_feasibility.pt \
  --dev-bank foundation_successor_models/runs/development_bank.json \
  --run-dir foundation_successor_models/runs/meta_v4_seed101 --seed 101 --wandb online

python -u foundation_successor_models/train_flexpfn.py \
  --config foundation_successor_models/configs/flexpfn.yaml \
  --cache foundation_successor_models/cache/foundation_v1 \
  --baseline-checkpoint foundation_successor_models/runs/baseline_seed101/best_feasibility.pt \
  --dev-bank foundation_successor_models/runs/development_bank.json \
  --run-dir foundation_successor_models/runs/flexpfn_seed101 --seed 101 --wandb online
```

Episodic defaults:16 episodes*32 queries*4 accumulation=2048 query targets/update,200 updates/virtual epoch,1000-step warmup,50000 maximum steps,full K-curve every1000 updates,6000-step early-stop patience. Positive K values are sampled approximately uniformly; context selection mixes random/diverse50/50. `best_fewshot.pt` ranks DEVELOPMENT Task-macro joint-feasibility F1 averaged over positive K, ties by lower Task-macro false-feasible rate then lower Task-macro objective MAE. A single model handles every K.

GP/PFN training uses boundary-weighted predictive query NLL with output-family weights1/1/4. Few-shot uncertainties are in transformed residual space; coverage and interval-width reports state that space. Three observed runtime replicas do not certify calibration.

## Resume, precision, walltime, logs

Repeat the SAME config/seed/prior/dev-bank arguments with `--resume auto` or `--resume /path/to/checkpoint.pt`. Auto falls back to `latest_previous.pt` if latest is corrupt, reports fresh start if neither exists, and refuses if existing candidates are all invalid. Config/cache/seed/prior identities must match. Source Git SHA and actual code hashes are recorded; an extracted review ZIP without Git records its snapshot base instead.

```bash
python -u foundation_successor_models/train_baseline.py \
  --config foundation_successor_models/configs/baseline.yaml \
  --cache foundation_successor_models/cache/foundation_v1 \
  --run-dir foundation_successor_models/runs/baseline_seed101 \
  --seed 101 --resume auto --wandb online \
  --max-runtime-hours 12 --exit-before-minutes 10
```

The same runtime/resume flags work for both successors. A small SIGINT/SIGTERM flag saves at the next optimizer boundary and exits cleanly. Atomic checkpoints include model,optimizer,scheduler position,all RNG states,episode RNG,sampler position,normalization,config/cache/prior hashes,early-stop/history/best metrics,W&B ID and source/environment metadata. BF16 does not use a GradScaler; its checkpoint field is explicitlyNone. `auto` precision uses supported CUDA BF16, otherwise fp32; GP matrix algebra stays float32. An already early-stopped or complete resume performs zero optimizer updates.

Run directories contain training.log,history.csv,environment.json,run_config.yaml,latest.pt,latest_previous.pt,best_loss.pt,best_objective.pt,best_feasibility.pt,final.pt and best_fewshot.pt for successors, plus DEVELOPMENT metrics/K-curves and breakdowns. Live tqdm displays epoch/virtual epoch,step,loss components,LR,memory,speed/ETA; validation/checkpoint/resume events are also persisted. W&B online/offline/disabled modes never replace local records; no credentials are hardcoded or read from .env files by this package.

## Evaluation / comparison

```bash
python -u foundation_successor_models/evaluate.py \
  --model all --cache foundation_successor_models/cache/foundation_v1 \
  --baseline-checkpoint /path/to/FROZEN_SELECTED_BASELINE.pt \
  --meta-checkpoint /path/to/meta_v4/best_fewshot.pt \
  --flexpfn-checkpoint /path/to/flexpfn/best_fewshot.pt \
  --bank foundation_successor_models/runs/development_bank.json \
  --output foundation_successor_models/runs/comparison --device cuda
```

Comparison rejects differing prior identities. Outputs include all K values, physical tracking/Pj/objective errors,R2/Spearman,feasibility precision/recall/F1/balanced accuracy,Task-macro metrics,uncertainty NLL/coverage50/90/95/width/error association and hardware/J/N/U/source/workload/Task breakdowns. Per-task canonical job-position MAE is available separately; positions do not represent a common physical family across different workloads. False-feasible rate=FP/(FP+TN),false-infeasible rate=FN/(FN+TP),false-discovery rate is also recorded. Undefined R2/Spearman are JSONnull, never invented numeric scores. A single-class balanced accuracy averages only observed classes; F1 uses zero for undefined positive-class support.

LOCKED evaluation requires `--partition locked --unlock-locked-test` and a separately built locked episode bank. No training CLI can select LOCKED. Locked outputs are named locked_metrics/locked_k_curve and retain partition identity; they must never feed model selection.

Repeats remain in cache and Baseline training with normalized group sampling. `evaluate.py --repeat-analysis --cache ... --output ...` emits empirical transformed noise statistics across seeds20/21/22. This does NOT automatically tune a GP noise floor. `--repeat-assessment` is a separately labelled same-geometry seed20 -> seed21/22 diagnostic, deliberately outside ordinary geometry-disjoint evaluation. Ordinary episode validation rejects those records.

## Benchmark BEFORE full L40S training

Run500–1000 real optimizer steps on the actual allocated GPU BEFORE choosing full-job walltime. For example:

```bash
python -u foundation_successor_models/benchmark.py \
  --model flexpfn --config foundation_successor_models/configs/flexpfn.yaml \
  --cache foundation_successor_models/cache/foundation_v1 \
  --baseline-checkpoint /path/to/FROZEN_SELECTED_BASELINE.pt \
  --steps 500 --run-dir foundation_successor_models/runs/benchmark_flexpfn
```

Baseline and meta_v4 use the same benchmark entry. It reports compute walltime,steps/s,query predictions/s,data wait,forward/backward/optimizer timing,CPU utilization,PyTorch peak allocated/reserved VRAM and projected compute-only training duration. GPU utilization is null when not queried; NVIDIA monitoring is not a dependency. CUDA events time matrix work; no per-step NVIDIA polling. Projections exclude validation/checkpoint overhead and are not guaranteed runtime. The earlier1–3h/3–8h/5–12h figures are NOT guarantees. GPU behavior/performance remains unvalidated on this CPU-only development machine.

Use an already allocated SCC GPU node, activate your training venv and run these Python commands. No unverified scheduler names/directives are supplied.

## Local correctness / complete mini pipeline

```bash
python -B -m pytest -c foundation_successor_models/pytest.ini foundation_successor_models/tests \
  --confcutdir foundation_successor_models \
  --basetemp foundation_successor_models/.validation_runs/pytest -q
python -u -B foundation_successor_models/smoke.py \
  --output foundation_successor_models/.validation_runs/smoke
```

Use a NEW smoke output directory on repeat invocations. Fixtures are prominently TEST FIXTURE ONLY / NOT SCIENTIFIC DATA. Tiny dimensions/effective batches are only software tests, not additional scientific model definitions. Exact CPU model/optimizer/LR/RNG resume parity is tested for all3 trainers. Float32 GP permutation comparisons allow5e-5 conditioning roundoff; Baseline reference parity and K=0 are exact. GPU correctness is not asserted.

## Review bundle / isolation

Read audit/IMPLEMENTATION_AUDIT.md,REVIEW_GUIDE.md,V3_REFERENCE_AUDIT.md and FOUNDATION_DATA_CONTRACT.md. Run `python foundation_successor_models/package_review.py` only after validation. All changes live in this one new directory. No original V3/collector/scientific plan is edited. Reference sources/notebooks are copied only into ZIP reference_only/, with original hashes; full reference parity tests require those references or the original adjacent checkout. The package and review ZIP remain uncommitted: no add/commit/push/merge is performed.

Provisional DEVELOPMENT decisions: QoS logit epsilon; Meta kernel projection width; PFN width/depth;1e-4 vs3e-4 episodic LR; feasible microbatch/accumulation on the real GPU; empirical uncertainty/noise-floor calibration. No DEVELOPMENT winner or model-quality conclusion has been selected during preparation.

## Already-allocated SCC node example

This checkout path is the example supplied for the project; adjust it to the actual deployment. No scheduler resource names are assumed.

```bash
cd /projectnb/peaclab-mon/amenon06/FlexDC-FoundationV1
source /path/to/training-venv/bin/activate
# Run the prepare/benchmark/train/resume commands above from this checkout.
```

W&B was unavailable in the local validation environment. Disabled/unavailable fallback passed; authenticated online/offline integration is not claimed tested. Installing requirements.txt provides the optional W&B package for later use.


## Supervisor correction pass (uncommitted review)

Read audit/PATCH_AUDIT.md for the ten focused corrections. DEVELOPMENT banks are frozen and hashed in run directories; source identity is enforced on resume. Cache schema is now V2 (canonical Foundation geometry and retained evidence counts); rebuild derived V1 caches. No architecture/loss change. Latest checkpoint cadence defaults to Baseline 200 and Meta/PFN 500 updates. Partial stops never write final.pt. Routine validation writes compact summaries; use evaluate.py explicitly for detailed predictions. The master supervisor ZIP, including the separately approved collector recovery patch, supersedes the old package_review.py isolation-only bundle workflow. Do not overwrite or reuse the old review ZIP.


## Final four-fix review

The authoritative current review is supervisor_final_fix_audit/FINAL_FIX_AUDIT.md in the new supervisor ZIP. Fresh training now refuses any nonempty run directory without explicit --resume; no files are deleted or renamed. Model architecture/loss/feature/cost definitions are unchanged. Training tests auto-bootstrap read-only Foundation references after unzip; run python -B run_supervisor_tests.py from the extracted archive root. Earlier audit counts describe earlier review revisions.
