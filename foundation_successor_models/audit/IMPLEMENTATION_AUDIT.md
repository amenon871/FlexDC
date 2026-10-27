# Foundation successor models: implementation audit

## Repository and scope

Base branch foundation-dataset-v1-prep, base HEAD0190fe60d53cddb6afbaffad9a67b82e50ccf0de. Working branch foundation-successor-models-prep. The branch was created from a clean tree. No git add,commit,push or merge is performed. Every new source/config/test/fixture/output/packaging file is inside foundation_successor_models/. The completed review's ISOLATION_RESULT.json and FILE_INVENTORY.json record exact status,content hashes and every added file. Root FILE_HASHES/preexecution seals are NOT modified.

V3 sources were absent from the FlexDC checkout, so the actual successful exported model/training utilities/tests were inspected in the adjacent ARGOS Hybrid artifact directory; the actual V3 training notebook and generic profile-holdout notebook were inspected in adjacent comder-main. Exact absolute paths and hashes are recorded in REFERENCE_FILES.json, with read-only copies in ZIP reference_only/ only. No historical checkpoint was loaded for new initialization or training. Reference parity uses random weights only.

## Added implementation map

- flexmodels/baseline.py: original masked SAB/PMA/mean/residual-trunk design, new random Foundation weights,13 job/14 global hardware-aware inputs,optional frozen candidate/job latents.
- features.py: exact V3 engineered features and separate feature/target constant guards; one workload-specific legal-domain check.
- objective.py: centralized canonical transforms,feasibility,cost constants,boundary ranges,Baseline loss and predictive NLL.
- data.py / prepare_cache.py: canonical final exports only,full-corpus checks,physical/evidence/group validation,TRAIN-only normalization,one disposable tensor bundle,provenance/cache hashes,TRAIN sampler.
- episodes.py: uniform Task sampling,input-only random/maximin selection,ordinary geometry-disjoint seed20 episodes,all8 K,shared deterministic/nested banks,separate repeat noise/scenario diagnostics.
- meta_v4.py: pure PyTorch exact deep-kernel residual GP. Global mean/p90 use independent kernel parameters; real-job QoS uses one shared job-aware kernel formulation. Positive ARD/scales/noise; bounded jitter; no matrix inverse. Baseline remains frozen.
- flexpfn.py: global/per-job observation tokens,position-free Transformer memory,independent query cross-attention,shared job outputs,mean/log-std residual heads. Baseline remains frozen.
- checkpoint.py: full atomic model/optimizer/schedule/RNG/sampler state,latest/previous fallback,identity validation,strict refusal of historical V3 training checkpoint formats.
- training.py / train_baseline.py / train_meta_v4.py / train_flexpfn.py: one lean PyTorch loop with row/episode batches,gradient accumulation,validation,best roles,early stopping,tqdm,local history,W&B modes,bf16/fp32 and signal/walltime checkpointing.
- metrics.py / evaluate.py: common physical/objective/feasibility/Task-macro/uncertainty metrics,breakdowns,per-task canonical job-position MAE,same-prior comparison,LOCKED unlock and separately named LOCKED reports.
- benchmark.py: short actual optimizer timing through the SAME training loop,not a separate scientific implementation.
- smoke.py: actual CLI fixture->cache->Baseline->frozen prior->shared bank->both successors->comparison->resume all3->comparison.
- tests/: data rejection,normalization,all K,masking,permutation,GP stability,objective/reference parity,ordinary/repeat separation,shared-prior/bank guards,full resume,corruption,signals,walltime and portable source metadata.
- configs/{baseline,meta_v4,flexpfn}.yaml,requirements.txt,README.md,fixtures/make_fixture.py,package_review.py and audit/: user configuration,synthetic schema,documentation and independent-review package.

## Foundation contract / isolation

See FOUNDATION_DATA_CONTRACT.md. The real final gate is500000 rows,400 TRAIN/50 DEV/50 LOCKED workloads,2000 tasks,250 rows/task with230 primary geometries and10 paired repeat groups. Actual Foundation J range2..6 is read from immutable workload definitions. Only complete valid evidence labels are accepted; missing jobs never become zero probabilities. TRAIN numerical normalization and gradient updates are distinct from natural DEVELOPMENT selection. LOCKED is encoded only during explicit cache preparation; no trainer indexes its labels for fitting or selection. Independent LOCKED evaluation requires unlock and uses a separate output identity.

Full final real-corpus cache preparation has NOT been executed because the real dataset may still be collecting. The incomplete fixture fails the production full-count gate without --fixture. Synthetic fixtures are clearly labelled and must not be interpreted as FlexDC scientific outcomes. No real Foundation export or worker ledger is used to fit these models during preparation.

## Baseline behavior / confirmed recipe

13 original job features,12 original global features +2 hardware one-hot indicators. Two512-wide masked SABs/4 heads,one-seed PMA plus masked mean,128 global MLP,512 residual trunk,256 shared QoS projection. Labels log(mean+1e-6),log(p90+1e-3),Pj; predicted scalar cost is forbidden. Formula M_power+R_actual*mean*piE*hour+softplus(10*(p90-.3))+20*sum softplus(2*(Pj-.1)). Equality-inclusive thresholds. Original random-weight architecture,features,loss and objective parity tests are exact; hardware augmentation is an explicit change rather than a byte-identical V3 claim.

AdamW3e-4/min1e-6/decay1e-4,betas(.9,.999),eps1e-8,batch2048,250 epochs,5 warmup epochs+cosine,clip1,patience30,min_delta1e-5. Tracking SmoothL1 weights1/1,QoS Brier4; p90[.2,.4] and Pj[.05,.15] boundary loss multipliers2. Sampling mixes50/25/25 natural-geometry/Task-balanced/priority mass with4/2/3 group boosts and repeat normalization. DEVELOPMENT natural/unweighted. Baseline best_feasibility uses the original global joint F1; Task-macro metrics are additionally reported.

## Successor models

Both require the same selected Foundation prior file and cache normalization; the comparison entry rejects different prior identities. Encoder weights are frozen; context/query candidates are encoded in batched prior calls. K=0 returns Baseline outputs exactly with uncertainty unavailable, avoiding a learned null-context model. Positive K are sampled approximately uniformly.

Meta V4: global projection and shared(candidate,job) projection into latent_dim16; three kernel-parameter families(mean,p90,sharedQoS),Matern5/2 ARD,softplus-positive covariance/noise,Cholesky solves with jitter escalation. Float32 matrix operations explicitly disable bf16 autocast; the whole model is NOT float64. Float32 conditioning/permutation checks allow5e-5 absolute/relative tolerance, while K=0 and original Baseline parity are exact. An initial overly strict5e-6 permutation test saw1.14e-5 roundoff; the declared tolerance reflects float32 solve conditioning, not a modified allocation/model method. Historical development test logs are retained; TEST_RESULTS.txt is authoritative for the final source.

FlexPFN: default256-wide,4-layer,8-head,GELU/LayerNorm,dropout0 context Transformer. No observation position embeddings. Global tokens contain candidate latent+tracking residuals; shared per-job tokens contain candidate/job latents+QoS residual. Query tokens cross-attend only to context memory,never to other queries. Shared tracking/job mean+log-std heads; log-std bounded[-7,3]. Scalar invariance,Pj equivariance,context-order invariance,padding and query-batch independence are tested.

Residual tracking uses original log space; QoS uses configurable epsilon-clipped logit space. Probabilistic query NLL weights1/1/4 with boundary emphasis. Uncertainty metrics are transformed-space NLL/coverage50/90/95/width/error association; existence of variance is NOT calibration evidence. Repeats support descriptive noise analysis and an explicitly separate same-geometry runtime diagnostic,not ordinary spatial episodes or automatic noise-floor fitting.

Defaults16 episodes*32 queries*4 accumulation=2048 queries/update;200 updates/virtual epoch,1000 warmup,50000 max,full DEVELOPMENT validation every1000 updates,6000-update early-stop patience. Ten episodes per DEVELOPMENT Task per K yields16000 shared episodes. best_fewshot ranks positive-K Task-macro F1,then lower Task-macro false-feasible rate,then lower Task-macro objective MAE. No LOCKED score participates.

## Checkpoints, interruption and training UX

Atomic temporary write/fsync/rename with previous-latest preservation. latest.pt/latest_previous.pt,best_loss,best_objective,best_feasibility,final,and best_fewshot for successors. Full optimizer momentum,LR position,epoch/virtual epoch,global step,best scores/steps,history,early-stop state,normalization,model/train configs,cache/prior hashes,Python/NumPy/Torch CPU/CUDA RNG,episode RNG,sampler seed/offset,W&B ID and source/environment metadata are saved. BF16 uses no GradScaler; the field is explicitlyNone. Auto resume falls back from corrupt latest,refuses incompatible config/cache/seed/prior,and performs zero updates for completed or early-stopped runs. Resume parity compares ALL parameter/optimizer tensors,LR position and next RNG values,not just successful loading.

SIGINT/SIGTERM and configurable walltime margin stop at an optimizer boundary,save latest/final,flush history and return cleanly. This is a signal flag,not an orchestration framework. Source metadata supports either Git SHA or an extracted snapshot-base SHA plus actual package-code hashes.

Visible startup banner,tqdm epoch/virtual epoch/global step,loss families/NLL,LR,Kmean,memory,speed/ETA; concise validation/K-curve and checkpoint/resume messages; same events persist in training.log/history.csv. Real fixture console excerpts are in sample_training_output.txt. W&B online/offline/disabled modes are optional; local outputs survive unavailable W&B. Authenticated online W&B and GPU/BF16 service integration are NOT claimed tested; disabled/unavailable behavior is tested. No credentials or .env files are accessed or bundled.

## Validation and benchmark interpretation

See TEST_RESULTS.txt/test_results.xml,VALIDATION_SUMMARY.json,MINI_PIPELINE_RESULT.json and BENCHMARK_RESULTS.json for final measured results. All correctness tests use CPU. The full tiny actual-CLI pipeline and all3 exact CPU resume tests must pass before packaging. The minipipeline's frozen prior copy remains unchanged even when the original Baseline trainer is resumed. All methods then consume the exact same comparison bank/prior.

Tiny CPU benchmarks execute only2 optimizer steps/model to validate timing/output paths. Their throughput is NOT a projection for512-wide Baseline/256-wide PFN/L40S/full data. Before any full SCC run the user MUST benchmark500–1000 steps on the real allocated GPU,record VRAM/steps/s,and include validation/checkpoint overhead when choosing walltime. No scheduler resources are guessed. Exact future commands are in README.md.

Validation environment: Python3.12.4,PyTorch2.4.1+cpu,NumPy2.2.6,pandas2.2.2,SciPy1.14.0; all package versions and machine availability are recorded in VALIDATION_SUMMARY.json. No CUDA device is available. No full400k/500k training,no FlexDC simulations,no few-shot scientific assessment,no retraining of historical V3,and no controller modification is performed.

## Provisional DEVELOPMENT choices

QoS logit epsilon1e-4;Meta projection16;PFN256/4layers/8heads;episodic LR3e-4 versus a small manual1e-4 comparison;microbatch16/accum4 subject to real GPU memory;eventual uncertainty/noise-floor calibration from allowed data. These are starting configurations,not DEVELOPMENT winners. No large tuning framework is included.

## Final isolation statement

NO PRE-EXISTING V3 TRAINING, INFERENCE, OPTIMIZATION,
FOUNDATION COLLECTOR, OR SCIENTIFIC-PLAN FILE WAS MODIFIED.

NO ORIGINAL V3 TRAINING FILE WAS MODIFIED.
NO ORIGINAL V3 INFERENCE/OPTIMIZATION FILE WAS MODIFIED.
NO FOUNDATION V1 COLLECTOR OR SCIENTIFIC PLAN FILE WAS MODIFIED.
ALL NEW RUNNABLE TRAINING CODE LIVES UNDER foundation_successor_models/.

NO COMMIT WAS CREATED. NO PUSH WAS PERFORMED. NO MERGE WAS PERFORMED.

The final package guard verifies every pre-existing tracked-file hash and every read-only V3/Foundation reference hash. `git status --short` should show only the new foundation_successor_models/ directory; `git diff --name-only` and staged diff must be empty. Exact captured output is in ISOLATION_RESULT.json. Inventory and archive hash checks cover the review bundle without recursive dist,real data,caches,checkpoints,W&B,Git or pycache contents.

## Final saved-output consistency correction

A stronger post-save check caught one-ULP CPU differences when standalone Baseline evaluation retained requires_grad flags but successors used a frozen prior. Identical weights were verified; freezing standalone evaluation reproduced exact equality. evaluate.load_model now freezes every evaluation model. A multi-threaded regression and the smoke script both require bit-exact saved K=0 physical predictions. This is an inference consistency correction in the NEW package, not a change to historical V3 or collector science. Earlier mini output checks are superseded by the final frozen-evaluation rerun.

## Final measured acceptance results

93 passed,0 failed,0 skipped,0 warnings in43.31s. Final actual-CLI mini pipeline PASS; Baseline6 updates,Meta4,PFN4,each resumed. Saved K=0 physical predictions match exactly across all3 methods after the frozen-evaluation consistency correction. Full512-width random-weight V3 reference parity has zero maximum absolute difference for tracking,QoS,candidate and job latents (V3_DEFAULT_WIDTH_PARITY.json). All3 CPU2-step benchmark paths PASS; they exercise the unchanged training loop,not scientific model quality or GPU throughput. W&B is not installed in the validation ML environment; disabled/unavailable fallback is tested. No authenticated W&B or GPU/BF16 integration result is claimed. The3,921,155-parameter original reference gains256 global-input weights for the14-global-feature Foundation version. Training root seed is reported; the episodic RNG starts at root+917 and its entire state is saved. No model-quality conclusion follows from tiny fixture metrics.


## Supervisor correction result

See PATCH_AUDIT.md and VALIDATION_SUMMARY.json for the authoritative current validation. The original audit is preserved as historical preparation evidence. Architecture/loss/scientific targets are unchanged. The separate collector recovery implementation is intentionally modified only under this subsequent explicit authorization.


## Final four-fix review

The authoritative current review is supervisor_final_fix_audit/FINAL_FIX_AUDIT.md in the new supervisor ZIP. Fresh training now refuses any nonempty run directory without explicit --resume; no files are deleted or renamed. Model architecture/loss/feature/cost definitions are unchanged. Training tests auto-bootstrap read-only Foundation references after unzip; run python -B run_supervisor_tests.py from the extracted archive root. Earlier audit counts describe earlier review revisions.
