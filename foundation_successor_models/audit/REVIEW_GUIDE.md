# Supervisor review map

1. V3_REFERENCE_AUDIT.md: actual notebook/source, architecture, recipe, parity and intentional hardware/RNG changes. REFERENCE_FILES.json seals read-only references; ZIP reference_only/ contains them without duplicating historical code into the runnable package.
2. FOUNDATION_DATA_CONTRACT.md and flexmodels/data.py: strict exports, full quotas, TRAIN-only statistics, dimensions/evidence, cache/source hashes and dynamic physical domains. features.py implements13+14 features.
3. baseline.py: masked SAB/PMA/mean/trunk/shared QoS head and optional h_x/e_x,j. test_models.py compares random-weight original V3 architecture,feature formulas,loss and objective exactly.
4. objective.py: ONE transform/threshold/cost/loss definition. metrics.py: physical errors,Task-macro feasibility,undefined-score handling,NLL/coverage and breakdowns. Intervals are transformed-space quantities, not certified reliability.
5. episodes.py: task-uniform random/diverse contexts,geometry-disjoint query validation,deterministic/nested shared banks. Dedicated repeat_bank records are explicitly outside ordinary episodes; validate_episode rejects them.
6. meta_v4.py: exact ARD Matern5/2 kernels,shared job-aware QoS parameters,float32 Cholesky/jitter,positive scales/noise. No inverse. K=0 returns the frozen prior directly.
7. flexpfn.py: position-free global/job observation tokens,context self-attention,independent query cross-attention,shared mean/log-std heads. No query-to-query self-attention or learned null-context offset.
8. checkpoint.py and training.py: atomic latest/previous roles,full optimizer/scheduler/RNG/sampler/early-stop state,prior/cache/config checks,signal/walltime handling,tqdm/local/W&B logging. evaluate.py enforces prior identity and LOCKED unlock.
9. configs/: original Baseline recipe and visibly provisional episodic epsilon/projection/Transformer/batch/LR choices. No large search or model-quality tuning was performed.
10. tests/: data rejection,normalization,isolation,all K,masking,permutation,GP stability,shared banks,actual resume parity,corruption fallback,signals,walltime,early-stop idempotency. smoke.py exercises actual CLIs through cache/all3trainers/evaluation/resume. CPU only; GPU/BF16 performance requires later L40S validation.
11. TEST_RESULTS.txt,test_results.xml,sample_training_output.txt,MINI_PIPELINE_RESULT.json,BENCHMARK_RESULTS.json,IMPLEMENTATION_AUDIT.md contain observed results,not formatting examples.
12. FILE_INVENTORY.json and the ZIP top-level FILE_HASHES.sha256 seal contents. The inventory excludes its own self-reference and dist/runtime caches/checkpoints explicitly; no recursive ZIP or real dataset is included.

Baseline prior is a NEW Foundation-trained random-initialization model, never historical V3 learned weights. All3 methods share the exact physical target/objective and one frozen prior when compared. No pre-existing source is edited; no real Foundation training/FlexDC experiment is started; no commit,push or merge is performed.


## Supervisor corrections

Start with PATCH_AUDIT.md and ADVERSARIAL_TEST_RESULTS.md. Training identity and exit logic are in flexmodels/training.py; frozen bank publication is episodes.freeze_bank; canonical coordinates are features.canonical_coordinates; provenance/quota gates and evidence_counts are in data.py. Detailed prediction writes are controlled by metrics.save_evaluation. The master supervisor guide also maps the separate collector recovery patch.


## Final four-fix review

The authoritative current review is supervisor_final_fix_audit/FINAL_FIX_AUDIT.md in the new supervisor ZIP. Fresh training now refuses any nonempty run directory without explicit --resume; no files are deleted or renamed. Model architecture/loss/feature/cost definitions are unchanged. Training tests auto-bootstrap read-only Foundation references after unzip; run python -B run_supervisor_tests.py from the extracted archive root. Earlier audit counts describe earlier review revisions.
