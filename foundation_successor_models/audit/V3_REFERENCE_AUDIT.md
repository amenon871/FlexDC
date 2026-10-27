# V3 reference audit (read-only)

The FlexDC checkout contains the collector and profile-split helper, not the successful neural training artifact. The authoritative successful artifact was located in the adjacent ARGOS Hybrid checkout. Absolute source paths and byte hashes are in REFERENCE_FILES.json. The actual training notebook is in adjacent comder-main/am_flexdc/train/. No historical checkpoint is loaded into a new Foundation model.

## Verified implementation

- data_center_model_flexdc_behavior_v3.py: FlexDCBehaviorModelConfig, MaskedMAB/SAB/PMA, DataCenterBehaviorModel.
- am_flexdc_behavior_training_utilities_v3.py: build_token_features_raw, build_global_features_raw, _guarded_mean_std, build_training_sampling_weights, behavior_loss, reconstruct_flexdc_outputs, cosine_warmup_lr, train_behavior_model and checkpoint routines.
- am_unified_model_training_wandb_colab_paths_configured_objective_flexdc_behavior_v3.ipynb: actual code cells 18/20/24/26/28/30/32 establish recipe and validation-only selected role best_feasibility; test is explicitly guarded.
- test_flexdc_behavior_training_v3.py: masking, permutation, features, objective and training/reference tests.
- flexdc_generic_sources/flexdc_profile_split.py: grouped profile holdout and seed-group isolation; not reused to replace Foundation's predeclared splits.

Architecture: 13 job features, 12 global features, two masked SABs with hidden width512, four heads, per-head sqrt(d_head) attention scaling, residual attention/SiLU feedforward, LayerNorm. PMA with one seed plus masked mean pooling; global MLP128/128; three512-wide SiLU trunk layers with the exact residual topology; tracking head2, shared QoS token/context projections256 and shared256-hidden sigmoid head. Masks explicitly supplied; padded queries zeroed.

Features, in order: pmin_kw,pmax_kw,power_range_kw,log10_tmin,log10_tmax,runtime_ratio,qos_threshold,job_size,qos_headroom,weight,weight_relative_to_equal,allocated_nodes_per_1000,queue_pressure. Globals: Pbar,R,Pbar-R,Pbar+R,R/Pbar,Pbar_ratio,R_ratio,U,N/1000,log10(N),1/J,J/8. Runtime ratio=tmax/tmin; headroom=qos/runtime_ratio; pressure=U/(J*weight).

Labels: natural log(mean+1e-6), natural log(p90+1e-3), Pj probabilities. Features use TRAIN population mean/std, with std<1e-3 replaced by1; padded tokens zeroed AFTER standardization. Tracking loss is SmoothL1 on logs divided by TRAIN target std (no mean subtraction), weights1/1. QoS masked Brier weight4. p90 band[.20,.40] gets loss multiplier2; per-job Pj band[.05,.15] gets multiplier2; weighted denominators normalize these losses.

Sampler: base-geometry groups normalized across repeats; 50% uniform geometries,25% context-balanced geometries,25% priority geometries. Priority multipliers combine feasible4, tracking-band2 and QoS-band3 using group fractions. Validation remains natural/unweighted.

Training: AdamW lr3e-4,min1e-6,weight_decay1e-4, default betas(.9,.999),eps1e-8; batch2048;250 epochs;5 epoch warmup followed by cosine;clip1; early stop patience30,min_delta1e-5. Notebook selected role best_feasibility. Utilities save latest,best_loss,best_objective,best_feasibility,final; resume optimizer/epoch/history/best state. Historical checkpoints do NOT contain complete Python/NumPy/Torch/sampler RNG or GradScaler state, and have no latest_previous; the successor package intentionally adds these reliability requirements.

Costs: M_track=R_actual_watts*mean*piE*3600; M_RSR=M_power+M_track. Objective=M_RSR+softplus(10*(p90-.3))+20*sum softplus(2*(Pj-.1)). Equality passes p90<=.3 and all Pj<=.1. Use serialized actual power/price fields, not a learned scalar cost.

## Intentional Foundation changes

Random initialization; no historical learned weights. Append two hardware one-hot GLOBAL features (P100,A100), yielding14 globals, while keeping the13 job features and set architecture. Fit all feature/target statistics on Foundation TRAIN only. Task_ID replaces old workload/context grouping. Add strict Foundation schema validation, shared frozen Foundation prior, residual models, complete atomic resume and live progress. Model/training defaults above otherwise carry forward. Exact numerical parity tests cover the original12-global architecture and ported features/loss/objective; hardware augmentation is separately identified, not called byte-identical V3.

The notebook MODEL_CONFIG explicitly confirms512 hidden,4 heads,128 global,512 trunk and256 QoS; these are not inferred solely from dataclass defaults. The generic full-parity profile-holdout training notebook was also inspected read-only; it applies preassigned grouped profile splits and retains validation-only checkpoint selection. Target-log std has its own original1e-8 guard, distinct from the1e-3 feature guard.
