"""Pre-execution audits and review bundle, never scientific simulation."""
import csv
import gzip
import io
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

import am_generate_flexdc_exact_sweep_plan as exact
from am_flexdc_foundation_v1_common import (ROOT,ARTIFACTS,PHASE_A,Domain,contexts,
    read_json,write_json,write_csv,file_hash,weight_bounds,row_geometry,make_row,digest,parser)


def text(name,body):
    (ARTIFACTS/name).write_text(body.strip()+'\n',encoding='utf8')


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',*['|'+'|'.join(str(x) for x in r)+'|' for r in rows]])


def preflight():
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from peacsim.parsing.job_profile_reader import JobProfileReader
    data=read_json(ARTIFACTS/'profiles.json');profiles=data['synthetic_profiles'];real=data['real_anchors'];w=read_json(ARTIFACTS/'workloads.json');tasks=contexts()
    assert len(profiles)==400 and len(real)==13 and len(w)==500 and len(tasks)==2000
    lookup={x['profile_id']:x for x in [*profiles,*real]}
    assert Counter(p['split'] for p in profiles)=={'TRAIN':300,'DEVELOPMENT':50,'LOCKED_SYNTHETIC_TEST':50}
    assert Counter(p['hardware_class'] for p in profiles)=={'P100':240,'A100':160}
    compositions=set();distributions=[]
    for split,n,js,qs in [('TRAIN',400,[60,80,120,80,60],[100,60,140,100]),('DEVELOPMENT',50,[8,10,14,10,8],[12,8,18,12]),('LOCKED_SYNTHETIC_TEST',50,[8,10,14,10,8],[12,8,18,12])]:
        subset=[a for a in w if a['split']==split]
        assert len(subset)==n
        assert Counter(a['J'] for a in subset)==dict(zip(range(2,7),js))
        assert Counter(a['QoS_pattern'] for a in subset)==dict(zip(('ALL_5','ALL_4','BALANCED_MIXED','BOTTLENECK'),qs))
        assert Counter(a['hardware_class'] for a in subset)=={'P100':int(.6*n),'A100':int(.4*n)}
        for j in range(2,7):
            for h in ('P100','A100'):distributions.append({'split':split,'J':j,'hardware_class':h,'workloads':sum(a['J']==j and a['hardware_class']==h for a in subset)})
    for a in w:
        c=JobProfileReader(str(ROOT/a['INI_path']));assert c.job_type_count==a['J']
        assert file_hash(ROOT/a['INI_path'])==a['INI_SHA256']
        key=tuple(sorted(a['Profile_IDs']));assert key not in compositions;compositions.add(key)
        assert len(set(a['Profile_IDs']))==a['J']
        if a['split']!='TRAIN':assert not any(i.startswith('REAL_') for i in a['Profile_IDs'])
        for i,pid in enumerate(a['Profile_IDs']):
            p=lookup[pid]
            assert p['hardware_class']==a['hardware_class']
            assert pid.startswith('REAL_') or p['split']==a['split']
            for field,actual in [('min_job_power_watts',c.all_min_job_power[i]),('max_job_power_watts',c.all_max_job_power[i]),('min_time_seconds',c.all_min_execution_time[i]),('max_time_seconds',c.all_max_execution_time[i])]:assert p[field]==actual
            assert c.all_job_qos_constraints[i]==a['QoS_values'][i] and c.all_job_size[i]==p['job_size']
    grounded=[a for a in w if a['grounding']=='ANCHOR_GROUNDED'];assert len(grounded)==80
    assert Counter(a['hardware_class'] for a in grounded)=={'P100':48,'A100':32}
    assert all(a['real_profile_slots']>=(a['J']+1)//2 for a in grounded)
    domain_rows=[]
    for j in range(2,7):
        for n in (250,1000):
            reqlo,reqhi,l,u=weight_bounds(j,n);oldlo,oldhi=exact.calculate_legacy_weight_bounds(j,n)
            mandatory=exact.mandatory_weight_candidates(j,l,u)
            assert j*l<=1+1e-9 and j*u>=1-1e-9 and l<=1/j<=u
            domain_rows.append({'J':j,'N':n,'requested_lower':reqlo,'requested_upper':reqhi,'legacy_lower':oldlo,'legacy_upper':oldhi,'effective_lower':l,'effective_upper':u,'simplex_feasible':True,'equal_legal':True,'extreme_legal_example':list(mandatory[-1][2])})
    write_csv(ARTIFACTS/'foundation_v1_variable_J_domain_audit.csv',domain_rows)
    text('DOMAIN_CONTRACT_REPORT.md','# Foundation V1 variable-J domain audit\n\nDerived using the unchanged current exact-plan helper at both server counts.\n\n'+table(['J','N','Requested lower','Requested upper','Effective lower','Effective upper','Equal legal'],[[r['J'],r['N'],r['requested_lower'],r['requested_upper'],r['effective_lower'],r['effective_upper'],r['equal_legal']] for r in domain_rows])+'\n\nJ2 requested upper.9 narrows to.7 due to the bounded simplex. Other effective intervals:J3 [.2,.6],J4 [.15,.45],J5 [.12,.36],J6 [.1,.3]. Every proposal is validated;equal and face/corner examples are in the CSV. Pbar/R are kW/server,from each actual workload: Pbar=[.9Pmin,Pmax],R≥.01,R≤.6Pbar,Pbar+R≤1.2Pmax,Pbar−R≥0. Unit sum tolerance1e-9;inherited P/R boundary tolerance1e-9. Input legality is separate from strict numerical feasibility and required evidence.')
    write_csv(ARTIFACTS/'foundation_v1_distribution_summary.csv',distributions)
    taskmap={t['Task_ID']:t for t in tasks};domains={k:Domain(t['domain']) for k,t in taskmap.items()}
    path=ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz';seal=read_json(ARTIFACTS/'phaseA_plan_seal.json')
    assert file_hash(path)==seal['sha256']
    counts=Counter();source=Counter();ids=set();geometries={};rows=0
    with gzip.open(path,'rt',encoding='utf8',newline='') as f:
        for r in csv.DictReader(f):
            t=taskmap[r['Task_ID']];d=domains[r['Task_ID']];x=d.validate(row_geometry(r));rows+=1
            expected=make_row(t,x,r['Candidate_Source'])
            assert r['plan_row_id']==expected['plan_row_id']==r['Plan_Row_ID']==r['Foundation_Row_ID']
            assert r['Base_Geometry_ID']==expected['Base_Geometry_ID']
            assert r['plan_row_id'] not in ids;ids.add(r['plan_row_id'])
            seen=geometries.setdefault(r['Task_ID'],set());assert r['Base_Geometry_ID'] not in seen;seen.add(r['Base_Geometry_ID'])
            assert int(r['simulation_seed'])==20 and r['design_version']=='FOUNDATION_V1'
            counts[r['Task_ID']]+=1;source[r['Candidate_Source']]+=1
            for key in ('Cost_SHA256','Experiment_SHA256','Workload_SHA256','ISO_SHA256','Policy_Cluster_SHA256'):assert r[key]==t[key]
            if rows%30000==0:print(f'Preflight rows checked:{rows}/300000',flush=True)
    assert rows==300000 and len(counts)==2000 and set(counts.values())=={150}
    assert dict(source)=={k:v*2000 for k,v in PHASE_A.items()}
    assert not list(ARTIFACTS.glob('foundation_v1_wave*_plan*.csv*')) and not (ARTIFACTS/'foundation_v1_repeat_plan.csv.gz').exists()
    write_json(ARTIFACTS/'pre_execution_audit.json',{'status':'PASS','synthetic_profiles':400,'real_training_anchors':13,'workloads':500,'contexts':2000,'phaseA_rows':300000,'adaptive_expected':160000,'repeats_expected':40000,'final_expected':500000,'source_counts':dict(source),'profile_leakage':'PASS','hardware_separation':'PASS','all500_real_parser_INIs':'PASS','domain_checks':'PASS','old_V3_observation_rows':0,'simulation_calls':0,'adaptive_coordinate_plans_generated':False})
    return rows


def docs_and_package():
    state=read_json(ARTIFACTS/'pre_execution_audit.json');assert state['status']=='PASS'
    tests=read_json(ARTIFACTS/'foundation_v1_tests.json');assert tests['status']=='PASS' and tests['exit_code']==0
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    commit=git('rev-parse','HEAD');branch=git('branch','--show-current')
    source_files=sorted([*ROOT.glob('tests/test_foundation_v1*.py'),*ROOT.glob('am_*foundation*v1*.py'),*list((ROOT/'foundation_v1').glob('*.py')),ROOT/'src/peacsim/foundation_v1_evidence.py',ROOT/'src/peacsim/am_data_extraction_wizard.py',ROOT/'am_generate_flexdc_exact_sweep_plan.py',ROOT/'am_run_flexdc_dataset_parallel_combo.py',ROOT/'tests/test_foundation_v1.py'])
    provenance={str(p.relative_to(ROOT)).replace('\\','/'):file_hash(p) for p in source_files}
    # Guard every tracked simulator/parser source, not only the entry points.
    for name in git('ls-files','src/peacsim').splitlines():
        if name.endswith('.py') and not any(part.startswith('.env') for part in Path(name).parts):provenance[name]=file_hash(ROOT/name)
    for p in sorted((ROOT/'configs/foundation_v1').rglob('*.ini'))+sorted((ROOT/'configs/workload/foundation_v1').glob('*.ini')):
        provenance[str(p.relative_to(ROOT)).replace('\\','/')]=file_hash(p)
    for name in ('profiles.json','workloads.json','contexts.json','profile_partition_freeze.json','canonical_contract.json','foundation_v1_adaptive_protocol.json'):
        provenance['foundation_v1_artifacts/'+name]=file_hash(ARTIFACTS/name)
    for p in sorted(ARTIFACTS.glob('foundation_v1_phaseA__*')):
        if p.is_file():provenance[p.relative_to(ROOT).as_posix()]=file_hash(p)
    canonical_inputs=['configs/optimization/simulated_annealing/SA_train_low_util_RSR_1000server.ini','configs/gradient_descent/gradient_descent.ini','configs/cluster/cluster.ini','data/1-17-2025_traditional.csv',*['configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_'+str(n)+'.ini' for n in (250,1000)],'configs/workload/W1-train-qos4444.ini','configs/workload/W2-short-qos5555.ini']
    for name in canonical_inputs:provenance[name]=file_hash(ROOT/name)
    import platform,scipy,pandas
    write_json(ARTIFACTS/'preexecution_release.json',{'status':'PREPARED_NOT_RUN','version':'FOUNDATION_V1','base_git_SHA':commit,'branch':branch,'uncommitted_source_sha256':digest(provenance),'source_files':provenance,'phaseA_sha256':file_hash(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz'),'scientific_simulations_started':False,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__}},False)
    names=['am_generate_flexdc_sweep_plan_v3.py','am_generate_flexdc_exact_sweep_plan.py','am_run_flexdc_dataset_parallel_combo.py','am_vet_flexdc_sweep_dataset_v3.py','am_generate_paper_iso_experiment_configs.py','src/peacsim/am_data_extraction_wizard.py','am_combine_csv_v3.py','prepare_flexdc_training_dataset.py','am_generate_flexdc_j2_pair_workloads.py']
    for name in ('flexdc_generic_sources/flexdc_profile_split.py','flexdc_generic_sources/prepare_flexdc_evaluation_benchmark.py'):
        if git('ls-files','--',name):names.append(name)
    lineage=[]
    for name in names:
        old=subprocess.check_output(['git','-C',str(ROOT),'show','origin/main:'+name])
        import hashlib
        local=(ROOT/name).read_bytes()
        status='BYTE_IDENTICAL to fetched origin/main' if local==old else 'Same source text;working-copy CRLF/LF differs' if local.replace(b'\r\n',b'\n')==old.replace(b'\r\n',b'\n') else 'Foundation-only evidence hook added after clean audit' if name.endswith('am_data_extraction_wizard.py') else 'DIFFERENT:review before execution'
        lineage.append([name,hashlib.sha256(old).hexdigest(),file_hash(ROOT/name),status])
    text('SCRIPT_LINEAGE_REPORT.md',f'# Script lineage\n\nRoot: `{ROOT}`. Initial branch main, HEAD `{commit}`. Fetch succeeded;initial HEAD and origin/main matched;initial worktree clean. No pull/reset/merge/checkout over existing work. Current branch `{branch}`;no commit/push.\n\nThe nearby FlexDC_V5_Eval and its flexdc_workspace are non-Git model/data workspaces, not newer FlexDC research checkouts. The requested checkout is authoritative. The extraction wizard is under src/peacsim.\n\n'+table(['File','origin/main SHA256','local SHA256','Basis/action'],lineage)+'\n\nExisting exact-plan domain/cost/weight/boundary helpers and combo-runner are reused. Existing combine/training prep remain unchanged and are NOT used to import old observations. New Foundation export owns only Foundation row joins.\n')
    text('FOUNDATION_V1_REAL_HOLDOUT_POLICY.md','# Untouched real-family holdouts\n\nOnly the13 current measured anchors are included:8 P100 and5 A100 MoE profiles. Future2–3 profiles from Can MUST NOT enter generation, training normalization, adaptation or tuning. Reserve them as untouched REAL FAMILY HOLDOUTS. Synthetic locked-test profiles are disjoint identities within the current anchor convex envelope;this is not evidence of real-family extrapolation. Real anchors occur directly only in TRAIN workloads. Hardware classes are never mixed. A100 attribution is frozen for Foundation V1.')
    text('FOUNDATION_V1_IDLE_POWER_NOTE.md','# Canonical idle-power assumption\n\nKerim explicitly chose to preserve canonical idle_watts=120.0 for both P100- and A100-derived workloads. Foundation V1 does not introduce hardware-specific idle-node models. This is a modeling assumption, not a measurement of both hardware classes. Revisit with a separately controlled future experiment;do not change it in this campaign.')
    text('FOUNDATION_V1_NO_V3_DATA_AUDIT.md','# No old V3 observation data\n\nPASS for this pre-execution release:zero simulator observations exist in Foundation planning. Profiles read ONLY three physical INIs (two P100 sources and the copied measured MoE INI);existing code/configs are permitted reuse. No old results/diagnostics CSV, V3 checkpoint/prediction, ARGOS/FS/OC output or learned acquisition enters generation.\n\nAt execution, every terminal receipt must possess an F1_ row ID recomputed from an immutable Foundation task/geometry/seed, an atomic evidence receipt with body checksum and exact applied/config identity. Only sufficient per-job evidence enters the valid-label table; technically complete insufficient-evidence receipts remain terminal in a separate table. Ingestion records source receipt and plan hashes in the Foundation-only SQLite ledger. Adaptive inputs come only from that audited ledger, filtered to the same task/partition. Foreign IDs/unknown tasks/mismatched vectors are rejected. This audits pipeline provenance;it is not a security proof against deliberately fabricated measurements.\n\nMeasured MoE source copied once from the existing physical-only all-five-profile INI in ARGOS-Unseen-Preflight/configs/unseen_moe_v1/workloads/UNSEEN_MOE_U5_DPMOQ_qos_equal.ini. No simulator outcomes from that worktree were read for Foundation acquisition.')
    text('FOUNDATION_DATASET_V1_DESIGN.md',DESIGN)
    testlog=ARTIFACTS/'foundation_v1_test_results.txt'
    text('PRE_EXECUTION_AUDIT.md','# Foundation V1 pre-execution audit\n\n**NO FOUNDATION SCIENTIFIC DATASET SIMULATIONS HAVE BEEN STARTED.**\n\n'+table(['Gate','Status'],[['400 synthetic profiles;300/50/50 profile split','PASS'],['500 workloads;400/50/50 workload split','PASS'],['Hardware separation and grounded80 (48P100/32A100)','PASS'],['J and QoS marginal distributions','PASS'],['500 INIs through real FlexDC JobProfileReader','PASS'],['2000 canonical N/U contexts','PASS'],['300000 immutable joint Phase-A rows','PASS'],['160000 adaptive and40000 repeats expected;coordinates not fabricated','PASS'],['Dynamic J2..6 weights/coupled P/R/simplex','PASS'],['Canonical cost and ISO inheritance','PASS'],['No old V3 observations or model-based acquisition','PASS'],['Recovery/sidecar/request-applied synthetic tests','See test log'],['Actual SCC/simulator execution','NOT RUN;intentionally forbidden']])+'\n\nAll runtime behavior remains owned by the existing simulator/extraction path. One new Foundation-only extraction hook records post-simulation evidence and applied inputs;non-Foundation calls remain unchanged. No real smoke was authorized or run. Production behavior is statically and synthetically checked, not certified by execution. Immutable file writes use fsync/replace and SQLite FULL transactions;no software can guarantee power-loss recovery against every storage failure. Corrupt checksums stop/require identical-row recovery rather than becoming infeasibility.\n\n'+(testlog.read_text(encoding='utf8') if testlog.exists() else 'Test log pending;package must not be called successful until tests pass.'))
    text('COST_CONTRACT_AUDIT.md','# Cost contract\n\n'+json.dumps(read_json(ARTIFACTS/'canonical_contract.json'),indent=2)+'\n\nOnly cost_function/dr_program constants are imported from SA. No initialization/annealing/boundary/step_size settings enter the planner or execution method. Generated gradient config is the existing gradient base with seven canonical cost/DR overrides;no gradient optimizer is run.')
    text('ISO_EXPERIMENT_CONTRACT_AUDIT.md','# Canonical ISO/experiment contract\n\nN250 and N1000 canonical new-ISO traditional-signal templates are inherited. Only system utilization and random_seed change. Duration3600,idle120,time granularity1 and complete ISO block are preserved. The file SHA is in every context/row manifest. Grid hour16,granularity2,normalizationTrue,randomize startFalse. Primary rows seed20;repeat rows21/22 change coupled workload/runtime randomness under the existing pipeline,not the grid. No separate arrival/runtime factorization is claimed.')
    text('SCC_COMMANDS.md',SCC)
    text('README.md','# Foundation Dataset V1 review\n\n**NO FOUNDATION SCIENTIFIC DATASET SIMULATIONS HAVE BEEN STARTED.**\n\nStart with PRE_EXECUTION_AUDIT.md,then FOUNDATION_DATASET_V1_DESIGN.md and SCRIPT_LINEAGE_REPORT.md. The package contains400 synthetic profiles,13 separate measured training anchors,500 workload INIs,2000 contexts,and the full compressed300000-row Phase-A plan plus preview. Waves/repeats are protocol definitions only,not fabricated coordinates. All test output is synthetic/planning-only. The final corpus will be500000 NEW simulator observations only after separate reviewed authorization on SCC.\n\nfoundation_v1_variable_J_domain_audit.csv and canonical_contract.json hold exact derived domains/config provenance. Source diffs include new files and the single Foundation-only observational hook. Current changes are uncommitted/unpushed by explicit instruction. FILE_HASHES.sha256 covers packaged source/manifests/data. No environment files or old simulation data are included. See SCC_COMMANDS.md for staged dry-run/execution/recovery templates;do not execute the campaign during review.')
    write_json(ARTIFACTS/'repository_state.json',{'root':str(ROOT),'initial_branch':'main','initial_head':commit,'working_branch':branch,'remote':git('remote','get-url','origin'),'status':git('status','--short'),'committed':False,'pushed':False},False)
    diff=git('diff','--','src/peacsim/am_data_extraction_wizard.py')
    import difflib
    for p in source_files:
        relative=str(p.relative_to(ROOT)).replace('\\','/')
        if git('ls-files','--',relative):continue
        diff+='\n'+''.join(difflib.unified_diff([],p.read_text(encoding='utf8').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+relative))
    text('foundation_v1_review.patch',diff)
    from foundation_v1.patch_review import patch_docs
    patch_docs()
    from foundation_v1.release import write_operational_docs
    write_operational_docs()
    # Include the frozen canonical source/input bytes too, so deploying the ZIP
    # avoids Git platform newline conversion invalidating strict byte hashes.
    paths=[*source_files,*[ROOT/n for n in provenance if (ROOT/n).exists()],*[p for p in ARTIFACTS.iterdir() if p.is_file() and p.suffix!='.zip']]
    paths += [p for p in (ARTIFACTS/'patch_v1').rglob('*') if p.is_file() and 'before_source' not in p.parts]
    paths=[p for p in paths if p.name!='FILE_HASHES.sha256' and '.tmp.' not in p.name]
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):file_hash(p) for p in paths if not any(x.startswith('.env') for x in p.parts)}
    text('FILE_HASHES.sha256','\n'.join(f'{v}  {k}' for k,v in sorted(hashes.items())))
    paths.append(ARTIFACTS/'FILE_HASHES.sha256')
    archive=ROOT/'FOUNDATION_DATASET_V1_PREEXECUTION_REVIEW_PATCHED.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(paths)):
            if p.is_file() and not any(x.startswith('.env') for x in p.parts):z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(f'PACKAGED {archive} ({archive.stat().st_size/2**20:.2f}MiB);zero simulations/no commit/no push',flush=True)


DESIGN='''# Foundation Dataset V1 frozen collection design

One shared raw/clean simulator corpus supports later Meta V4 and FlexPFN. No model training, normalized model tensors or PFN episodes are built here. Future K=0,1,2,4,8,16,32 can sample the same task corpus;architecture changes do not require a second simulator campaign.

500 workloads:400 TRAIN/50 DEVELOPMENT/50 LOCKED_SYNTHETIC_TEST. Each has N250/N1000 × U.6/.8, giving2000 tasks. Each task has150 static unique +4×20 measured-adaptive unique +20 repeated-realization rows=250. Final total500000,partition rows400000/50000/50000. Only300000 Phase-A coordinates are generated now.

400 latent-convex synthetic profiles:240 P100 (120 training-like/120 inference-like),160 A100 inference-only. Synthetic split300/50/50.13 real physical anchors are allowed directly only in TRAIN;future Can profiles remain untouched. Synthetic identity splits freeze before compositions. Workload split is explicit in manifests,not inferred from names.

Profile modes50/30/20 use deterministic largest remainder within each compatible hardware/class/size subgroup. Latent coordinates log Pmin,log(Pmax/Pmin),log Tmin,log(Tmax/Tmin). Positive anchor barycentric weights stay inside measured convex envelopes;no raw independent noise/extrapolation. Current anchors are all job_size1;new incompatible size groups stop for explicit allocation. A100 hardware attribution is frozen for Foundation V1.

Workloads use deterministic candidate-composition maximin descriptors, measured-envelope descriptor scaling and reuse penalties.80 train workloads are grounded (48P100/32A100),at least ceil(J/2) real slots;the remainder and all dev/test workloads are synthetic-only. P100 train class counts60 training-only/60 inference-only/120 mixed. Mixed type counts are balanced for J≥4. QoS counts and J marginals are exact and independently audited. Heterogeneity category labels are construction goals,not measured simulator bottleneck claims.

Legal geometry reuses current exact-plan P/R helpers and variable-J weight bounds. J2 effective [.3,.7] (requested upper .9 narrowed by simplex),J3 [.2,.6],J4 [.15,.45],J5 [.12,.36],J6 [.1,.3],at N250/N1000. All candidates validate sum1,finite values,coupled reserve R≤.6Pbar,R≥.01,Pbar∈[.9Pmin,Pmax],Pbar+R≤1.2Pmax. Conditional simplex Sobol mapping never clip-renormalizes. Distances use normalized P,conditional normalized R and J−1 Helmert weight contrasts as three RMS metric blocks. Local convex steps are shrunk until the actual normalized radius passes.

Per-task static counts:48 JOINT_SOBOL_INTERIOR,30 STRUCTURED_PR (6×5 P/R with varied legal bank weights),24 PHYSICAL_PR_BOUNDARY (edges and legal corners),24 WEIGHT_SIMPLEX_BOUNDARY,16 CROSS_EXTREME,8 EQUAL_WEIGHT_REFERENCE. This is a joint150-point design,not a P/R×weight Cartesian expansion.

Four adaptive waves use radii .12/.08/.05/.03 and quotas6 local/6 measured boundary/4 bridges/4 global. No predictor is used. Rank actual Vmax,Vsum before objective. Local proposals use up to3 distinct anchors (two points each before additional fallback);feasible anchors coexist with low-violation alternatives. Boundary pairs straddle joint/tracking/QoS outcomes. Bridges use exclusive tracking-safe and QoS-safe anchors with .5,.25,.75,.375 fractions. Global maximin uses4096 legal Sobol candidates against completed/pending/already chosen points;reserve4 global first,then adaptive allocation. Fallback bridge→boundary→local→global,boundary→local→global,local→global. Every wave has20 novel legal geometries or fails.

Only valid sufficient actual current-task observations from the Foundation ledger can generate waves. Each wave records source/input hashes,counts,fallback,parentage and novelty;its coordinates are immutable. No locked-task results select train-task points. A technical failure is not scientific infeasibility.

After230 unique seed20 observations,select10 distinct measured base geometries:4 closest joint/constraint boundary,2 feasible near-boundary if available,one tracking-safe/QoS-fail and one opposite transition,2 equal/reference points;unavailable categories use deterministic measured maximin fill. Avoid selecting an obvious catastrophic point (Vmax≥4,at least5 times one constraint limit) when an unselected Vmax<4 observation exists. Maximin fallback first uses that more informative pool when it can fill the remaining slots;all-catastrophic tasks still choose10 real measured points and are not fabricated as feasible. New seeds21/22 yield20 repeat rows. All three seeds share Base_Geometry_ID;later context/query sampling must keep these groups together unless explicitly testing scenario prediction.

Execution remains existing parallel-combo→exact extraction wizard→unmodified FlexDC,normal lowercase-module AQA,node-count controlTrue. Foundation orchestration creates reproducible missing-row plans,uses bounded old-runner workers,recovers per-row atomic evidence,stores transactional checksum-protected observations and never reruns an already audited valid row. Interrupted/retry attempts retain the same scientific IDs. Locks prevent duplicate orchestrators;independent SCC chunk jobs should not share one active ledger/orchestrator concurrently. There is no scheduler-specific submission assumption.

Only a guarded extraction-sidecar hook was added to the wizard. It records the actual scheduling weights/power/N/U/seed/grid/config identity,original result+diagnostic rows and counts of the exact population used by the current QoS estimator. It does not recompute/change Pj or simulation. Unobserved Pj0 is insufficient evidence,never a pass. Requested power rounding is int(round(kW/server×1000×N)),≤.5W cluster error;weights compare within1e-9. Numerical feasibility and evidence validity are separate.

The final exporter requires500000 valid rows,250 per task and correct20/21/22 grouping;it writes canonical observation/diagnostic/row/repeat manifests and split/sampling/feasibility/provenance summaries. No-old-V3-data lineage is checked through F1 identities and Foundation-only receipts. Plans record the planning base Git SHA plus a separately reviewed source-tree fingerprint,not a fictitious commit of these uncommitted changes. Runtime sidecars additionally record the actual execution Git SHA after the later reviewed commit;this keeps pre-execution and execution provenance distinct. Default orchestration is dry-run;scientific execution requires explicit --execute --resume after review.
'''

SCC='''# SCC / staged execution templates — DO NOT RUN during preparation

Use an activated Python environment with existing repository requirements. Run from this FlexDC root. No scheduler convention exists in the inspected repo;these are commands for an allocation,not invented Slurm/SGE directives. The declared --max-workers is the global combo-runner process bound within one orchestrator. Do not launch multiple orchestrators against one database.

Deployment after review/commit/push:clone the reviewed repository,then extract the frozen review ZIP at that repository root. The ZIP includes exact scientific source/config/ISO/manifest/plan bytes;this avoids Windows/Linux Git newline conversion invalidating strict hashes. Do NOT regenerate the frozen scientific manifests or static coordinates on workers or merely to update the planning-base Git SHA. The runtime receipt separately records the later actual execution Git SHA. Run the dry-run command first to verify all file hashes. Source/config disagreement requires review,a new version,or restoring the approved bundle;never silently update the expected hashes.

Original local preparation only (already completed;do not rerun these on SCC after freezing):
```
python -B am_generate_flexdc_foundation_profiles_v1.py
python -B am_generate_flexdc_foundation_workloads_v1.py
python -B am_generate_flexdc_foundation_plan_v1.py
python -B am_vet_flexdc_foundation_dataset_v1.py --preflight
python -B am_run_flexdc_foundation_campaign_v1.py --stage phaseA --max-workers 10
```

Only after review/commit/push and a separate execution authorization:
```
python -B am_run_flexdc_foundation_campaign_v1.py --stage phaseA --max-workers 10 --execute --resume
python -B am_generate_flexdc_foundation_adaptive_plan_v1.py --wave 1 --database foundation_v1_artifacts/observations.sqlite
python -B am_run_flexdc_foundation_campaign_v1.py --stage wave01 --max-workers 10 --execute --resume
```
Repeat generation/execution for waves2,3,4 only after auditing the previous stage. Then generate --repeats,execute --stage repeat,and export with am_vet_flexdc_foundation_dataset_v1.py --export. Primary seed20/repeat21/22 are frozen coordinates/realizations in plans;workers do not resample scientific coordinates.

Use --partition TRAIN/DEVELOPMENT/LOCKED_SYNTHETIC_TEST or --task <Task_ID> and --chunk-index i --chunk-count n for selected reproducible shards. Membership is assigned from the immutable selected plan BEFORE removing completed IDs,so resume cannot shift shard ownership. Partial-stage use still requires all230 unique rows for a task before its repeats.

The legacy runner uses6 stable chunks per context,about25 rows per Phase-A worker. Foundation receipts allow recovering individual complete rows after a partially written worker CSV. Recovery never treats bad/empty JSON or missing evidence as feasible. Check campaign_status.json and last_ingest.json. Do not delete old attempts or change source/configs under a frozen release. Resume requires exact identity;scientific mismatches stop for review. Source/run SHA provenance is separate from the current uncommitted Git base identity.
'''


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',action='store_true');a=p.parse_args()
    if a.package:docs_and_package()
    else:preflight()
