# Script lineage

Root: `C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main`. Initial branch main, HEAD `525dc684d73ab0c6f6c479f5b54811ddf02f1221`. Fetch succeeded;initial HEAD and origin/main matched;initial worktree clean. No pull/reset/merge/checkout over existing work. Current branch `foundation-dataset-v1-prep`;no commit/push.

The nearby FlexDC_V5_Eval and its flexdc_workspace are non-Git model/data workspaces, not newer FlexDC research checkouts. The requested checkout is authoritative. The extraction wizard is under src/peacsim.

|File|origin/main SHA256|local SHA256|Basis/action|
|---|---|---|---|
|am_generate_flexdc_sweep_plan_v3.py|0b111ab803c755d50ba518e9e963bc1f863f642bcb6914080660ce870b29489c|0b111ab803c755d50ba518e9e963bc1f863f642bcb6914080660ce870b29489c|BYTE_IDENTICAL to fetched origin/main|
|am_generate_flexdc_exact_sweep_plan.py|5a05476f7fee45bb85071668899bdf1d4eaa85c5ef0d67fa0ba91a157f9c7625|5a05476f7fee45bb85071668899bdf1d4eaa85c5ef0d67fa0ba91a157f9c7625|BYTE_IDENTICAL to fetched origin/main|
|am_run_flexdc_dataset_parallel_combo.py|84f0453e8fb020dcbb4ada5564722e749a977cc0e60d45535756652d5d638034|84f0453e8fb020dcbb4ada5564722e749a977cc0e60d45535756652d5d638034|BYTE_IDENTICAL to fetched origin/main|
|am_vet_flexdc_sweep_dataset_v3.py|b43c1a2c81b79c1c445e24cbac7e73942bb7eba0345ca7722421495360b863fc|b43c1a2c81b79c1c445e24cbac7e73942bb7eba0345ca7722421495360b863fc|BYTE_IDENTICAL to fetched origin/main|
|am_generate_paper_iso_experiment_configs.py|776d47a0684e65827183f2727a9636ac552405964fa0868529fb2ff29862b6a2|776d47a0684e65827183f2727a9636ac552405964fa0868529fb2ff29862b6a2|BYTE_IDENTICAL to fetched origin/main|
|src/peacsim/am_data_extraction_wizard.py|9818a653eb596288ddb6cfe9bab09923614b68fccc0912b339242375e602dc42|4c689b136d2f533694ebb6a0971faccdfbc2fb2543ed9badb6da0af8649197aa|Foundation-only evidence hook added after clean audit|
|am_combine_csv_v3.py|7eb63fb731c2144df40081d8e2b2704dc6ddb3d7715147192410d45883cccbef|e36490945df86660aa0b3130abba924ad1569a61f665510658079b3e743638cf|Same source text;working-copy CRLF/LF differs|
|prepare_flexdc_training_dataset.py|1d455da9e809fff6e8c832432d57603c367c719bc3a2e2f71f5050aac5f7faee|1d455da9e809fff6e8c832432d57603c367c719bc3a2e2f71f5050aac5f7faee|BYTE_IDENTICAL to fetched origin/main|
|am_generate_flexdc_j2_pair_workloads.py|d6be81a1f007fdcf8d1546dd93214f45998c44b4c4cb7f1f489ee284b4ef5471|d6be81a1f007fdcf8d1546dd93214f45998c44b4c4cb7f1f489ee284b4ef5471|BYTE_IDENTICAL to fetched origin/main|
|flexdc_generic_sources/flexdc_profile_split.py|60903ab851d320d5f89ceb1c1ac01d51287728534e31ecaa0762492e00c4ce6b|60903ab851d320d5f89ceb1c1ac01d51287728534e31ecaa0762492e00c4ce6b|BYTE_IDENTICAL to fetched origin/main|

Existing exact-plan domain/cost/weight/boundary helpers and combo-runner are reused. Existing combine/training prep remain unchanged and are NOT used to import old observations. New Foundation export owns only Foundation row joins.
