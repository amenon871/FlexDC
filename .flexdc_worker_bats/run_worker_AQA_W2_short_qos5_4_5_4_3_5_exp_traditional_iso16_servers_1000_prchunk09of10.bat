@echo off
title FlexDC AQA W2_short_qos5_4_5_4_3_5 exp_traditional_iso16_servers_1000 prchunk09of10
cd /d "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main"
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\src;%PYTHONPATH%
echo Starting FlexDC worker: AQA W2_short_qos5_4_5_4_3_5 exp_traditional_iso16_servers_1000 prchunk09of10
echo Repo root: C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main
echo Command:
echo "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\venv\Scripts\python.exe" -u "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\am_run_flexdc_dataset_parallel_combo.py" --worker-experiment ../../configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_1000.ini --worker-workload ../../configs/workload/W2-short-qos5_4.5_4_3.5.ini --worker-pr-chunk-index 8 --gradient-config ../../configs/gradient_descent/gradient_descent.ini --cluster-config ../../configs/cluster/cluster.ini --policy-name AQA --output-prefix w2_dense_local_v1 --pr-num-chunks 10 --node-count-control true --pbar-kw-per-server-values 0.464,0.466,0.468,0.470,0.472,0.472128,0.474,0.476,0.478,0.480 --r-kw-per-server-values 0.075,0.085,0.095,0.102206,0.110,0.120,0.130,0.140 --utilization-values 0.60 --auto-workload-pr-sweep false --pbar-lower-factor 0.9 --pbar-upper-factor 1.0 --pr-upper-factor 1.2 --weight-file w2_local_70_weights.csv
echo.
"C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\venv\Scripts\python.exe" -u "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\am_run_flexdc_dataset_parallel_combo.py" --worker-experiment ../../configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_1000.ini --worker-workload ../../configs/workload/W2-short-qos5_4.5_4_3.5.ini --worker-pr-chunk-index 8 --gradient-config ../../configs/gradient_descent/gradient_descent.ini --cluster-config ../../configs/cluster/cluster.ini --policy-name AQA --output-prefix w2_dense_local_v1 --pr-num-chunks 10 --node-count-control true --pbar-kw-per-server-values 0.464,0.466,0.468,0.470,0.472,0.472128,0.474,0.476,0.478,0.480 --r-kw-per-server-values 0.075,0.085,0.095,0.102206,0.110,0.120,0.130,0.140 --utilization-values 0.60 --auto-workload-pr-sweep false --pbar-lower-factor 0.9 --pbar-upper-factor 1.0 --pr-upper-factor 1.2 --weight-file w2_local_70_weights.csv
echo.
echo Worker finished: AQA W2_short_qos5_4_5_4_3_5 exp_traditional_iso16_servers_1000 prchunk09of10
echo Press any key to close this window.
pause > nul
