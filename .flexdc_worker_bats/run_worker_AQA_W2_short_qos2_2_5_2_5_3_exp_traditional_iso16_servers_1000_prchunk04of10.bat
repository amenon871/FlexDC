@echo off
title FlexDC AQA W2_short_qos2_2_5_2_5_3 exp_traditional_iso16_servers_1000 prchunk04of10
cd /d "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main"
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\src;%PYTHONPATH%
echo Starting FlexDC worker: AQA W2_short_qos2_2_5_2_5_3 exp_traditional_iso16_servers_1000 prchunk04of10
echo Repo root: C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main
echo Command:
echo "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\venv\Scripts\python.exe" -u "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\am_run_flexdc_dataset_parallel_combo.py" --worker-experiment ../../configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_1000.ini --worker-workload ../../configs/workload/W2-short-qos2-2.5-2.5-3.ini --worker-pr-chunk-index 3 --gradient-config ../../configs/gradient_descent/gradient_descent.ini --cluster-config ../../configs/cluster/cluster.ini --policy-name AQA --output-prefix traditional_iso16_fullpilot --pr-num-chunks 10 --utilization-values 0.6,0.8 --auto-workload-pr-sweep true --pbar-points 10 --r-points-per-pbar 10 --r-lower-kw-per-server 0.01 --pbar-lower-factor 0.9 --pbar-upper-factor 1.0 --pr-upper-factor 1.2
echo.
"C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\venv\Scripts\python.exe" -u "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\am_run_flexdc_dataset_parallel_combo.py" --worker-experiment ../../configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_1000.ini --worker-workload ../../configs/workload/W2-short-qos2-2.5-2.5-3.ini --worker-pr-chunk-index 3 --gradient-config ../../configs/gradient_descent/gradient_descent.ini --cluster-config ../../configs/cluster/cluster.ini --policy-name AQA --output-prefix traditional_iso16_fullpilot --pr-num-chunks 10 --utilization-values 0.6,0.8 --auto-workload-pr-sweep true --pbar-points 10 --r-points-per-pbar 10 --r-lower-kw-per-server 0.01 --pbar-lower-factor 0.9 --pbar-upper-factor 1.0 --pr-upper-factor 1.2
echo.
echo Worker finished: AQA W2_short_qos2_2_5_2_5_3 exp_traditional_iso16_servers_1000 prchunk04of10
echo Press any key to close this window.
pause > nul
