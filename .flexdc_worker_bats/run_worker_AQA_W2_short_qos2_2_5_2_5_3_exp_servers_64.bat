@echo off
title FlexDC AQA W2_short_qos2_2_5_2_5_3 exp_servers_64
cd /d "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main"
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\src;%PYTHONPATH%
echo Starting FlexDC worker: AQA W2_short_qos2_2_5_2_5_3 exp_servers_64
echo Repo root: C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main
echo Command:
echo "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\venv\Scripts\python.exe" -u "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\am_run_flexdc_dataset_parallel_combo.py" --worker-experiment ../../configs/experiment/generated_server_counts/exp_servers_64.ini --worker-workload ../../configs/workload/W2-short-qos2-2.5-2.5-3.ini --gradient-config ../../configs/gradient_descent/gradient_descent.ini --cluster-config ../../configs/cluster/cluster.ini --policy-name AQA --output-prefix targeted_pr_test --utilization-values 0.85 --auto-workload-pr-sweep true --pbar-points 10 --r-points-per-pbar 10 --r-lower-kw-per-server 0.01 --pbar-lower-factor 0.9 --pbar-upper-factor 1.1 --pr-upper-factor 1.3
echo.
"C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\venv\Scripts\python.exe" -u "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\am_run_flexdc_dataset_parallel_combo.py" --worker-experiment ../../configs/experiment/generated_server_counts/exp_servers_64.ini --worker-workload ../../configs/workload/W2-short-qos2-2.5-2.5-3.ini --gradient-config ../../configs/gradient_descent/gradient_descent.ini --cluster-config ../../configs/cluster/cluster.ini --policy-name AQA --output-prefix targeted_pr_test --utilization-values 0.85 --auto-workload-pr-sweep true --pbar-points 10 --r-points-per-pbar 10 --r-lower-kw-per-server 0.01 --pbar-lower-factor 0.9 --pbar-upper-factor 1.1 --pr-upper-factor 1.3
echo.
echo Worker finished: AQA W2_short_qos2_2_5_2_5_3 exp_servers_64
echo Press any key to close this window.
pause > nul
