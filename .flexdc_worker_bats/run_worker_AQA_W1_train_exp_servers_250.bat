@echo off
title FlexDC AQA W1_train exp_servers_250
cd /d "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main"
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\src;%PYTHONPATH%
echo Starting FlexDC worker: AQA W1_train exp_servers_250
echo Repo root: C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main
echo Command:
echo "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\venv\Scripts\python.exe" -u "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\am_run_flexdc_dataset_parallel_combo.py" --worker-experiment ../../configs/experiment/generated_server_counts/exp_servers_250.ini --worker-workload ../../configs/workload/W1-train.ini --gradient-config ../../configs/gradient_descent/gradient_descent.ini --cluster-config ../../configs/cluster/cluster.ini --policy-name AQA --output-prefix final_aqa_s250_1000_u060_080 --utilization-values 0.6,0.8 --auto-workload-pr-sweep true
echo.
"C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\venv\Scripts\python.exe" -u "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\am_run_flexdc_dataset_parallel_combo.py" --worker-experiment ../../configs/experiment/generated_server_counts/exp_servers_250.ini --worker-workload ../../configs/workload/W1-train.ini --gradient-config ../../configs/gradient_descent/gradient_descent.ini --cluster-config ../../configs/cluster/cluster.ini --policy-name AQA --output-prefix final_aqa_s250_1000_u060_080 --utilization-values 0.6,0.8 --auto-workload-pr-sweep true
echo.
echo Worker finished: AQA W1_train exp_servers_250
echo Press any key to close this window.
pause > nul
