@echo off
title FlexDC AQA W_train_2app exp_servers_64
cd /d "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main"
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\src;%PYTHONPATH%
echo Starting FlexDC worker: AQA W_train_2app exp_servers_64
echo Repo root: C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main
echo Command:
echo "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\venv\Scripts\python.exe" -u "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\am_run_flexdc_dataset_parallel_combo.py" --worker-experiment ../../configs/experiment/generated_server_counts/exp_servers_64.ini --worker-workload ../../configs/workload/W-train-2app.ini --gradient-config ../../configs/gradient_descent/gradient_descent.ini --cluster-config ../../configs/cluster/cluster.ini --policy-name AQA --output-prefix rangefind64_p020_084_rtargeted --pbar-kw-per-server-values 0.3066666667,0.4133333333,0.52,0.6266666667,0.7333333333,0.84 --r-kw-per-server-values 0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80 --utilization-values 0.5,0.75,0.8,0.85,0.9,0.95
echo.
"C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\venv\Scripts\python.exe" -u "C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\am_run_flexdc_dataset_parallel_combo.py" --worker-experiment ../../configs/experiment/generated_server_counts/exp_servers_64.ini --worker-workload ../../configs/workload/W-train-2app.ini --gradient-config ../../configs/gradient_descent/gradient_descent.ini --cluster-config ../../configs/cluster/cluster.ini --policy-name AQA --output-prefix rangefind64_p020_084_rtargeted --pbar-kw-per-server-values 0.3066666667,0.4133333333,0.52,0.6266666667,0.7333333333,0.84 --r-kw-per-server-values 0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80 --utilization-values 0.5,0.75,0.8,0.85,0.9,0.95
echo.
echo Worker finished: AQA W_train_2app exp_servers_64
echo Press any key to close this window.
pause > nul
