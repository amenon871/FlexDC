# Rangefind64 visualization outputs

Recommended order to inspect:

1. `00_pr_grid_coverage_counts.png`
   Confirms physical P/R grid coverage and skipped invalid pair.

2. `03_line_vs_r_by_workload_simulator_tracking_cost_median.png`
   Best plot for the tracking-cost vs R question.

3. `04_line_vs_p_by_workload_simulator_power_cost_median.png`
   Shows simulator power-cost behavior versus Pbar.

4. `05_qos_vs_p_by_utilization_qos_delay_probability_sum_median.png`
   Shows QoS behavior versus Pbar by utilization.

5. `02_surface_all_simulator_rsr_total_cost_median.png`
   Main combined 3D surface over physical P/R.

There is no `QoS_Cost` in the raw combined CSV because the simulator returns raw RSR monetary components as power cost and tracking cost. QoS is computed separately from delay probabilities. To avoid repeating the earlier mistake, the raw CSV stores QoS observables (`QoS_Delay_Probability_Sum`, `QoS_Violation_Ratio`) and the diagnostics CSV stores residual/SoftPlus-style QoS terms.
