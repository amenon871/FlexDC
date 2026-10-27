# Recorded test timing note

All 356 collector tests passed, with zero failures, skips or pytest warnings. The full suite recorded 15395.32 seconds (4h16m35s) elapsed. The existing test `test_full_production_state_machine_real_planners_synthetic_simulator` accounts for 15103.802 seconds of that total. It uses real scientific planners with a fake simulator boundary; it does not launch FlexDC.

The saved timing data do not establish the cause of this outlier (for example, execution versus external suspension). It is reported without inventing a cause and was not hidden or replaced by another run. This duration is not a collection-throughput or SCC recovery benchmark.

The separately measured multi-stage recovery fixture reached its active stage in 0.1537915 seconds, with zero prior-stage receipt opens or strong-audit calls. SCC GPFS timing remains unverified. Review the separate performance JSONs for precisely scoped measurements.
