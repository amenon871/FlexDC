# Certified-stage tests — PASS

Tests in tests/test_foundation_v1_final_fixes.py cover: strong-PASS-only creation; missing/corrupt/missing-field certificates; wrong plan/source identity; modified audit artifact; missing/corrupt audit DB; changed recovery generation; new journal entry; receipt deletion; post-certificate invalidation; row-local repair and replacement certificate; actual run_full dispatch with certified Phase A + W1 and partially complete W2.

Synthetic benchmark: 3 checks, 2 stages skipped, 0 historical receipt opens and 0 prior-stage strong audits. Only ['wave02'] executed. Startup to active stage 0.153791 s; active-stage receipt opens 148 include fake execution/acceptance and its real strong audit. Entire synthetic resume 13.053836 s.

SCC GPFS <5-minute target remains unverified until approved deployment. No real simulator was launched.
