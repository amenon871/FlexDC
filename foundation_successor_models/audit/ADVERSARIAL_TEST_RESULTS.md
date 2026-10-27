# Required adversarial cases

All cases below PASS; see tests/test_supervisor_corrections.py and PATCH_TEST_RESULTS.xml.

| Case | Result |
|---|---|
| External DEVELOPMENT bank changed after checkpoint | PASS: frozen run-local bytes remain in use, both successors |
| Run-local DEVELOPMENT bank changed | PASS: resume refuses with SHA mismatch |
| Runnable source changed | PASS: normal resume refuses and shows both hashes |
| Audit/dist/runtime source-like noise added | PASS: source identity unchanged |
| --stop-after-steps | PASS: latest exists; final absent for all three |
| Legitimate tiny completion | PASS: final exists and terminal reason recorded |
| Slow validation with insufficient runtime | PASS: validation never entered; safe latest exists |
| Benchmark limit | PASS: latest exists, final absent |
| J2..6 geometry parity | PASS: canonical effective bounds; atol 1e-14 |
| J3/J5 models | PASS: Baseline and both successors |
| Evidence counts / padded zeros | PASS |
| Exact source quotas and legal fallback families | PASS |
| Wrong/missing master Phase-A SHA | PASS: rejected |

All fixtures are synthetic; no FlexDC was launched.
