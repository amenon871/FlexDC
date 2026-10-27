# Status regressions — PASS

* Success t=10 then invalidation t=20: not completed.
* Later stale success t=10: remains invalidated.
* Repair success t=30: completed.
* Invalidated rows contribute neither completed counts nor throughput counts; pending rises accordingly.
* Delete/rebuild status cache: identical final projection.
* Legacy cache containing misclassified invalidation: discarded/replayed safely.

See the complete collector test log; no skipped tests or warnings.
