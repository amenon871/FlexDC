# Final adversarial cases

| Case | Expected and observed behavior | Result |
|---|---|---|
|Dirty training directory|Refuse before opening log or overwriting state|PASS|
|Certified Phase A|Skip recovery/certification; zero historical receipt opens|PASS|
|Certified Phase A + W1, partial W2|Only W2 executes; earlier-stage audit calls zero|PASS|
|Wrong certificate plan hash|Reject and strongly re-certify|PASS|
|Certificate after invalidation|Revoke; repair only affected row; create new certificate|PASS|
|Status success then invalidation|Completed/count/rate removed|PASS|
|Status invalidation then repair success|Completion restored at newer timestamp|PASS|
|Out-of-order stale status event|Cannot resurrect newer invalidation|PASS|
|Operational source transition|Original source remains explicit; no transition means changed-source resume refused|PASS|
|Unpacked package|One command, no PYTHONPATH edits|PASS|
