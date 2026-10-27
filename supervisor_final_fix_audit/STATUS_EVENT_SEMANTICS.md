# Final fix 3 — PASS

Disposable status SQLite events now store (stage,id,event time,valid_complete). SQL upsert changes a row only when the incoming timestamp is >= its stored timestamp. Success sets valid_complete=1; invalidation sets 0; repaired later success sets 1 again. Old success events and stale pending receipts cannot resurrect a newer invalidation. Strong-audit missing/quarantine reports enter the same timestamp-ordered projection.

Completed, pending, overall/partition completion, last-hour count and last-minute rate include only currently valid_complete=1 records. The legacy three-column status cache is rebuilt from durable events rather than retaining already-misinterpreted invalidations. Deleting this cache rebuilds the same status. Collection does not use status SQLite as authority. No production status cache was touched.
