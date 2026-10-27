# Disposable fast recovery index

New module: foundation_v1/recovery_index.py. Per-stage path: <area>/recovery_index.sqlite. One parent writer, no worker DB writers and no new dependencies.

Tables: metadata(key,value) seals index schema, partition, stage, immutable plan SHA and receipt schema; completed(id,timestamp) contains only IDs/timestamps; journal_offsets(name,pos,dev,ino,head,tail,size,mtime) contains byte offsets and small file-identity/boundary fingerprints. No simulator metrics are stored in the index.

Routine recovery validates quick_check, identities and journal replacement/truncation checks; loads indexed IDs; reads only appended complete journal lines; validates each newly discovered receipt through the unchanged receipt loader; checks pending pointers for unindexed IDs; commits IDs and journal offsets in one transaction. Partial trailing lines remain unconsumed. New journal files start at zero.

Index missing, corrupt, incompatible or journal rewritten: full authoritative scan, recover live-status snapshot, rebuild derived index. The first deployment into the existing no-index campaign can still incur a full scan. Subsequent resumes avoid historical receipt opens. Deleting this disposable index never loses scientific completion.

After already validating receipts during collection, the parent queues IDs for a batch of 100 or periodic 5-second flush; synchronous=NORMAL applies only to this disposable index. Atomic receipts and fsynced journals retain their existing durability. Index write debt cannot revoke scientific completion. Broken cache metadata triggers a rebuild on next recovery.

Pending metadata alone is not success. Receipt published without journal is recovered from pending. A journal claim with no valid receipt is not accepted. Indexed receipt deletion/corruption is detected by the unchanged strong stage audit; audit repair appends an invalidation event so only the affected row is recollected. No indexed count certifies stage advancement.

The existing full campaign still performs strong certification at stage boundaries (including completed stages revisited by its existing orchestration). This patch accelerates ordinary collector reconciliation, not those authoritative audits or immutable-plan parsing. Their time is outside the indexed-recovery microbenchmark; no end-to-end SCC startup-time guarantee is claimed.

Maintenance, after review/deployment and while the campaign lock is free:

```
python -B -m foundation_v1.recovery_index --plan <IMMUTABLE_STAGE_PLAN> --area <STAGE_AREA> --rebuild
```

This command takes the existing campaign lock, scans receipts and rebuilds only the derived cache. It never launches workers, changes receipts or generates coordinates. No maintenance or collection was run against production here.

SCC GPFS <5-minute target remains to be verified after deployment. Future measurements should separate recovery from strong certification and include ~117k, 250k and 450k completed rows.
