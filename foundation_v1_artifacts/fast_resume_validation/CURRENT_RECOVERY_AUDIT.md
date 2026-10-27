# Existing collector recovery audit

Authoritative source at base 0190fe60d53cddb6afbaffad9a67b82e50ccf0de: foundation_v1/collector_receipts.py publish/load/completed, simple_collector.py collect, and simple_campaign.py certify/execute_stage.

Publication order is unchanged: validate compact output/evidence and applied identity; fsync relevant directories; write events/<worker>.pending.json atomically; publish checksum-bearing receipts/<prefix>/<Plan_Row_ID>.json atomically; append/flush/fsync events/<worker>.jsonl. The pending pointer may remain and is deduplicated.

Before this patch, collect called receipts.completed(area,by_id), reopening every historical receipt, then rewrote events/recovered.jsonl. Parent acceptance also validates receipts and appends parent.jsonl. live_status has a separate disposable status cache; it is not completion authority.

Strong certification remains in simple_campaign.certify: it loads authoritative completion/evidence, verifies compact CSVs and requested/applied scientific identity, reconstructs row identity/objective/evidence, updates the audited observation ledger, and checks cleanup debt. Invalid/missing rows enter bounded repair; insufficient evidence halts safely and is not resimulated identically.

No production receipt, journal, worker tree, SCC path or active campaign was accessed in this task.
