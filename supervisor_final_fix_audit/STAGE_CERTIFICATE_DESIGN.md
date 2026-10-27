# Final fix 2 — derived stage certification cache

Path: foundation_v1_runs/<partition>/collector_v2/<stage>/stage_certification.json.

Creation occurs only after execute_stage returns strong VALID_COMPLETE and validate_plan succeeds. The existing certify function revokes any old certificate before auditing; its original scientific checks remain. Certificate writes use the existing flush/fsync/atomic rename helper. A small stage_certification_cache row in the existing derived observation database anchors the stage-specific identity without hashing an ever-growing database on every resume. This table is excluded from scientific observation hashes/acquisition; it stores no simulator outputs.

Certificate fields include schema, partition/stage, exact plan path/hash, expected/certified row counts, receipt schema, scientific master SHA, audit artifact path/hash, audit-database path/stage-marker hash, certificate ID, recovery-index generation, journal/pending watermarks, bounded receipt-prefix-directory fingerprints, collector source identity, certification timestamp and checksum.

Validation reads small metadata, exact plan/audit hashes and a stage-marker SQL row. It opens no historical receipts. Journal watermarks include consumed byte length, file identity and small prefix/boundary fingerprints; invalidation/repair appends change them. Recovery rebuild changes a persisted UUID generation. Receipt prefix-directory metadata detects removal and atomic replacement without per-row scans.

A valid earlier-stage certificate causes run_full to skip collector recovery, strong certification, repair and repeated plan parsing for that stage. Missing/corrupt certificates, plan/audit/source/science mismatch, missing/unreadable database marker, changed recovery generation, changed journals or receipt buckets fall back to existing strong certification and bounded repair. Invalidation/quarantine revokes the certificate before later use. New repaired success does not resurrect an old certificate; strong certification must produce a new one.

The receipt contract is immutable atomic publication through controlled pipeline paths. This is not continuous bit-rot monitoring: arbitrary in-place changes that bypass journals and atomic replacement cannot be proved absent by O(1) metadata checks. If external corruption is suspected, revoke that derived certificate (stage_certification.invalidate(area)) and run the normal strong certification/repair before downstream use. Strong audits remain authoritative. No claim is made that the cache replaces receipt validation.

First deployment into a campaign with no certificates requires one strong certification per already completed stage. Subsequent allocations skip those unchanged stages. Missing generation metadata in an existing recovery index is upgraded without a full historical scan. Final export verification remains intact and outside startup-to-active-stage timing.
