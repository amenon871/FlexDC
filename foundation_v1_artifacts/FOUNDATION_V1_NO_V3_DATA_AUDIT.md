# No old V3 observation data

PASS for this pre-execution release:zero simulator observations exist in Foundation planning. Profiles read ONLY three physical INIs (two P100 sources and the copied measured MoE INI);existing code/configs are permitted reuse. No old results/diagnostics CSV, V3 checkpoint/prediction, ARGOS/FS/OC output or learned acquisition enters generation.

At execution, every terminal receipt must possess an F1_ row ID recomputed from an immutable Foundation task/geometry/seed, an atomic evidence receipt with body checksum and exact applied/config identity. Only sufficient per-job evidence enters the valid-label table; technically complete insufficient-evidence receipts remain terminal in a separate table. Ingestion records source receipt and plan hashes in the Foundation-only SQLite ledger. Adaptive inputs come only from that audited ledger, filtered to the same task/partition. Foreign IDs/unknown tasks/mismatched vectors are rejected. This audits pipeline provenance;it is not a security proof against deliberately fabricated measurements.

Measured MoE source copied once from the existing physical-only all-five-profile INI in ARGOS-Unseen-Preflight/configs/unseen_moe_v1/workloads/UNSEEN_MOE_U5_DPMOQ_qos_equal.ini. No simulator outcomes from that worktree were read for Foundation acquisition.
