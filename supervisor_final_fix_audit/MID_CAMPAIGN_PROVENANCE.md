# Final fix 4 — PASS; not deployed

The original foundation_v1_artifacts/preexecution_release.json has been restored exactly to base HEAD bytes. It truthfully remains the historical pre-execution artifact; it has not been relabeled as a campaign-wide new release.

Original production source identity: 75109d5f0953dcf60a6e2761db1f838d97dda4554d9937a06436a281fe25f0d5

New local operational patch identity: 6bfc4c802deb2088e9df723260a9ae0602865e8f13b15309cf0e7b19e7ae0348

Separate immutable record: foundation_v1_artifacts/operational_patch_release_resume_v3.json. It records scientific_simulations_started=true and operational_patch_after_campaign_start=true based on the user's reported production state, while scientific_master_changed=false. All actual deployment timestamp/count/active-stage fields remain null; deployed_to_scc=false. The previous local development-review identity is separately labeled never deployed. It is not confused with the original production release.

verify_release validates canonical master bytes first, then the original release bytes/identity and all 626 effective guarded paths through the operational record. No old protected path was removed. A changed/missing patch seal refuses verification instead of silently bypassing the guard.

Deployment-time helper: python -B -m foundation_v1.operational_release transition. It validates the reviewed source, takes the existing campaign lock, consumes an operator-captured pre-update status snapshot and publishes a checksum-bearing transition receipt by atomic no-overwrite link. It records actual timestamp, old/new identities, reported completed rows, active partition/stage, master SHA, operator/release ID and snapshot hash. An existing campaign with a changed source refuses run_full until a matching transition receipt exists. Future state preserves the initial source and transition receipt paths.

Only test-fixture transition receipts were created, under temporary roots. No production transition receipt was created.

## Future SCC workflow — only after supervisor approval and allocation ends

1. Let the current allocation end normally; do not stop its workers for this task.
2. Before updating the checkout, capture its status with the old code:
   python -B am_run_flexdc_foundation_campaign_v1.py --status > /path/to/status_before_patch.json
   Preserve the old source identity. If it differs from 75109d5f0953dcf60a6e2761db1f838d97dda4554d9937a06436a281fe25f0d5, stop and review provenance rather than guessing.
3. Install the exact supervisor-approved source/config bytes and operational release record through the approved deployment process. No commit or remote release has been created in this task. Never fresh-start, delete receipts or reseal changed SCC bytes to bypass review.
4. With the same pinned Python3.10/NumPy2.2.6/pandas2.3.3/SciPy1.15.3 environment:

   python -B -m foundation_v1.operational_release verify

5. Write the actual immutable transition receipt:

   python -B -m foundation_v1.operational_release transition --status-snapshot /path/to/status_before_patch.json --old-source-identity 75109d5f0953dcf60a6e2761db1f838d97dda4554d9937a06436a281fe25f0d5 --operator NetherMoon --release-id foundation_resume_v3

6. Missing indexes/certificates rebuild safely. Optional index-only maintenance (no workers) uses python -B -m foundation_v1.recovery_index --plan <immutable-stage-plan> --area <stage-area> --rebuild. First certificate creation still requires the strong audit; do not fabricate certificates.
7. Resume once with the normal command, choosing N for the allocated fat node:

   python -u -B am_run_flexdc_foundation_campaign_v1.py --full-campaign --partition ALL --unlock-locked-test --max-workers <N> --execute --resume

8. Confirm STAGE CERTIFIED skips for previously certified stages, active-stage recovery, worker relaunch, status counts and no row-count regression. SCC startup timing remains to be measured.

None of these deployment commands were executed during local development.
