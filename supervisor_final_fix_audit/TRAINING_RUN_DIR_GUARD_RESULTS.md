# Final fix 1 — PASS

Shared helper: foundation_successor_models/flexmodels/training.py prepare_run_directory, called by the common loop before opening training.log or writing any metadata/checkpoint. Missing/empty paths are allowed. Any nonempty path is refused without --resume, including all checkpoint/history/config/DEVELOPMENT-bank artifacts and unknown files. No cleanup, rename, overwrite or dangerous force option exists. Existing explicit auto-resume semantics are retained.

Tests: tests/test_training_run_directory.py. All three trainer kinds are checked for 15 kinds of existing state; nonexistent/empty/different paths pass; actual existing runs resume only when requested; refusal preserves checkpoint bytes. The old prior test fixture now uses a fresh temporary directory each session instead of silently reusing .validation_runs/unit_prior.

Complete training suite: 171 passed; no failures/skips/warnings; 54.389 s pytest runtime.
Clean-unpack one-command suite: 171 passed; 49.767 s; PYTHONPATH unset.
