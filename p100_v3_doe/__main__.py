"""CLI for the P100 V3 generalization DOE data collection."""

import argparse
import json
from pathlib import Path

from .audit import audit_design, require
from .collection import (
    choose_contexts,
    collect_attempts,
    exclusive,
    export,
    generate_all,
    prepare_b,
    run,
)
from .io import ROOT, load, sha
from .ledger import ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "runs/p100_v3_doe_release",
        help="New isolated DOE directory under this checkout's runs directory",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser(
        "generate",
        help="Freeze and audit all manifests and 244,800 Stage A planned rows",
    )
    actions.add_parser("audit-design")
    for name in ("run", "smoke", "stage-b"):
        command = actions.add_parser(name)
        command.add_argument(
            "--case",
            action="append",
            default=[],
            help="Select a context or workload ID. Repeat as needed.",
        )
        command.add_argument("--model", help="Limit to one frozen model manifest")
        command.add_argument(
            "--partition", choices=("train", "validation", "test"), default="test"
        )
        if name in ("run", "smoke"):
            command.add_argument("--workers", type=int, default=2)
            command.add_argument("--chunks", type=int, default=4)
            command.add_argument(
                "--dry-run",
                action="store_true",
                help="Use the runner's exact-plan dry run",
            )
            if name == "run":
                command.add_argument(
                    "--batch-contexts",
                    type=int,
                    default=24,
                    help="Bound transient retention and harvest frequency",
                )
                command.add_argument("--stage", choices=("A", "B"), required=True)
                command.add_argument(
                    "--execute",
                    action="store_true",
                    help="Explicitly launch selected planned rows",
                )
    finish = actions.add_parser("audit-results")
    finish.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    require(
        (ROOT / "runs").resolve() in root.parents
        and root.name.startswith("p100_v3_doe"),
        "Use a DOE root named p100_v3_doe* under this checkout's runs/",
    )
    if args.action == "generate":
        root.mkdir(parents=True, exist_ok=True)
        with exclusive(root):
            result = generate_all(root)
    else:
        manifest = load(root / "design.json")
        receipt = load(root / "preflight.json")
        require(
            sha(root / "design.json") == receipt["design_sha256"],
            "Frozen design seal mismatch",
        )
        if args.action == "audit-design":
            result = audit_design(root, manifest)
            require(
                sha(root / "stage_a.csv") == receipt["stage_a_sha256"],
                "Stage A aggregate seal mismatch",
            )
        else:
            smoke = args.action == "smoke"
            execution = root / "smoke" if smoke else root
            execution.mkdir(exist_ok=True)
            with exclusive(root), ledger(execution) as connection:
                if args.action == "audit-results":
                    audit_design(root, manifest)
                    collect_attempts(root, manifest, connection)
                    result = export(root, manifest, connection, args.require_complete)
                else:
                    cases = choose_contexts(
                        root, manifest, args.case, args.model, args.partition
                    )
                    if smoke and not args.case and not args.model:
                        subset = {}
                        for cid in cases:
                            context = manifest["contexts"][cid]
                            if context["u"] == 0.6:
                                subset.setdefault(context["j"], cid)
                        cases = list(subset.values())
                    if smoke:
                        require(
                            len(cases) <= 12, "Smoke scope is limited to 12 contexts"
                        )
                    if args.action == "stage-b":
                        audit_design(root, manifest)
                        collect_attempts(root, manifest, connection)
                        result = prepare_b(root, manifest, connection, cases)
                    else:
                        dry = args.dry_run or (not smoke and not args.execute)
                        audit_design(root, manifest)
                        batch = len(cases) if smoke or dry else args.batch_contexts
                        require(batch > 0, "Batch context count must be positive")
                        reports = []
                        for offset in range(0, len(cases), batch):
                            report = run(
                                root,
                                manifest,
                                connection,
                                "A" if smoke else args.stage,
                                cases[offset : offset + batch],
                                args.workers,
                                args.chunks,
                                dry,
                                smoke,
                            )
                            reports.append(report)
                            if report["status"] == "INCOMPLETE_OR_INVALID":
                                break
                        result = (
                            reports[0]
                            if len(reports) == 1
                            else dict(
                                status=reports[-1]["status"],
                                scheduled=sum(
                                    report["scheduled"] for report in reports
                                ),
                                attempts=reports,
                            )
                        )
    if "splits" in result:
        result = {
            **result,
            "splits": len(result["splits"]),
            "exposure": "see design.json and preflight.json",
        }
    print(json.dumps(result, indent=2))
    return int(
        result.get("status") == "INCOMPLETE_OR_INVALID"
        or (getattr(args, "require_complete", False) and result["status"] != "PASS")
    )


if __name__ == "__main__":
    raise SystemExit(main())
