"""Tests for the P100 DOE planning and recovery contracts."""

import csv

import pytest

from p100_v3_doe.audit import audit_design
from p100_v3_doe.collection import prepare_b, read_plan, seal_plan
from p100_v3_doe.design import A_COUNT, B_COUNT, generate
from p100_v3_doe.io import ROOT, json_write, protected, sha
from p100_v3_doe.ledger import accept, ledger, measured, reconcile
from p100_v3_doe.sampling import (
    adaptive,
    broad,
    coords,
    domain_for,
    margins,
    plan_rows,
    point_key,
)


@pytest.fixture(scope="module")
def design(tmp_path_factory):
    root = tmp_path_factory.mktemp("design")
    return root, generate(root)


def test_complete_science_counts_and_splits(design):
    root, manifest = design
    report = audit_design(root, manifest)
    assert [
        report[k]
        for k in (
            "profiles",
            "workloads",
            "contexts",
            "stage_a_rows",
            "stage_b_rows",
            "total_rows",
        )
    ] == [104, 741, 1632, 244800, 81600, 326400]
    assert len(report["splits"]) == 22
    assert len(manifest["exposure"]["by_variant"]) == 96
    assert (
        max(manifest["exposure"]["by_mode"].values())
        - min(manifest["exposure"]["by_mode"].values())
        <= 3
    )
    assert [len(fold) for fold in manifest["folds"]] == [48, 48, 48, 47, 47]
    assert protected()


def test_physical_variants_and_arrivals(design):
    _, d = design
    for workload_id, workload in d["workloads"].items():
        if not workload["variant"]:
            continue
        original = d["groups"][workload["parent"]][0]
        v, a = (d["contexts"][f"{k}_U6"] for k in (workload_id, original))
        diff = [p for p, s in zip(workload["jobs"], workload["sources"]) if p != s]
        assert len(diff) == 1
        if d["profiles"][diff[0]]["kind"] == "power":
            assert (
                v["arrival_rates"] == a["arrival_rates"]
                and v["arrival_sha256"] == a["arrival_sha256"]
            )
        else:
            assert v["arrival_rates"] != a["arrival_rates"]


@pytest.mark.parametrize("j", range(2, 9))
def test_stage_a_geometry(design, j):
    _, d = design
    cid, context = next((cid, c) for cid, c in d["contexts"].items() if c["j"] == j)
    domain = domain_for(d, context)
    points = broad(domain, cid)
    assert len(points) == A_COUNT == len({point_key(p) for p, *_ in points})
    assert broad(domain, cid) == points
    assert {k for _, k, _ in points} == {"physical_edge", "sobol"}
    assert all(domain.validate(p) == p for p, *_ in points)


@pytest.mark.parametrize("shape", ["near", "distant"])
def test_stage_b_requires_audit_and_stays_distinct(design, shape):
    _, d = design
    cid, c = next(iter(d["contexts"].items()))
    domain = domain_for(d, c)
    a = list(plan_rows(ROOT / "runs/p100_v3_doe", d, cid, "A", broad(domain, cid)))
    observed = {
        r["plan_row_id"]: dict(
            p90=(0.29 if i % 2 else 0.31) if shape == "near" else 0.9,
            qos_probabilities=[0.04] * c["j"],
        )
        for i, r in enumerate(a)
    }
    rejected = dict(observed)
    rejected.pop(next(iter(rejected)))
    with pytest.raises(ValueError, match="150 audited"):
        adaptive(domain, cid, a, rejected)
    b, diagnostics = adaptive(domain, cid, a, observed)
    assert len(b) == B_COUNT == len({point_key(p) for p, *_ in b})
    assert not {point_key(coords(r)) for r in a} & {point_key(p) for p, *_ in b}
    assert (diagnostics["boundary_directed"] > 0) == (shape == "near")
    assert diagnostics["boundary_directed"] + diagnostics["fallback"] == 50
    assert adaptive(domain, cid, a, observed) == (b, diagnostics)
    assert margins(0.30, [0.10] * c["j"]) == (0, 0)
    assert margins(0.60, [0.20] * c["j"]) == pytest.approx((1, 2**0.5))


def test_restart_only_missing_and_ledger_tamper(tmp_path):
    with ledger(tmp_path) as db:
        observation = dict(
            plan_row_id="a",
            context_id="C",
            stage="A",
            evidence={"arrival_sha256": "same"},
        )
        with db:
            accept(db, observation)
    with ledger(tmp_path) as db:
        assert set(["a", "b"]) - measured(db, "C", "A").keys() == {"b"}
        with pytest.raises(ValueError, match="Common-arrival"):
            with db:
                accept(
                    db,
                    dict(
                        observation,
                        plan_row_id="b",
                        evidence={"arrival_sha256": "other"},
                    ),
                )
        with pytest.raises(ValueError, match="Multiple distinct"):
            with db:
                accept(db, dict(observation, p90=0.5))
        db.execute("UPDATE observations SET body='{}' WHERE row_id='a'")
        with pytest.raises(ValueError, match="integrity"):
            measured(db, "C", "A")


def test_interrupted_worker_accepts_only_complete_audited_triplets(
    tmp_path, monkeypatch
):
    for name, entries in (
        ("grid_search_results.csv", ("a", "b")),
        ("grid_search_diagnostics.csv", ("a",)),
    ):
        with (tmp_path / name).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Iteration", "Plan_Row_ID"])
            writer.writerows((i, rid) for i, rid in enumerate(entries, 1))
    (tmp_path / "p100_doe_evidence.jsonl").write_text(
        '{"arrival_sha256":"same"}\n{"incomplete"', encoding="utf8"
    )
    monkeypatch.setattr(
        "p100_v3_doe.ledger.audit_observation",
        lambda row, result, diag, evidence, manifest: dict(
            plan_row_id=result["Plan_Row_ID"],
            context_id="C",
            stage="A",
            evidence=evidence,
        ),
    )
    with ledger(tmp_path) as db:
        report = reconcile(db, tmp_path, {"a": {}, "b": {}}, {}, {})
        assert report["committed"] == 1 and report["result_rows"] == 2
        assert set(measured(db, "C", "A")) == {"a"}
        assert set(["a", "b"]) - measured(db, "C", "A").keys() == {"b"}
        (tmp_path / "grid_search_diagnostics.csv").write_text(
            "Iteration,Plan_Row_ID\n1,F1_FOREIGN\n"
        )
        with pytest.raises(ValueError, match="Foreign"):
            reconcile(db, tmp_path, {"a": {}, "b": {}}, {}, {})


def test_stage_b_seals_complete_source_and_skips_partial_context(
    design, tmp_path, monkeypatch
):
    _, manifest = design
    monkeypatch.setattr("p100_v3_doe.sampling.ROOT", tmp_path.parent)
    json_write(tmp_path / "design.json", manifest)
    complete, partial = sorted(manifest["contexts"])[:2]
    for cid in (complete, partial):
        domain = domain_for(manifest, manifest["contexts"][cid])
        seal_plan(
            tmp_path,
            cid,
            "A",
            plan_rows(tmp_path, manifest, cid, "A", broad(domain, cid)),
        )
    with ledger(tmp_path) as db:
        for cid, count in ((complete, 150), (partial, 149)):
            rows, _ = read_plan(tmp_path, manifest, cid, "A")
            j = manifest["contexts"][cid]["j"]
            with db:
                for row in rows[:count]:
                    accept(
                        db,
                        dict(
                            plan_row_id=row["plan_row_id"],
                            context_id=cid,
                            stage="A",
                            p90=0.29,
                            qos_probabilities=[0.04] * j,
                            planned=row,
                            evidence={"arrival_sha256": "fixture"},
                        ),
                    )
        result = prepare_b(tmp_path, manifest, db, [complete, partial])
        assert result["ready_contexts"] == 1 and result["plan_rows"] == 50
        assert result["incomplete"] == [dict(context_id=partial, valid=149, missing=1)]
        b_path = tmp_path / "plans/B" / (complete + ".csv")
        before = sha(b_path)
        assert (
            prepare_b(tmp_path, manifest, db, [complete, partial]) == result
            and sha(b_path) == before
        )
        assert not (tmp_path / "plans/B" / (partial + ".csv")).exists()
