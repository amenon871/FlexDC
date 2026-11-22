"""Independent DOE checks for experiment membership, plan and output contracts."""

import json
import math
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import combinations

from am_vet_flexdc_sweep_dataset_v3 import parse_vector

from .design import (
    A_COUNT,
    B_COUNT,
    CLUSTER,
    EXPERIMENT,
    ISO,
    MODES,
    POWER,
    RUNTIME,
    SOURCES,
)
from .io import ROOT, SEED, VERSION, load, sha
from .sampling import coords, domain_for, margins, plan_rows, point_key


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matched_stage_a(manifest, read_rows):
    """Compare all utilization pairs by the exact-plan sample index."""
    groups = defaultdict(list)
    for cid, context in manifest["contexts"].items():
        groups[context["workload"]].append(cid)
    report = dict(
        workloads_checked=0,
        cross_utilization_comparisons=0,
        bid_coordinate_comparisons=0,
        maximum_coordinate_difference=0.0,
        mismatches=0,
        examples=[],
    )
    fields = ("Pbar_kw_per_server", "R_kw_per_server", "weights")
    index = lambda row: int(row["weight_index_within_pr"])
    for workload, contexts in sorted(groups.items()):
        if len(contexts) < 2:
            continue
        report["workloads_checked"] += 1
        require(
            len({manifest["contexts"][cid]["u"] for cid in contexts}) == len(contexts),
            f"Repeated utilization for {workload}",
        )
        rows = {cid: sorted(read_rows(cid), key=index) for cid in sorted(contexts)}
        domains = {domain_for(manifest, manifest["contexts"][cid]) for cid in rows}
        require(len(domains) == 1, f"Utilization-dependent bid domain for {workload}")
        domain = next(iter(domains))
        seen = set()
        for cid, plan in rows.items():
            require(
                [index(row) for row in plan] == list(range(A_COUNT)),
                f"Invalid Stage A indices {cid}",
            )
            ids = {row["plan_row_id"] for row in plan}
            require(
                len(ids) == A_COUNT
                and not ids & seen
                and all(row["context_id"] == cid for row in plan),
                f"Context or Plan_Row_ID collision for {cid}",
            )
            seen.update(ids)
            require(
                len({point_key(domain.validate(coords(row))) for row in plan})
                == A_COUNT,
                f"Duplicate bids for {cid}",
            )
        for left, right in combinations(rows, 2):
            report["cross_utilization_comparisons"] += 1
            for first, second in zip(rows[left], rows[right], strict=True):
                report["bid_coordinate_comparisons"] += 1
                delta = max(
                    abs(a - b)
                    for a, b in zip(coords(first), coords(second), strict=True)
                )
                report["maximum_coordinate_difference"] = max(
                    report["maximum_coordinate_difference"], delta
                )
                if delta != 0 or any(first[key] != second[key] for key in fields):
                    report["mismatches"] += 1
                    if len(report["examples"]) < 5:
                        report["examples"].append([left, right, index(first)])
    return report


def audit_design(root, manifest):
    profiles, workloads, groups, contexts = (
        manifest[key] for key in ("profiles", "workloads", "groups", "contexts")
    )
    require(
        manifest["version"] == VERSION and set(SOURCES) <= profiles.keys(),
        "Wrong design or sources",
    )
    require(
        sha(root / "evaluation_reporting.json") == manifest["reporting_sha256"],
        "Evaluation reporting seal changed",
    )
    require(
        len(profiles) == 104
        and len(workloads) == 741
        and len(groups) == 247
        and len(contexts) == 1632,
        "DOE cardinality mismatch",
    )
    require(
        Counter(w["j"] for w in workloads.values())
        == {2: 84, 3: 168, 4: 210, 5: 168, 6: 84, 7: 24, 8: 3},
        "Workload J distribution mismatch",
    )
    require(
        Counter(c["u"] for c in contexts.values()) == {0.6: 741, 0.7: 150, 0.8: 741},
        "Utilization distribution mismatch",
    )
    for source in SOURCES:
        require(
            {
                (p["kind"], p["percent"])
                for p in profiles.values()
                if p["source"] == source
            }
            == {("measured", 0), *MODES},
            f"Profile variation mismatch: {source}",
        )
    for profile in profiles.values():
        require(
            sha(root / profile["path"]) == profile["sha256"],
            "Profile config hash mismatch",
        )
        source = profiles[profile["source"]]
        changed = (
            POWER
            if profile["kind"] == "power"
            else RUNTIME
            if profile["kind"] == "runtime"
            else ()
        )
        for field, value in profile["physical"].items():
            expected = Decimal(source["original"][field])
            if field in changed:
                expected *= 1 + Decimal(profile["percent"]) / 100
            require(
                Decimal(value) == expected,
                f"Incorrect profile scaling {profile['path']}/{field}",
            )
        require(
            profile["original"] == source["original"]
            and profile["qos"] == (5 if source["role"] == "T" else 4)
            and sha(ROOT / profile["source_config"]) == profile["source_sha256"],
            "Profile source/QoS mismatch",
        )
    require(
        sha(ISO) == manifest["iso_sha256"]
        and sha(EXPERIMENT) == manifest["canonical_experiment_sha256"]
        and sha(CLUSTER) == manifest["cluster_sha256"]
        and sha(root / manifest["gradient"]) == manifest["gradient_sha256"],
        "Base configuration changed",
    )
    for group, children in groups.items():
        require(
            len(children) == 3 and len(set(children)) == 3,
            f"Wrong sibling count {group}",
        )
        original = workloads[children[0]]
        changed = set()
        for index, child in enumerate(children):
            w = workloads[child]
            require(
                w["parent"] == group
                and w["variant"] == index
                and len(set(w["sources"])) == w["j"]
                and [profiles[p]["source"] for p in w["jobs"]] == original["sources"]
                and sha(root / w["path"]) == w["sha256"],
                f"Workload lineage/hash mismatch {child}",
            )
            difference = [
                i for i, (a, b) in enumerate(zip(w["jobs"], original["jobs"])) if a != b
            ]
            require(
                len(difference) == bool(index) and not changed.intersection(difference),
                f"Variant substitution mismatch {child}",
            )
            changed.update(difference)
    for experiment, checksum in {
        (c["experiment"], c["experiment_sha256"]) for c in contexts.values()
    }:
        require(
            sha(root / experiment) == checksum,
            "Generated experiment config hash mismatch",
        )

    folds = manifest["folds"]
    require(
        [len(f) for f in folds] == [48, 48, 48, 47, 47]
        and len(set().union(*map(set, folds))) == 238,
        "Baseline fold overlap",
    )
    require(
        len(manifest["models"]) == len(set(manifest["models"])) == 22,
        "Model split count",
    )
    u07, pair_sets = set(), {1: [], 2: []}
    counts = {}
    for model in manifest["models"]:
        path = root / "splits" / (model + ".json")
        require(
            sha(path) == manifest["split_hashes"][model], f"Split hash mismatch {model}"
        )
        split = load(path)
        train, val = (set(split[k]) for k in ("train", "validation"))
        tests = set().union(*map(set, split["test"].values()))
        require(
            not train & val
            and not (train | val) & tests
            and (train | val | tests) <= contexts.keys(),
            f"Split overlap {model}",
        )
        for cid in train | val:
            c = contexts[cid]
            w = workloads[c["workload"]]
            require(
                c["u"] in (0.6, 0.8)
                and 2 <= c["j"] <= 6
                and not set(split.get("held_out", [])).intersection(w["sources"]),
                f"Source/utilization/cardinality leakage {model}/{cid}",
            )
            require(
                model != "physical_variation" or w["variant"] == 0,
                "Physical split includes synthetic profiles",
            )
        if model.startswith("baseline_"):
            index = int(model.split("_")[-1]) - 1
            selected = set(split["u07_groups"])
            require(
                len(selected) == 10 and selected <= set(folds[index]),
                f"U0.7 fold ownership {model}",
            )
            require(
                len(split["test"]["u07_matched"]) == 90,
                "Expected 30 matched contexts per utilization",
            )
            require(
                set(split["test"]["held_out_groups"])
                == {
                    f"{child}_U{u}"
                    for group in folds[index]
                    for child in groups[group]
                    for u in (6, 8)
                },
                "Incorrect baseline test fold",
            )
            require(
                set(split["test"]["u07_matched"])
                == {
                    f"{child}_U{u}"
                    for group in selected
                    for child in groups[group]
                    for u in (6, 7, 8)
                },
                "Incorrect U0.7 matched contexts",
            )
            u07.update(selected)
        if model.startswith(("leave_one_", "leave_two_")):
            held = set(split["held_out"])

            def matching(cid, n):
                w = workloads[contexts[cid]["workload"]]
                return (
                    w["j"] <= 6
                    and contexts[cid]["u"] in (0.6, 0.8)
                    and len(held.intersection(w["sources"])) == n
                )

            universe = {
                cid for cid in contexts if any(matching(cid, n) for n in range(3))
            }
            require(
                train | val | tests == universe, f"Holdout split omits contexts {model}"
            )
            require(
                all(matching(cid, 1) for cid in split["test"]["one_unseen"]),
                f"Wrong one-unseen split {model}",
            )
            if model.startswith("leave_two_"):
                pair_sets[int(model.split("_")[2])].append(tuple(sorted(held)))
                require(
                    all(matching(cid, 0) for cid in split["test"]["familiar_control"])
                    and all(matching(cid, 2) for cid in split["test"]["both_unseen"]),
                    f"Wrong leave-two evaluation split {model}",
                )
        if model == "physical_variation":
            for key, side in (
                ("variants_of_training", train),
                ("variants_of_validation", val),
            ):
                expected = {
                    f"{child}_U{cid[-1]}"
                    for cid in side
                    for child in groups[workloads[contexts[cid]["workload"]]["parent"]][
                        1:
                    ]
                }
                require(
                    set(split["test"][key]) == expected,
                    f"Wrong paired physical evaluation {key}",
                )
        if model != "physical_variation":
            for side in (train, val):
                for cid in side:
                    c = contexts[cid]
                    require(
                        {
                            f"{child}_U{u}"
                            for child in groups[workloads[c["workload"]]["parent"]]
                            for u in (6, 8)
                        }
                        <= side,
                        f"Parent group divided across partitions {model}",
                    )
        counts[model] = dict(
            train=len(train),
            validation=len(val),
            test={key: len(value) for key, value in split["test"].items()},
        )
    require(
        len(u07) == 50
        and u07
        == set(manifest["u07_groups"])
        == {
            workloads[c["workload"]]["parent"]
            for c in contexts.values()
            if c["u"] == 0.7
        },
        "U0.7 context mismatch",
    )
    for pairs in pair_sets.values():
        require(
            len(pairs) == 4
            and sorted(source for pair in pairs for source in pair) == sorted(SOURCES),
            "Pair set not disjoint",
        )
        types = []
        for a, b in pairs:
            family_a, role_a = a.rsplit("-", 1)
            family_b, role_b = b.rsplit("-", 1)
            types.append(
                role_a + role_b
                if role_a == role_b
                else ("same" if family_a == family_b else "different")
            )
        require(
            set(types) == {"TT", "II", "same", "different"},
            "Pair-set category mismatch",
        )
    require(not set(pair_sets[1]) & set(pair_sets[2]), "Repeated leave-two pair")
    return dict(
        profiles=len(profiles),
        workloads=len(workloads),
        groups=len(groups),
        contexts=len(contexts),
        stage_a_rows=len(contexts) * A_COUNT,
        stage_b_rows=len(contexts) * B_COUNT,
        total_rows=len(contexts) * (A_COUNT + B_COUNT),
        splits=counts,
        exposure=manifest["exposure"],
    )


def audit_plan(root, manifest, context_id, stage, rows, previous=()):
    count = A_COUNT if stage == "A" else B_COUNT
    require(len(rows) == count, f"Wrong {stage} count for {context_id}")
    domain = domain_for(manifest, manifest["contexts"][context_id])
    occupied = {point_key(coords(row)) for row in previous}
    ids = set()
    for index, row in enumerate(rows):
        point = domain.validate(coords(row))
        key = point_key(point)
        require(
            key not in occupied and row["plan_row_id"] not in ids,
            f"Duplicate bid or ID {context_id}",
        )
        expected = list(
            plan_rows(
                root,
                manifest,
                context_id,
                stage,
                [(point, row["weight_family"], row["anchor_type"])],
            )
        )[0]
        from .io import digest

        expected["plan_row_id"] = (
            f"P100D_{stage}_{digest([context_id, index, key])[:24]}"
        )
        expected["base_plan_row_id"] = expected["plan_row_id"]
        expected["weight_index_within_pr"] = index
        require(
            all(str(v) == row[k] for k, v in expected.items()),
            f"Plan metadata mismatch {row['plan_row_id']}",
        )
        ids.add(row["plan_row_id"])
        occupied.add(key)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=2e-9, abs_tol=2e-10)


def audit_observation(row, result, diagnostic, evidence, manifest):
    rid, cid = row["plan_row_id"], row["context_id"]
    context = manifest["contexts"][cid]
    workload = manifest["workloads"][context["workload"]]
    p, r, *weights = coords(row)
    j = context["j"]
    for record in (result, diagnostic):
        require(
            record["Iteration"] == result["Iteration"]
            and record["Plan_Row_ID"] == rid
            and record["Context_ID"] == cid
            and record["Design_Version"] == VERSION
            and record["Workload_Config"] == row["workload_config"]
            and int(record["Simulation_Seed"]) == SEED
            and int(record["server_count"]) == 1000
            and close(record["utilization"], context["u"])
            and close(record["Pbar_kw_per_server"], p)
            and close(record["R_kw_per_server"], r)
            and int(record["P_actual_watts"]) == round(p * 1e6)
            and int(record["R_actual_watts"]) == round(r * 1e6),
            f"Applied compact coordinates mismatch {rid}",
        )
    require(
        result["Policy_Name"] == "AQA" and int(result["workload_mix_size"]) == j,
        f"Policy/J mismatch {rid}",
    )
    require(
        all(close(result[f"Weight_{i}"], w) for i, w in enumerate(weights)),
        f"Serialized weights mismatch {rid}",
    )
    mix = json.loads(result["workload_mix"])
    expected_mix = []
    for name in workload["jobs"]:
        profile = manifest["profiles"][name]
        expected_mix.append(
            [float(profile["physical"][key]) for key in (*POWER, *RUNTIME)]
            + [profile["qos"], 1]
        )
    require(mix == expected_mix, f"Applied job physics or QoS differs {rid}")
    counts = evidence["job_counts"]
    require(
        evidence["fast_aqa"] is True
        and evidence["policy_module"] == "peacsim.aqa_runtimepolicy"
        and evidence["n"] == 1000
        and close(evidence["u"], context["u"])
        and evidence["seed"] == SEED
        and evidence["duration"] == 3600
        and evidence["p_watts"] == round(p * 1e6)
        and evidence["r_watts"] == round(r * 1e6)
        and len(evidence["weights"]) == j
        and all(close(a, b) for a, b in zip(evidence["weights"], weights))
        and evidence["qos"] == [p[4] for p in expected_mix]
        and [(c["job_id"], c["name"]) for c in counts]
        == list(enumerate(workload["jobs"])),
        f"Runtime evidence identity differs {rid}",
    )
    require(
        all(
            0 < c["completed"] + c["eligible_unfinished"] <= c["generated"]
            for c in counts
        )
        and sum(c["generated"] for c in counts) == evidence["arrival_rows"],
        f"Insufficient job-type evidence {rid}",
    )
    arrival = evidence["arrival_sha256"]
    require(
        arrival == context["arrival_sha256"]
        and evidence["arrival_rows"] == context["arrival_rows"],
        f"Realized arrival table differs from frozen context {rid}",
    )
    probabilities = parse_vector(result["QoS_Delay_Probabilities"])
    mean = float(result["Mtrack_Error_MeanAbs_Normalized"])
    p90 = float(diagnostic["Ctrack_Epsilon_90th"])
    require(
        len(probabilities) == j
        and all(0 <= q <= 1 for q in probabilities)
        and math.isfinite(mean)
        and mean >= 0
        and math.isfinite(p90)
        and p90 >= 0,
        f"Invalid tracking/QoS label {rid}",
    )
    track = math.log1p(math.exp(-abs(10 * (p90 - 0.30)))) + max(10 * (p90 - 0.30), 0)
    qos = 20 * sum(math.log1p(math.exp(2 * (q - 0.10))) for q in probabilities)
    power = float(result["Simulator_Power_Cost"])
    mtrack = float(result["Mtrack_Cost"])
    objective = power + mtrack + track + qos
    require(
        close(power, 0.1 * (round(p * 1e6) - round(r * 1e6)) / 1000)
        and close(mtrack, 0.1 * round(r * 1e6) * mean / 1000)
        and close(result["Simulator_RSR_Total_Cost"], power + mtrack)
        and close(result["QoS_Delay_Probability_Sum"], sum(probabilities))
        and close(diagnostic["Ctrack_Weighted_Cost"], track)
        and close(diagnostic["Diagnostic_FlexDC_SoftPlus_QoS_Cost"], qos)
        and close(diagnostic["Diagnostic_FullPaperObjective_Cost"], objective),
        f"Cost reconstruction mismatch {rid}",
    )
    margin, distance = margins(p90, probabilities)
    return dict(
        plan_row_id=rid,
        context_id=cid,
        stage=row["pr_stratum"],
        mean_tracking=mean,
        p90=p90,
        qos_probabilities=probabilities,
        margin=margin,
        violation_distance=distance,
        feasible=margin <= 0,
        objective=objective,
        power_cost=power,
        tracking_cost=mtrack,
        tracking_constraint_cost=track,
        qos_cost=qos,
        planned=row,
        evidence=evidence,
        simulator_results=result,
        simulator_diagnostics=diagnostic,
    )
