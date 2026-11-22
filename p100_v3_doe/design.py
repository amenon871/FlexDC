"""Deterministic profile, workload, context and model split manifests."""

import io
import random
import tempfile
from collections import Counter, defaultdict
from configparser import ConfigParser
from decimal import Decimal
from itertools import combinations
from pathlib import Path

from am_generate_flexdc_exact_sweep_plan import (
    DEFAULT_BASE_GRADIENT_CONFIG,
    DEFAULT_SA_CONSTANTS_CONFIG,
    read_cost_constants,
    write_canonical_gradient_config,
)
from am_generate_flexdc_j2_pair_workloads import read_ini
from peacsim.create_tables import generate_arrival_rates, init_job_table
from peacsim.p100_doe_evidence import arrival_digest
from peacsim.parsing.experiment_config_reader import ExperimentConfigReader
from peacsim.parsing.job_profile_reader import JobProfileReader

from .io import ROOT, SEED, VERSION, atomic, digest, json_write, sha

FAMILIES = ("ResNet", "GPT-2", "Llama", "Bloom")
SECTIONS = ("Resnet", "GPT2", "Llama", "Bloom")
SOURCES = tuple(f"{family}-{role}" for family in FAMILIES for role in ("T", "I"))
SOURCE_FILES = {"T": "W1-train-qos4444.ini", "I": "W2-short-qos5555.ini"}
POWER = ("min_job_power_watts", "max_job_power_watts")
RUNTIME = ("min_time_seconds", "max_time_seconds")
MODES = tuple(
    (kind, percent)
    for kind in ("power", "runtime")
    for percent in (-15, -10, -5, 5, 10, 15)
)
EXPERIMENT = (
    ROOT
    / "configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_1000.ini"
)
ISO = ROOT / "data/1-17-2025_traditional.csv"
CLUSTER = ROOT / "configs/cluster/cluster.ini"
DESIGN_SEED = 20261121
A_COUNT, B_COUNT = 150, 50


def order(namespace, value):
    return digest([DESIGN_SEED, namespace, value])


def ini_bytes(parser):
    buffer = io.StringIO()
    parser.write(buffer)
    return buffer.getvalue().encode()


def profile_library(root):
    profiles = {}
    for source in SOURCES:
        family, role = source.rsplit("-", 1)
        path = ROOT / "configs/workload" / SOURCE_FILES[role]
        parser = read_ini(path)
        section = f"{SECTIONS[FAMILIES.index(family)]}.{'train' if role == 'T' else 'infer'}.4"
        original = {key: parser[section][key] for key in (*POWER, *RUNTIME, "job_size")}
        if int(original["job_size"]) != 1:
            raise ValueError(f"Expected single-server P100 profile {source}")
        for kind, pct in (("measured", 0), *MODES):
            name = source if not pct else f"{source}_{kind}_{pct:+03d}"
            physical = dict(original)
            for key in (
                POWER if kind == "power" else RUNTIME if kind == "runtime" else ()
            ):
                physical[key] = format(
                    Decimal(physical[key]) * (1 + Decimal(pct) / 100), "f"
                )
            config = ConfigParser()
            config[name] = {
                **physical,
                "qos_constraint": "5.0" if role == "T" else "4.0",
            }
            destination = root / "profiles" / f"{name}.ini"
            atomic(destination, ini_bytes(config))
            profiles[name] = dict(
                source=source,
                family=family,
                role=role,
                kind=kind,
                percent=pct,
                original=original,
                physical=physical,
                qos=5.0 if role == "T" else 4.0,
                path=destination.relative_to(root).as_posix(),
                sha256=sha(destination),
                source_config=path.relative_to(ROOT).as_posix(),
                source_sha256=sha(path),
                source_section=section,
            )
    assert len(profiles) == 104
    return profiles


def workload_library(root, profiles):
    workloads, groups = {}, {}
    exposure, source_exposure, per_j, mode_exposure = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    for j in range(2, 9):
        combinations_j = sorted(
            combinations(SOURCES, j), key=lambda x: order("composition", x)
        )
        for number, sources in enumerate(combinations_j, 1):
            parent = f"P100D_J{j}_{number:03d}"
            choices = [tuple(sources)]
            changed = set()
            for variant in range(2):
                candidates = [
                    (i, name)
                    for i, source in enumerate(sources)
                    if i not in changed
                    for name, p in profiles.items()
                    if p["source"] == source and p["percent"]
                ]

                def score(item):
                    i, name = item
                    p = profiles[name]
                    return (
                        exposure[name],
                        mode_exposure[p["kind"], p["percent"]],
                        source_exposure[sources[i]],
                        per_j[j, sources[i]],
                        order(parent + str(variant), name),
                    )

                slot, profile = min(candidates, key=score)
                changed.add(slot)
                exposure[profile] += 1
                p = profiles[profile]
                mode_exposure[p["kind"], p["percent"]] += 1
                source_exposure[sources[slot]] += 1
                per_j[j, sources[slot]] += 1
                children = list(sources)
                children[slot] = profile
                choices.append(tuple(children))
            group = []
            for index, jobs in enumerate(choices):
                name = f"{parent}_V{index}"
                config = ConfigParser()
                for job in jobs:
                    p = profiles[job]
                    config[job] = {**p["physical"], "qos_constraint": str(p["qos"])}
                path = root / "workloads" / f"{name}.ini"
                atomic(path, ini_bytes(config))
                parsed = JobProfileReader(str(path))
                if list(parsed.all_jobs) != list(range(j)) or list(
                    parsed.all_jobs.values()
                ) != list(jobs):
                    raise ValueError(f"FlexDC job index order mismatch for {name}")
                for slot, job in enumerate(jobs):
                    p = profiles[job]
                    parsed_values = (
                        parsed.all_min_job_power[slot],
                        parsed.all_max_job_power[slot],
                        parsed.all_min_execution_time[slot],
                        parsed.all_max_execution_time[slot],
                        parsed.all_job_size[slot],
                    )
                    expected = tuple(
                        float(p["physical"][key])
                        for key in (*POWER, *RUNTIME, "job_size")
                    )
                    if (
                        parsed_values != expected
                        or parsed.all_job_qos_constraints[slot] != p["qos"]
                    ):
                        raise ValueError(
                            f"FlexDC parser altered job physics or QoS: {name}/{job}"
                        )
                group.append(name)
                workloads[name] = dict(
                    parent=parent,
                    sources=list(sources),
                    jobs=list(jobs),
                    j=j,
                    variant=index,
                    path=path.relative_to(root).as_posix(),
                    sha256=sha(path),
                )
            groups[parent] = group
    assert len(workloads) == 741 and len(exposure) == 96
    return (
        workloads,
        groups,
        dict(
            by_variant=dict(exposure),
            by_source=dict(source_exposure),
            by_mode={f"{kind}_{pct:+d}": n for (kind, pct), n in mode_exposure.items()},
            by_j_source={f"J{j}_{source}": n for (j, source), n in per_j.items()},
        ),
    )


def split_sample(groups, metadata, count, salt):
    """Allocate stratified quotas by largest remainder across J and training count."""
    strata = defaultdict(list)
    for group in groups:
        strata[metadata[group]].append(group)
    total = len(groups)
    quotas = {key: len(values) * count // total for key, values in strata.items()}
    remainder = sorted(
        strata, key=lambda key: (-(len(strata[key]) * count % total), order(salt, key))
    )
    for key in remainder[: count - sum(quotas.values())]:
        quotas[key] += 1
    return sorted(
        group
        for key, values in strata.items()
        for group in sorted(values, key=lambda value: order(salt, value))[: quotas[key]]
    )


def pair_sets():
    def match(remaining):
        if not remaining:
            yield ()
        else:
            for i in range(1, len(remaining)):
                pair = (remaining[0], remaining[i])
                for rest in match(remaining[1:i] + remaining[i + 1 :]):
                    yield (pair, *rest)

    def kind(pair):
        (first, a), (second, b) = [item.rsplit("-", 1) for item in pair]
        return a + b if a == b else "same" if first == second else "different"

    valid = sorted(
        (
            pairs
            for pairs in match(SOURCES)
            if {kind(pair) for pair in pairs} == {"TT", "II", "same", "different"}
        ),
        key=lambda pairs: order("pairs", pairs),
    )
    pair_a = valid[0]
    pair_b = next(pairs for pairs in valid if not set(pairs) & set(pair_a))
    return pair_a, pair_b


def splits(workloads, groups):
    originals = {group: workloads[children[0]] for group, children in groups.items()}
    eligible = {g for g, w in originals.items() if w["j"] <= 6}
    metadata = {
        g: (w["j"], sum(s.endswith("-T") for s in w["sources"]))
        for g, w in originals.items()
    }
    remaining, folds = set(eligible), []
    for i, size in enumerate((48, 48, 48, 47, 47)):
        fold = split_sample(remaining, metadata, size, f"fold{i}")
        folds.append(fold)
        remaining.difference_update(fold)

    def contexts(group_ids, utils=(6, 8), originals_only=False):
        return sorted(
            f"{w}_U{u}"
            for g in group_ids
            for w in (groups[g][:1] if originals_only else groups[g])
            for u in utils
        )

    def partition(side, salt, count):
        val = set(split_sample(side, metadata, count, salt))
        return side - val, val

    models, u07 = {}, []
    for i, fold in enumerate(folds):
        label = f"baseline_{i + 1}"
        selected = split_sample(fold, metadata, 10, label + "util")
        u07.extend(selected)
        train, val = partition(eligible - set(fold), label, 24)
        models[label] = dict(
            train=contexts(train),
            validation=contexts(val),
            test={
                "held_out_groups": contexts(fold),
                "larger_j": contexts(set(groups) - eligible),
                "u07_matched": contexts(selected, (6, 7, 8)),
            },
            u07_groups=selected,
        )

    for source in SOURCES:
        held = {g for g in eligible if source in originals[g]["sources"]}
        train, val = partition(eligible - held, source, 12)
        assert len(held) == 119
        models["leave_one_" + source] = dict(
            held_out=[source],
            train=contexts(train),
            validation=contexts(val),
            test={"one_unseen": contexts(held)},
        )

    for set_index, pairs in enumerate(pair_sets(), 1):
        for pair_index, pair in enumerate(pairs, 1):
            label = f"leave_two_{set_index}_{pair_index}"
            categories = [
                {
                    g
                    for g in eligible
                    if len(set(pair).intersection(originals[g]["sources"])) == i
                }
                for i in range(3)
            ]
            assert [len(group) for group in categories] == [57, 124, 57]
            control = set(split_sample(categories[0], metadata, 11, label + "control"))
            train, val = partition(categories[0] - control, label, 6)
            models[label] = dict(
                held_out=list(pair),
                train=contexts(train),
                validation=contexts(val),
                test={
                    "familiar_control": contexts(control),
                    "one_unseen": contexts(categories[1]),
                    "both_unseen": contexts(categories[2]),
                },
            )

    train, val = partition(eligible, "physical", 24)

    def variants(ids):
        return sorted(f"{w}_U{u}" for g in ids for w in groups[g][1:] for u in (6, 8))

    models["physical_variation"] = dict(
        train=contexts(train, originals_only=True),
        validation=contexts(val, originals_only=True),
        test={
            "variants_of_training": variants(train),
            "variants_of_validation": variants(val),
        },
    )
    return models, folds, u07


def reporting_contract(models, contexts):
    def counts(model):
        train = model["train"]
        return dict(
            workload_definitions=len({contexts[cid]["workload"] for cid in train}),
            contexts=len(train),
            planned_a_rows=len(train) * A_COUNT,
            planned_b_rows=len(train) * B_COUNT,
            planned_total_rows=len(train) * (A_COUNT + B_COUNT),
        )

    baseline = {
        name: model for name, model in models.items() if name.startswith("baseline_")
    }
    one, two = {}, {}
    for name, model in models.items():
        if name.startswith("leave_one_"):
            test = set(model["test"]["one_unseen"])
            matches = {
                fold: sorted(test.intersection(spec["test"]["held_out_groups"]))
                for fold, spec in baseline.items()
            }
            assert set().union(*map(set, matches.values())) == test
            assert sum(map(len, matches.values())) == len(test)
            one[name] = dict(
                held_out_source=model["held_out"][0], baseline_test_contexts=matches
            )
        if name.startswith("leave_two_"):
            (a, role_a), (b, role_b) = [
                source.rsplit("-", 1) for source in model["held_out"]
            ]
            category = (
                "TT"
                if role_a == role_b == "T"
                else "II"
                if role_a == role_b == "I"
                else "same_family_T_I"
                if a == b
                else "different_family_T_I"
            )
            two[name] = dict(
                set=int(name.split("_")[2]),
                category=category,
                held_out_sources=model["held_out"],
                groups=model["test"].keys(),
            )
            two[name]["groups"] = list(two[name]["groups"])
    return dict(
        training_sample_counts={name: counts(model) for name, model in models.items()},
        leave_one=dict(
            models=one,
            delta="Error(Leave X Out) - Error(Baseline)",
            match_key="Plan_Row_ID within specified baseline test context",
            interpretation="Unequal training-set size confounds isolation of source novelty.",
        ),
        leave_two=dict(
            models=two,
            aggregation="Unweighted mean of the two pair-level metric values per category and presence group.",
        ),
        utilization=dict(
            matched_stage="A",
            match_keys=["workload_id", "stage_a_sample_index"],
            stage_b="Context-specific measured targeting. Analyze separately from matched Stage A.",
        ),
    )


def generate(root):
    profiles = profile_library(root)
    workloads, groups, exposure = workload_library(root, profiles)
    models, folds, u07 = splits(workloads, groups)
    experiments = {}
    for u in (6, 7, 8):
        parser = ConfigParser()
        parser.read(EXPERIMENT)
        parser["system"]["utilization"] = str(u / 10)
        parser["system"]["random_seed"] = str(SEED)
        parser["system"]["workload_trace"] = "poisson"
        experiment = root / "experiments" / f"N1000_U{u}.ini"
        atomic(experiment, ini_bytes(parser))
        experiments[u] = experiment.relative_to(root).as_posix()
    gradient = root / "experiments/cost.ini"
    cost = read_cost_constants(ROOT / DEFAULT_SA_CONSTANTS_CONFIG, ROOT)
    # Canonical helper creates the config in a scratch path before immutable publication.
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        path = Path(tmp) / "cost.ini"
        write_canonical_gradient_config(
            base_path=ROOT / DEFAULT_BASE_GRADIENT_CONFIG,
            output_path=path,
            constants=cost,
            overwrite=False,
        )
        base = str(ROOT / DEFAULT_BASE_GRADIENT_CONFIG)
        portable = path.read_text(encoding="utf8").replace(
            base, DEFAULT_BASE_GRADIENT_CONFIG
        )
        atomic(gradient, portable.encode())

    contexts = {}
    experiment_readers = {
        u: ExperimentConfigReader(str(root / path)) for u, path in experiments.items()
    }
    experiment_hashes = {u: sha(root / path) for u, path in experiments.items()}
    for name, workload in workloads.items():
        job = JobProfileReader(str(root / workload["path"]))
        for u in (6, 7, 8) if workload["parent"] in u07 else (6, 8):
            rates = [
                u
                / 10
                * 1000
                / workload["j"]
                / float(profiles[p]["physical"][RUNTIME[0]])
                for p in workload["jobs"]
            ]
            exp = experiment_readers[u]
            if rates != generate_arrival_rates(exp, job):
                raise ValueError(f"Unexpected simulator arrival rates for {name}/U{u}")
            state = random.getstate()
            try:
                random.seed(SEED)
                table = init_job_table(exp, job)
            finally:
                random.setstate(state)
            contexts[f"{name}_U{u}"] = dict(
                workload=name,
                u=u / 10,
                n=1000,
                j=workload["j"],
                seed=SEED,
                experiment=experiments[u],
                experiment_sha256=experiment_hashes[u],
                arrival_rates=rates,
                arrival_sha256=arrival_digest(table[:3].T),
                arrival_rows=table.shape[1],
            )
    for model, definition in models.items():
        json_write(root / "splits" / (model + ".json"), definition)
    json_write(root / "evaluation_reporting.json", reporting_contract(models, contexts))
    manifest = dict(
        version=VERSION,
        design_seed=DESIGN_SEED,
        profiles=profiles,
        workloads=workloads,
        groups=groups,
        exposure=exposure,
        contexts=contexts,
        folds=folds,
        u07_groups=u07,
        models=sorted(models),
        split_hashes={m: sha(root / "splits" / (m + ".json")) for m in models},
        reporting_sha256=sha(root / "evaluation_reporting.json"),
        gradient=gradient.relative_to(root).as_posix(),
        gradient_sha256=sha(gradient),
        canonical_experiment_sha256=sha(EXPERIMENT),
        iso_sha256=sha(ISO),
        cluster_sha256=sha(CLUSTER),
        metric=dict(
            p90_threshold=0.30,
            qos_probability_threshold=0.10,
            false_feasible_denominator="actual_infeasible",
        ),
    )
    json_write(root / "design.json", manifest)
    return manifest
