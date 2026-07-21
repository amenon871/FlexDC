#!/usr/bin/env python3
"""Generate the predeclared FlexDC Sweep V3 exact simulation plan.

Sweep V3 combines the row density of the original grid sweep with the
space-filling and provenance discipline introduced in Sweep V2.  It is
constructed entirely from physical/configuration constraints before any new
FlexDC simulation outputs are observed.

Default design per workload/server/utilization context:

* 16 Pbar values x 10 legal R values = 160 structured-grid P/R points;
* 96 scrambled-Sobol interior P/R points;
* 32 physical-boundary P/R points;
* one deterministic 40-vector weight bank at every P/R point;
* 5% of base configurations repeated with seeds 21 and 22;
* preassigned grouped train/validation/test labels (70/20/10).

With the repository's four default workloads, two server counts, and two
utilizations, the default plan contains 202,752 simulation rows.

The script does not run FlexDC.  It writes an immutable plan CSV and design
audit artifacts consumed by am_run_flexdc_dataset_parallel_combo.py.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import qmc


SCRIPT_DIR = Path(__file__).resolve().parent
PR_EPS = 1e-9
WEIGHT_EPS = 1e-9

DEFAULT_WORKLOAD_CONFIGS = [
    "configs/workload/W1-train-qos3333.ini",
    "configs/workload/W1-train-qos4444.ini",
    "configs/workload/W2-short-qos5_4.5_4_3.5.ini",
    "configs/workload/W2-short-qos5555.ini",
]
DEFAULT_EXPERIMENT_CONFIGS = [
    "configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_250.ini",
    "configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_1000.ini",
]
DEFAULT_UTILIZATIONS = [0.6, 0.8]
DEFAULT_SOFTMAX_TEMPERATURES = [0.5, 1.0, 2.0, 4.0]
DEFAULT_DIRICHLET_ALPHAS = [0.3, 1.0, 3.0, 10.0]
DEFAULT_SPLIT_FRACTIONS = (0.70, 0.20, 0.10)
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class Context:
    context_id: str
    workload_name: str
    workload_config: str
    experiment_config: str
    server_count: int
    utilization: float
    job_type_count: int
    pmin: float
    pmax: float
    p_lower: float
    p_upper: float
    pr_upper: float
    r_lower: float
    r_over_p_max: float
    configured_weight_lower: float
    configured_weight_upper: float
    legacy_weight_lower: float
    legacy_weight_upper: float
    effective_weight_lower: float
    effective_weight_upper: float
    experiment_seed: int
    simulation_duration: int
    iso_file_path: str
    iso_signal_start_hour: int
    iso_signal_granularity: int
    normalize_iso_signal: bool
    randomize_iso_start: bool


@dataclass(frozen=True)
class WeightEntry:
    weight_id: str
    weight_index: int
    weight_family: str
    weight_detail: str
    weights: tuple[float, ...]


def parse_csv_items(value: Optional[str], default: Iterable[str]) -> list[str]:
    if value is None or str(value).strip() == "":
        return list(default)
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated list.")
    return items


def parse_float_items(value: Optional[str], default: Iterable[float]) -> list[float]:
    return [float(item) for item in parse_csv_items(value, [str(x) for x in default])]


def parse_int_items(value: Optional[str], default: Iterable[int]) -> list[int]:
    return [int(item) for item in parse_csv_items(value, [str(x) for x in default])]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def resolve_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


def stable_hex(*parts: Any, length: int = 16) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate_legacy_weight_bounds(job_type_count: int, server_count: int) -> tuple[float, float]:
    equal = 1.0 / job_type_count
    relative_lower = 0.1 * equal
    relative_upper = 4.0 * equal
    server_lower = 1.0 / server_count
    lower = max(relative_lower, server_lower)
    upper_from_lower = 1.0 - (job_type_count - 1) * lower
    upper = min(relative_upper, upper_from_lower)
    if upper < lower:
        raise ValueError(
            f"Legacy weight bounds are infeasible for J={job_type_count}, "
            f"servers={server_count}: lower={lower}, upper={upper}"
        )
    return float(lower), float(upper)


def validate_configured_weight_domain(job_type_count: int, lower: float, upper: float) -> None:
    if job_type_count <= 0:
        raise ValueError("job_type_count must be positive.")
    if lower < 0 or upper > 1 or lower > upper:
        raise ValueError(f"Invalid configured weight interval [{lower}, {upper}].")
    if job_type_count * lower > 1.0 + WEIGHT_EPS:
        raise ValueError(
            f"Configured lower weight {lower} is infeasible for J={job_type_count}: "
            f"J*lower={job_type_count * lower}>1."
        )
    if job_type_count * upper < 1.0 - WEIGHT_EPS:
        raise ValueError(
            f"Configured upper weight {upper} is infeasible for J={job_type_count}: "
            f"J*upper={job_type_count * upper}<1."
        )


def read_experiment_metadata(path: Path) -> dict[str, Any]:
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise ValueError(f"Could not read experiment config: {path}")
    return {
        "simulation_duration": parser.getint("system", "simulation_duration"),
        "experiment_seed": parser.getint("system", "random_seed"),
        "iso_file_path": parser.get("iso", "iso_file_path"),
        "iso_signal_start_hour": parser.getint("iso", "iso_signal_start_hour"),
        "iso_signal_granularity": parser.getint("iso", "iso_signal_granularity"),
        "normalize_iso_signal": parser.getboolean("iso", "normalize_iso_signal"),
        "randomize_iso_start": parser.getboolean("iso", "randomize_iso_start"),
    }


def build_contexts(
    repo_root: Path,
    workload_configs: list[str],
    experiment_configs: list[str],
    utilizations: list[float],
    pbar_lower_factor: float,
    pbar_upper_factor: float,
    pr_upper_factor: float,
    r_lower: float,
    r_over_p_max: float,
    configured_weight_lower: float,
    configured_weight_upper: float,
) -> list[Context]:
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from peacsim.parsing.experiment_config_reader import ExperimentConfigReader
        from peacsim.parsing.job_profile_reader import JobProfileReader
    except Exception as exc:
        raise RuntimeError(
            f"Could not import FlexDC parsers from {src_dir}. "
            "Verify --repo-root points to the FlexDC repository root."
        ) from exc

    contexts: list[Context] = []
    seen_ids: set[str] = set()

    for workload_text in workload_configs:
        workload_path = resolve_path(workload_text, repo_root)
        if not workload_path.exists():
            raise FileNotFoundError(workload_path)
        job = JobProfileReader(str(workload_path))
        job_type_count = int(job.job_type_count)
        validate_configured_weight_domain(
            job_type_count,
            configured_weight_lower,
            configured_weight_upper,
        )

        pmin = min(float(x) for x in job.all_min_job_power.values()) / 1000.0
        pmax = max(float(x) for x in job.all_max_job_power.values()) / 1000.0
        p_lower = float(pbar_lower_factor * pmin)
        p_upper = float(pbar_upper_factor * pmax)
        pr_upper = float(pr_upper_factor * pmax)

        if p_lower <= 0 or p_upper < p_lower:
            raise ValueError(
                f"Invalid Pbar range for {workload_path.name}: "
                f"lower={p_lower}, upper={p_upper}"
            )

        for experiment_text in experiment_configs:
            experiment_path = resolve_path(experiment_text, repo_root)
            if not experiment_path.exists():
                raise FileNotFoundError(experiment_path)
            exp = ExperimentConfigReader(str(experiment_path))
            metadata = read_experiment_metadata(experiment_path)

            legacy_lower, legacy_upper = calculate_legacy_weight_bounds(
                job_type_count,
                int(exp.server_count),
            )
            effective_lower = max(legacy_lower, configured_weight_lower)
            effective_upper = min(legacy_upper, configured_weight_upper)
            validate_configured_weight_domain(job_type_count, effective_lower, effective_upper)

            for utilization in utilizations:
                if utilization <= 0 or utilization > 1:
                    raise ValueError(f"Utilization must be in (0,1], got {utilization}.")
                workload_name = workload_path.stem
                context_id = (
                    f"{workload_name}__N{int(exp.server_count)}__"
                    f"U{int(round(utilization * 100)):03d}"
                )
                if context_id in seen_ids:
                    raise ValueError(f"Duplicate context_id generated: {context_id}")
                seen_ids.add(context_id)

                context = Context(
                    context_id=context_id,
                    workload_name=workload_name,
                    workload_config=relative_or_absolute(workload_path, repo_root),
                    experiment_config=relative_or_absolute(experiment_path, repo_root),
                    server_count=int(exp.server_count),
                    utilization=float(utilization),
                    job_type_count=job_type_count,
                    pmin=pmin,
                    pmax=pmax,
                    p_lower=p_lower,
                    p_upper=p_upper,
                    pr_upper=pr_upper,
                    r_lower=float(r_lower),
                    r_over_p_max=float(r_over_p_max),
                    configured_weight_lower=float(configured_weight_lower),
                    configured_weight_upper=float(configured_weight_upper),
                    legacy_weight_lower=legacy_lower,
                    legacy_weight_upper=legacy_upper,
                    effective_weight_lower=effective_lower,
                    effective_weight_upper=effective_upper,
                    experiment_seed=int(metadata["experiment_seed"]),
                    simulation_duration=int(metadata["simulation_duration"]),
                    iso_file_path=str(metadata["iso_file_path"]),
                    iso_signal_start_hour=int(metadata["iso_signal_start_hour"]),
                    iso_signal_granularity=int(metadata["iso_signal_granularity"]),
                    normalize_iso_signal=bool(metadata["normalize_iso_signal"]),
                    randomize_iso_start=bool(metadata["randomize_iso_start"]),
                )
                if rmax(context, context.p_lower) < context.r_lower - PR_EPS:
                    raise ValueError(
                        f"No legal R interval at Pbar lower bound for {context_id}."
                    )
                contexts.append(context)

    return contexts


def rmax(context: Context, pbar: float) -> float:
    return min(
        float(context.r_over_p_max * pbar),
        float(context.pr_upper - pbar),
    )


def legal_pr(context: Context, pbar: float, reserve: float) -> bool:
    return (
        context.p_lower - PR_EPS <= pbar <= context.p_upper + PR_EPS
        and reserve >= context.r_lower - PR_EPS
        and reserve <= context.r_over_p_max * pbar + PR_EPS
        and pbar + reserve <= context.pr_upper + PR_EPS
        and pbar - reserve > -PR_EPS
    )


def pr_key(pbar: float, reserve: float, digits: int = 10) -> tuple[float, float]:
    return round(float(pbar), digits), round(float(reserve), digits)


def structured_pr_points(context: Context, p_points: int, r_points: int) -> list[dict[str, Any]]:
    if p_points <= 1 or r_points <= 1:
        raise ValueError("Structured P and R point counts must both exceed one.")
    output: list[dict[str, Any]] = []
    for p_index, pbar in enumerate(np.linspace(context.p_lower, context.p_upper, p_points)):
        pbar = float(pbar)
        max_r = rmax(context, pbar)
        if max_r < context.r_lower - PR_EPS:
            raise ValueError(
                f"No legal R values for {context.context_id} at Pbar={pbar}."
            )
        r_values = np.linspace(context.r_lower, max_r, r_points)
        for r_index, reserve in enumerate(r_values):
            reserve = float(reserve)
            if not legal_pr(context, pbar, reserve):
                raise ValueError(
                    f"Structured generator created an illegal pair: "
                    f"Pbar={pbar}, R={reserve}, context={context.context_id}"
                )
            output.append({
                "Pbar_kw_per_server": pbar,
                "R_kw_per_server": reserve,
                "pr_stratum": "structured_grid",
                "pr_stratum_detail": f"p{p_index:02d}_r{r_index:02d}",
            })
    if len(output) != p_points * r_points:
        raise AssertionError("Structured P/R count mismatch.")
    return output


def _midpoint_values(lower: float, upper: float, count: int) -> np.ndarray:
    if count <= 0:
        return np.asarray([], dtype=float)
    if upper < lower - PR_EPS:
        return np.asarray([], dtype=float)
    if abs(upper - lower) <= PR_EPS:
        return np.full(count, lower, dtype=float)
    return lower + (np.arange(count, dtype=float) + 0.5) / count * (upper - lower)


def _pick_even_unique(
    candidates: list[tuple[float, float]],
    count: int,
    context: Context,
    used: set[tuple[float, float]],
) -> list[tuple[float, float]]:
    valid: list[tuple[float, float]] = []
    local_seen: set[tuple[float, float]] = set()
    for pbar, reserve in candidates:
        key = pr_key(pbar, reserve)
        if key in used or key in local_seen:
            continue
        if not legal_pr(context, float(pbar), float(reserve)):
            continue
        local_seen.add(key)
        valid.append((float(pbar), float(reserve)))
    if len(valid) < count:
        raise RuntimeError(
            f"Boundary family has only {len(valid)} unique legal candidates; "
            f"requested {count} for {context.context_id}."
        )
    indices = np.linspace(0, len(valid) - 1, count).round().astype(int)
    selected: list[tuple[float, float]] = []
    for index in indices:
        point = valid[int(index)]
        if point not in selected:
            selected.append(point)
    if len(selected) < count:
        for point in valid:
            if point not in selected:
                selected.append(point)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError("Could not select requested unique boundary points.")
    return selected


def physical_boundary_points(
    context: Context,
    count: int,
    used: set[tuple[float, float]],
) -> list[dict[str, Any]]:
    if count != 32:
        raise ValueError(
            "Sweep V3 physical-boundary design is predeclared as exactly 32 points/context."
        )

    family_specs: list[tuple[str, int, list[tuple[float, float]]]] = []

    p_candidates = _midpoint_values(context.p_lower, context.p_upper, 256)
    family_specs.append((
        "r_floor",
        8,
        [(float(p), context.r_lower) for p in p_candidates],
    ))

    p_cap_upper = min(context.p_upper, context.pr_upper / (1.0 + context.r_over_p_max))
    p_cap_candidates = _midpoint_values(context.p_lower, p_cap_upper, 256)
    family_specs.append((
        "r_over_p_cap",
        8,
        [(float(p), context.r_over_p_max * float(p)) for p in p_cap_candidates],
    ))

    for label, pbar in (("p_lower", context.p_lower), ("p_upper", context.p_upper)):
        max_r = rmax(context, pbar)
        r_candidates = _midpoint_values(context.r_lower, max_r, 256)
        family_specs.append((
            label,
            4,
            [(pbar, float(reserve)) for reserve in r_candidates],
        ))

    pr_edge_p_lower = max(
        context.p_lower,
        context.pr_upper / (1.0 + context.r_over_p_max),
    )
    pr_edge_p_upper = min(
        context.p_upper,
        context.pr_upper - context.r_lower,
    )
    pr_edge_candidates = _midpoint_values(pr_edge_p_lower, pr_edge_p_upper, 256)
    family_specs.append((
        "pr_upper",
        8,
        [
            (float(p), context.pr_upper - float(p))
            for p in pr_edge_candidates
        ],
    ))

    output: list[dict[str, Any]] = []
    for family, family_count, candidates in family_specs:
        selected = _pick_even_unique(candidates, family_count, context, used)
        for pbar, reserve in selected:
            used.add(pr_key(pbar, reserve))
            output.append({
                "Pbar_kw_per_server": pbar,
                "R_kw_per_server": reserve,
                "pr_stratum": "physical_boundary",
                "pr_stratum_detail": family,
            })

    if len(output) != count:
        raise AssertionError(f"Boundary count mismatch: {len(output)} != {count}")
    return output


def sobol_interior_points(
    context: Context,
    count: int,
    seed: int,
    used: set[tuple[float, float]],
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    p_probe = np.linspace(context.p_lower, context.p_upper, 10001)
    global_r_max = max(rmax(context, float(pbar)) for pbar in p_probe)

    block_power = max(13, int(math.ceil(math.log2(max(count * 32, 2)))))
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    uv = sampler.random_base2(m=block_power)

    output: list[dict[str, Any]] = []
    local_seen: set[tuple[float, float]] = set()
    for u, v in uv:
        pbar = context.p_lower + float(u) * (context.p_upper - context.p_lower)
        reserve = context.r_lower + float(v) * (global_r_max - context.r_lower)
        key = pr_key(pbar, reserve)
        if key in used or key in local_seen:
            continue
        if not legal_pr(context, pbar, reserve):
            continue
        local_seen.add(key)
        output.append({
            "Pbar_kw_per_server": float(pbar),
            "R_kw_per_server": float(reserve),
            "pr_stratum": "sobol_interior",
            "pr_stratum_detail": "scrambled_area_rejection",
        })
        if len(output) == count:
            return output

    raise RuntimeError(
        f"Sobol rejection produced only {len(output)} unique legal points for "
        f"{context.context_id}; requested {count}."
    )


def validate_weight_vector(weights: Iterable[float], job_type_count: int, lower: float, upper: float) -> tuple[float, ...]:
    vector = tuple(float(value) for value in weights)
    if len(vector) != job_type_count:
        raise ValueError(f"Weight vector length {len(vector)} != J={job_type_count}: {vector}")
    if not np.isclose(sum(vector), 1.0, atol=1e-9):
        raise ValueError(f"Weight vector does not sum to one: sum={sum(vector)}, vector={vector}")
    if min(vector) < lower - WEIGHT_EPS or max(vector) > upper + WEIGHT_EPS:
        raise ValueError(f"Weight vector violates [{lower},{upper}]: {vector}")
    return vector


def _dedupe_weight_candidates(
    candidates: Iterable[tuple[str, str, Iterable[float]]],
    job_type_count: int,
    lower: float,
    upper: float,
) -> list[tuple[str, str, tuple[float, ...]]]:
    output: list[tuple[str, str, tuple[float, ...]]] = []
    seen: set[tuple[float, ...]] = set()
    for source, detail, weights in candidates:
        try:
            vector = validate_weight_vector(weights, job_type_count, lower, upper)
        except ValueError:
            continue
        key = tuple(round(value, 10) for value in vector)
        if key in seen:
            continue
        seen.add(key)
        output.append((source, detail, vector))
    return output


def lattice_weight_candidates(
    job_type_count: int,
    lower: float,
    upper: float,
    step: float,
) -> list[tuple[str, str, tuple[float, ...]]]:
    if step <= 0:
        raise ValueError("lattice step must be positive.")
    values = np.arange(lower, upper + step / 2.0, step)
    output: list[tuple[str, str, tuple[float, ...]]] = []

    def recurse(prefix: list[float], remaining: int, remaining_sum: float) -> None:
        if remaining == 1:
            final = remaining_sum
            if final < lower - WEIGHT_EPS or final > upper + WEIGHT_EPS:
                return
            vector = prefix + [final]
            try:
                validated = validate_weight_vector(
                    vector,
                    job_type_count,
                    lower,
                    upper,
                )
            except ValueError:
                return
            output.append(("lattice", f"step_{step:g}", validated))
            return

        min_remaining = (remaining - 1) * lower
        max_remaining = (remaining - 1) * upper
        for value in values:
            next_sum = remaining_sum - float(value)
            if next_sum < min_remaining - WEIGHT_EPS or next_sum > max_remaining + WEIGHT_EPS:
                continue
            recurse(prefix + [float(value)], remaining - 1, next_sum)

    recurse([], job_type_count, 1.0)
    return _dedupe_weight_candidates(output, job_type_count, lower, upper)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def stochastic_weight_candidates(
    job_type_count: int,
    lower: float,
    upper: float,
    seed: int,
    temperatures: list[float],
    alphas: list[float],
    draws_per_setting: int,
    sobol_draws: int,
) -> list[tuple[str, str, tuple[float, ...]]]:
    rng = np.random.default_rng(seed)
    candidates: list[tuple[str, str, Iterable[float]]] = []

    # Softmax candidates.
    for temperature in temperatures:
        if temperature <= 0:
            raise ValueError(f"Softmax temperature must be positive: {temperature}")
        for _ in range(draws_per_setting):
            scores = rng.normal(size=job_type_count)
            weights = softmax(scores / temperature)
            candidates.append(("softmax", f"temperature_{temperature:g}", weights))

    # Dirichlet candidates mapped to the lower-bounded simplex.
    residual_mass = 1.0 - job_type_count * lower
    for alpha in alphas:
        if alpha <= 0:
            raise ValueError(f"Dirichlet alpha must be positive: {alpha}")
        for _ in range(draws_per_setting):
            simplex = rng.dirichlet(np.full(job_type_count, alpha))
            weights = lower + residual_mass * simplex
            candidates.append(("dirichlet", f"alpha_{alpha:g}", weights))

    # Sobol candidates converted to simplex points by normalized exponential spacings.
    m = int(math.ceil(math.log2(max(sobol_draws, 2))))
    sampler = qmc.Sobol(d=job_type_count, scramble=True, seed=seed + 101)
    uv = sampler.random_base2(m=m)[:sobol_draws]
    for row in uv:
        exponentials = -np.log(np.clip(row, 1e-12, 1.0))
        simplex = exponentials / exponentials.sum()
        weights = lower + residual_mass * simplex
        candidates.append(("sobol_simplex", "normalized_exponential", weights))

    return _dedupe_weight_candidates(candidates, job_type_count, lower, upper)


def mandatory_weight_entries(job_type_count: int, lower: float, upper: float) -> list[WeightEntry]:
    if job_type_count < 3:
        raise ValueError("Sweep V3 ordered high/low anchors require at least three job types.")

    entries: list[WeightEntry] = []

    equal = tuple([1.0 / job_type_count] * job_type_count)
    validate_weight_vector(equal, job_type_count, lower, upper)
    entries.append(WeightEntry("W000", 0, "equal", "equal_weight_baseline", equal))

    next_index = 1
    for job_index in range(job_type_count):
        floor_vector = [(1.0 - lower) / (job_type_count - 1)] * job_type_count
        floor_vector[job_index] = lower
        entries.append(WeightEntry(
            f"W{next_index:03d}",
            next_index,
            "individual_anchor",
            f"job{job_index}_at_lower",
            validate_weight_vector(floor_vector, job_type_count, lower, upper),
        ))
        next_index += 1

        cap_vector = [(1.0 - upper) / (job_type_count - 1)] * job_type_count
        cap_vector[job_index] = upper
        entries.append(WeightEntry(
            f"W{next_index:03d}",
            next_index,
            "individual_anchor",
            f"job{job_index}_at_upper",
            validate_weight_vector(cap_vector, job_type_count, lower, upper),
        ))
        next_index += 1

    remainder = (1.0 - lower - upper) / (job_type_count - 2)
    if remainder < lower - WEIGHT_EPS or remainder > upper + WEIGHT_EPS:
        raise ValueError(
            f"Ordered high/low construction is infeasible: remaining weight={remainder}."
        )
    for low_job in range(job_type_count):
        for high_job in range(job_type_count):
            if low_job == high_job:
                continue
            vector = [remainder] * job_type_count
            vector[low_job] = lower
            vector[high_job] = upper
            entries.append(WeightEntry(
                f"W{next_index:03d}",
                next_index,
                "ordered_high_low",
                f"job{low_job}_lower_job{high_job}_upper",
                validate_weight_vector(vector, job_type_count, lower, upper),
            ))
            next_index += 1

    return entries


def build_weight_bank(
    job_type_count: int,
    lower: float,
    upper: float,
    total_count: int,
    lattice_step: float,
    temperatures: list[float],
    alphas: list[float],
    candidate_seed: int,
    draws_per_setting: int,
    sobol_draws: int,
) -> list[WeightEntry]:
    mandatory = mandatory_weight_entries(job_type_count, lower, upper)
    if len(mandatory) > total_count:
        raise ValueError(
            f"Mandatory weight bank already has {len(mandatory)} vectors, "
            f"exceeding requested total {total_count}."
        )

    candidate_pool = []
    candidate_pool.extend(
        lattice_weight_candidates(job_type_count, lower, upper, lattice_step)
    )
    candidate_pool.extend(
        stochastic_weight_candidates(
            job_type_count,
            lower,
            upper,
            candidate_seed,
            temperatures,
            alphas,
            draws_per_setting,
            sobol_draws,
        )
    )
    candidate_pool = _dedupe_weight_candidates(
        candidate_pool,
        job_type_count,
        lower,
        upper,
    )

    mandatory_matrix = np.asarray([entry.weights for entry in mandatory], dtype=float)
    mandatory_keys = {
        tuple(round(value, 10) for value in entry.weights)
        for entry in mandatory
    }

    filtered: list[tuple[str, str, tuple[float, ...]]] = []
    for item in candidate_pool:
        key = tuple(round(value, 10) for value in item[2])
        if key not in mandatory_keys:
            filtered.append(item)

    if len(filtered) < total_count - len(mandatory):
        raise RuntimeError(
            f"Weight candidate pool has only {len(filtered)} non-mandatory vectors."
        )

    candidate_matrix = np.asarray([item[2] for item in filtered], dtype=float)
    min_distance_sq = np.full(len(filtered), np.inf, dtype=float)
    for selected in mandatory_matrix:
        min_distance_sq = np.minimum(
            min_distance_sq,
            np.square(candidate_matrix - selected).sum(axis=1),
        )

    selected_indices: list[int] = []
    available = np.ones(len(filtered), dtype=bool)
    for _ in range(total_count - len(mandatory)):
        scores = np.where(available, min_distance_sq, -np.inf)
        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]):
            raise RuntimeError("Maximin selection exhausted the candidate pool.")
        selected_indices.append(index)
        available[index] = False
        selected = candidate_matrix[index]
        min_distance_sq = np.minimum(
            min_distance_sq,
            np.square(candidate_matrix - selected).sum(axis=1),
        )

    output = list(mandatory)
    next_index = len(output)
    for index in selected_indices:
        source, detail, vector = filtered[index]
        output.append(WeightEntry(
            f"W{next_index:03d}",
            next_index,
            "maximin_space_fill",
            f"selected_from_{source}__{detail}",
            vector,
        ))
        next_index += 1

    if len(output) != total_count:
        raise AssertionError("Final weight-bank count mismatch.")

    vectors = np.asarray([entry.weights for entry in output])
    if len({tuple(np.round(row, 10)) for row in vectors}) != len(output):
        raise AssertionError("Final weight bank contains duplicates.")

    return output


def largest_remainder_counts(total: int, fractions: tuple[float, ...]) -> list[int]:
    raw = np.asarray(fractions, dtype=float) * total
    floors = np.floor(raw).astype(int)
    remainder = int(total - floors.sum())
    order = np.argsort(-(raw - floors), kind="stable")
    for index in order[:remainder]:
        floors[int(index)] += 1
    return [int(value) for value in floors]


def assign_context_splits(base: pd.DataFrame, split_seed: int) -> pd.DataFrame:
    base = base.copy()
    group_columns = ["pr_stratum", "weight_family"]
    groups: dict[tuple[str, str], pd.DataFrame] = {}
    counts: dict[tuple[str, str], list[int]] = {}
    ideal: dict[tuple[str, str], np.ndarray] = {}

    for key, group in base.groupby(group_columns, sort=True):
        ordered = group.copy()
        ordered["__split_hash"] = ordered["base_plan_row_id"].map(
            lambda value: stable_hex("split", split_seed, value, length=32)
        )
        ordered = ordered.sort_values("__split_hash").drop(columns="__split_hash")
        groups[key] = ordered
        counts[key] = largest_remainder_counts(len(ordered), DEFAULT_SPLIT_FRACTIONS)
        ideal[key] = np.asarray(DEFAULT_SPLIT_FRACTIONS) * len(ordered)

    target = np.asarray(largest_remainder_counts(len(base), DEFAULT_SPLIT_FRACTIONS), dtype=int)
    current = np.sum(np.asarray(list(counts.values()), dtype=int), axis=0)

    while not np.array_equal(current, target):
        deficits = np.where(current < target)[0]
        excesses = np.where(current > target)[0]
        if len(deficits) == 0 or len(excesses) == 0:
            raise RuntimeError(
                f"Could not balance split totals: current={current}, target={target}"
            )
        destination = int(deficits[0])
        source = int(excesses[0])

        best_key: Optional[tuple[str, str]] = None
        best_delta = float("inf")
        for key, cell_counts in counts.items():
            if cell_counts[source] <= 0:
                continue
            before = (
                (cell_counts[source] - ideal[key][source]) ** 2
                + (cell_counts[destination] - ideal[key][destination]) ** 2
            )
            after = (
                (cell_counts[source] - 1 - ideal[key][source]) ** 2
                + (cell_counts[destination] + 1 - ideal[key][destination]) ** 2
            )
            delta = float(after - before)
            if delta < best_delta:
                best_delta = delta
                best_key = key
        if best_key is None:
            raise RuntimeError("Could not find a cell for split-count adjustment.")
        counts[best_key][source] -= 1
        counts[best_key][destination] += 1
        current[source] -= 1
        current[destination] += 1

    pieces: list[pd.DataFrame] = []
    for key, ordered in groups.items():
        cell_counts = counts[key]
        start = 0
        for split_name, split_count in zip(SPLIT_NAMES, cell_counts):
            part = ordered.iloc[start:start + split_count].copy()
            part["data_split"] = split_name
            pieces.append(part)
            start += split_count
        if start != len(ordered):
            raise AssertionError("Split assignment did not consume a group.")

    assigned = pd.concat(pieces, ignore_index=True)
    actual_counts = assigned["data_split"].value_counts()
    for split_name, expected in zip(SPLIT_NAMES, target):
        if int(actual_counts.get(split_name, 0)) != int(expected):
            raise AssertionError(
                f"Split count mismatch for {split_name}: "
                f"actual={actual_counts.get(split_name, 0)}, expected={expected}"
            )
    return assigned


def allocate_group_quotas(sizes: pd.Series, target_total: int) -> dict[Any, int]:
    if target_total < 0 or target_total > int(sizes.sum()):
        raise ValueError("Invalid replicate target total.")
    raw = sizes.astype(float) / float(sizes.sum()) * target_total
    floors = np.floor(raw).astype(int)
    remainder = int(target_total - floors.sum())
    order = (raw - floors).sort_values(ascending=False, kind="stable").index
    for key in order[:remainder]:
        floors.loc[key] += 1
    return {key: int(value) for key, value in floors.items()}


def choose_replicate_base_ids(
    base: pd.DataFrame,
    fraction: float,
    seed: int,
) -> set[str]:
    if fraction < 0 or fraction > 1:
        raise ValueError("replicate fraction must be in [0,1].")
    target = int(round(len(base) * fraction))
    group_columns = ["pr_stratum", "weight_family"]
    sizes = base.groupby(group_columns, sort=True).size()
    quotas = allocate_group_quotas(sizes, target)

    selected: set[str] = set()
    for key, group in base.groupby(group_columns, sort=True):
        quota = quotas[key]
        ranked = group.copy()
        ranked["__rep_hash"] = ranked["base_plan_row_id"].map(
            lambda value: stable_hex("replicate", seed, value, length=32)
        )
        ranked = ranked.sort_values("__rep_hash")
        selected.update(ranked.head(quota)["base_plan_row_id"].astype(str))

    if len(selected) != target:
        raise AssertionError(
            f"Replicate selection count mismatch: {len(selected)} != {target}"
        )
    return selected


def build_context_pr_points(
    context: Context,
    structured_p_points: int,
    structured_r_points: int,
    sobol_points: int,
    boundary_points: int,
    seed: int,
) -> list[dict[str, Any]]:
    structured = structured_pr_points(context, structured_p_points, structured_r_points)
    used = {
        pr_key(row["Pbar_kw_per_server"], row["R_kw_per_server"])
        for row in structured
    }
    boundaries = physical_boundary_points(context, boundary_points, used)
    used.update(
        pr_key(row["Pbar_kw_per_server"], row["R_kw_per_server"])
        for row in boundaries
    )
    sobol = sobol_interior_points(context, sobol_points, seed, used)
    all_points = structured + sobol + boundaries

    expected = structured_p_points * structured_r_points + sobol_points + boundary_points
    if len(all_points) != expected:
        raise AssertionError(f"P/R count mismatch: {len(all_points)} != {expected}")
    if len({pr_key(row["Pbar_kw_per_server"], row["R_kw_per_server"]) for row in all_points}) != expected:
        raise AssertionError(f"P/R points are not unique for {context.context_id}.")

    counters: dict[str, int] = {}
    for global_index, row in enumerate(all_points):
        stratum = str(row["pr_stratum"])
        local_index = counters.get(stratum, 0)
        counters[stratum] = local_index + 1
        row["pr_index"] = int(global_index)
        row["pr_index_within_stratum"] = int(local_index)
        row["pr_id"] = f"PR{global_index:03d}"
        row["R_over_P"] = float(row["R_kw_per_server"] / row["Pbar_kw_per_server"])
    return all_points


def make_plan(
    contexts: list[Context],
    design_version: str,
    weight_banks: dict[tuple[int, float, float], list[WeightEntry]],
    structured_p_points: int,
    structured_r_points: int,
    sobol_points: int,
    boundary_points: int,
    base_seed: int,
    replicate_fraction: float,
    replicate_seeds: list[int],
    split_seed: int,
) -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []

    for context in contexts:
        bank_key = (
            context.job_type_count,
            round(context.effective_weight_lower, 10),
            round(context.effective_weight_upper, 10),
        )
        weight_bank = weight_banks[bank_key]
        pr_points = build_context_pr_points(
            context,
            structured_p_points,
            structured_r_points,
            sobol_points,
            boundary_points,
            stable_seed(design_version, context.context_id, "sobol_pr"),
        )

        base_records: list[dict[str, Any]] = []
        for point in pr_points:
            for weight in weight_bank:
                base_plan_row_id = (
                    f"{design_version}__{context.context_id}__"
                    f"{point['pr_id']}__{weight.weight_id}"
                )
                record = {
                    "design_version": design_version,
                    "plan_row_id": f"{base_plan_row_id}__seed{base_seed:03d}",
                    "base_plan_row_id": base_plan_row_id,
                    "context_id": context.context_id,
                    "workload_name": context.workload_name,
                    "workload_config": context.workload_config,
                    "experiment_config": context.experiment_config,
                    "server_count": context.server_count,
                    "utilization": context.utilization,
                    "job_type_count": context.job_type_count,
                    "simulation_duration": context.simulation_duration,
                    "iso_file_path": context.iso_file_path,
                    "iso_signal_start_hour": context.iso_signal_start_hour,
                    "iso_signal_granularity": context.iso_signal_granularity,
                    "normalize_iso_signal": context.normalize_iso_signal,
                    "randomize_iso_start": context.randomize_iso_start,
                    "pmin_kw_per_server": context.pmin,
                    "pmax_kw_per_server": context.pmax,
                    "pbar_lower_bound_kw_per_server": context.p_lower,
                    "pbar_upper_bound_kw_per_server": context.p_upper,
                    "pr_upper_bound_kw_per_server": context.pr_upper,
                    "r_lower_bound_kw_per_server": context.r_lower,
                    "configured_r_over_p_max": context.r_over_p_max,
                    "Pbar_kw_per_server": point["Pbar_kw_per_server"],
                    "R_kw_per_server": point["R_kw_per_server"],
                    "R_over_P": point["R_over_P"],
                    "Pbar_plus_R": point["Pbar_kw_per_server"] + point["R_kw_per_server"],
                    "Pbar_minus_R": point["Pbar_kw_per_server"] - point["R_kw_per_server"],
                    "pr_id": point["pr_id"],
                    "pr_index": point["pr_index"],
                    "pr_index_within_stratum": point["pr_index_within_stratum"],
                    "pr_stratum": point["pr_stratum"],
                    "pr_stratum_detail": point["pr_stratum_detail"],
                    "weight_id": weight.weight_id,
                    "weight_index_within_pr": weight.weight_index,
                    "weight_family": weight.weight_family,
                    "weight_stratum": weight.weight_family,
                    "weight_detail": weight.weight_detail,
                    "configured_weight_lower": context.configured_weight_lower,
                    "configured_weight_upper": context.configured_weight_upper,
                    "legacy_weight_lower": context.legacy_weight_lower,
                    "legacy_weight_upper": context.legacy_weight_upper,
                    "effective_weight_lower": context.effective_weight_lower,
                    "effective_weight_upper": context.effective_weight_upper,
                    "weights": json.dumps([float(x) for x in weight.weights]),
                    "simulation_seed": int(base_seed),
                    "is_replicate": False,
                    "replicate_of_plan_row_id": "",
                }
                base_records.append(record)

        base = pd.DataFrame(base_records)
        base = assign_context_splits(base, split_seed)
        replicate_base_ids = choose_replicate_base_ids(
            base,
            replicate_fraction,
            stable_seed(design_version, context.context_id, "replicate_selection"),
        )

        replicas: list[dict[str, Any]] = []
        base_lookup = base.set_index("base_plan_row_id", drop=False)
        for base_id in sorted(replicate_base_ids):
            base_record = base_lookup.loc[base_id].to_dict()
            base_plan_row_id = str(base_record["base_plan_row_id"])
            base_plan_row = str(base_record["plan_row_id"])
            for repeat_seed in replicate_seeds:
                replica = dict(base_record)
                replica["plan_row_id"] = f"{base_plan_row_id}__seed{int(repeat_seed):03d}"
                replica["simulation_seed"] = int(repeat_seed)
                replica["is_replicate"] = True
                replica["replicate_of_plan_row_id"] = base_plan_row
                replicas.append(replica)

        context_plan = pd.concat([base, pd.DataFrame(replicas)], ignore_index=True, sort=False)
        all_rows.append(context_plan)

    plan = pd.concat(all_rows, ignore_index=True, sort=False)
    if plan["plan_row_id"].astype(str).duplicated().any():
        raise AssertionError("Generated plan has duplicate plan_row_id values.")
    return plan


def audit_plan(
    plan: pd.DataFrame,
    contexts: list[Context],
    expected_pr_points: int,
    expected_weights: int,
    base_seed: int,
    replicate_fraction: float,
    replicate_seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    context_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    hard_errors: list[str] = []

    context_lookup = {context.context_id: context for context in contexts}

    for context_id, group in plan.groupby("context_id", sort=True):
        context = context_lookup[str(context_id)]
        base = group[group["simulation_seed"].astype(int) == int(base_seed)].copy()
        repeats = group[group["simulation_seed"].astype(int) != int(base_seed)].copy()
        pr_count = base["pr_id"].nunique()
        weight_count = base["weight_id"].nunique()
        expected_base = expected_pr_points * expected_weights
        expected_repeat_bases = int(round(expected_base * replicate_fraction))
        expected_repeat_rows = expected_repeat_bases * len(replicate_seeds)

        if len(base) != expected_base:
            hard_errors.append(
                f"{context_id}: base rows={len(base)}, expected={expected_base}"
            )
        if len(repeats) != expected_repeat_rows:
            hard_errors.append(
                f"{context_id}: repeat rows={len(repeats)}, expected={expected_repeat_rows}"
            )
        if pr_count != expected_pr_points:
            hard_errors.append(
                f"{context_id}: P/R points={pr_count}, expected={expected_pr_points}"
            )
        if weight_count != expected_weights:
            hard_errors.append(
                f"{context_id}: weight vectors={weight_count}, expected={expected_weights}"
            )

        cross_counts = base.groupby("pr_id")["weight_id"].nunique()
        if not (cross_counts == expected_weights).all():
            hard_errors.append(f"{context_id}: not every P/R point has all weights.")

        p = base["Pbar_kw_per_server"].astype(float)
        r = base["R_kw_per_server"].astype(float)
        if (r > context.r_over_p_max * p + PR_EPS).any():
            hard_errors.append(f"{context_id}: R/P cap violation in plan.")
        if (p + r > context.pr_upper + PR_EPS).any():
            hard_errors.append(f"{context_id}: Pbar+R violation in plan.")

        for weights_text in base["weights"]:
            weights = json.loads(weights_text)
            try:
                validate_weight_vector(
                    weights,
                    context.job_type_count,
                    context.effective_weight_lower,
                    context.effective_weight_upper,
                )
            except ValueError as exc:
                hard_errors.append(f"{context_id}: {exc}")
                break

        split_counts = base["data_split"].value_counts().to_dict()
        context_rows.append({
            "context_id": context_id,
            "workload_name": context.workload_name,
            "server_count": context.server_count,
            "utilization": context.utilization,
            "base_rows": int(len(base)),
            "replicate_rows": int(len(repeats)),
            "total_rows": int(len(group)),
            "unique_pr_points": int(pr_count),
            "unique_weights": int(weight_count),
            "structured_pr_points": int(base.loc[base["pr_stratum"] == "structured_grid", "pr_id"].nunique()),
            "sobol_pr_points": int(base.loc[base["pr_stratum"] == "sobol_interior", "pr_id"].nunique()),
            "boundary_pr_points": int(base.loc[base["pr_stratum"] == "physical_boundary", "pr_id"].nunique()),
            "train_base_rows": int(split_counts.get("train", 0)),
            "validation_base_rows": int(split_counts.get("validation", 0)),
            "test_base_rows": int(split_counts.get("test", 0)),
            "weight_min": float(min(min(json.loads(text)) for text in base["weights"])),
            "weight_max": float(max(max(json.loads(text)) for text in base["weights"])),
            "r_over_p_min": float((r / p).min()),
            "r_over_p_max": float((r / p).max()),
        })

        for (pr_stratum, weight_family, data_split), cell in base.groupby(
            ["pr_stratum", "weight_family", "data_split"], sort=True
        ):
            split_rows.append({
                "context_id": context_id,
                "pr_stratum": pr_stratum,
                "weight_family": weight_family,
                "data_split": data_split,
                "base_rows": int(len(cell)),
            })

        for pr_stratum, cell in base.groupby("pr_stratum", sort=True):
            design_rows.append({
                "context_id": context_id,
                "design_dimension": "pr_stratum",
                "category": pr_stratum,
                "unique_pr_points": int(cell["pr_id"].nunique()),
                "base_rows": int(len(cell)),
            })
        for weight_family, cell in base.groupby("weight_family", sort=True):
            design_rows.append({
                "context_id": context_id,
                "design_dimension": "weight_family",
                "category": weight_family,
                "unique_pr_points": int(cell["pr_id"].nunique()),
                "base_rows": int(len(cell)),
            })

        # Every replicate group must inherit its base split and contain exactly all requested seeds.
        for base_id, replica_group in group[group["base_plan_row_id"].isin(replicate_base_ids_from_group(group))].groupby("base_plan_row_id"):
            seeds = set(int(x) for x in replica_group["simulation_seed"])
            expected_seeds = {int(base_seed), *[int(x) for x in replicate_seeds]}
            if seeds != expected_seeds:
                hard_errors.append(
                    f"{context_id}/{base_id}: replicate seeds={sorted(seeds)}, "
                    f"expected={sorted(expected_seeds)}"
                )
            if replica_group["data_split"].nunique() != 1:
                hard_errors.append(
                    f"{context_id}/{base_id}: seed variants cross data splits."
                )

    audit = {
        "status": "PASS" if not hard_errors else "FAIL",
        "plan_rows": int(len(plan)),
        "contexts": int(plan["context_id"].nunique()),
        "base_rows": int((plan["simulation_seed"].astype(int) == int(base_seed)).sum()),
        "replicate_rows": int((plan["simulation_seed"].astype(int) != int(base_seed)).sum()),
        "hard_errors": hard_errors,
        "split_counts_all_rows": {
            str(key): int(value)
            for key, value in plan["data_split"].value_counts().to_dict().items()
        },
        "pr_strata_all_rows": {
            str(key): int(value)
            for key, value in plan["pr_stratum"].value_counts().to_dict().items()
        },
        "weight_families_all_rows": {
            str(key): int(value)
            for key, value in plan["weight_family"].value_counts().to_dict().items()
        },
    }

    return (
        pd.DataFrame(context_rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(design_rows),
        audit,
    )


def replicate_base_ids_from_group(group: pd.DataFrame) -> set[str]:
    counts = group.groupby("base_plan_row_id")["simulation_seed"].nunique()
    return set(counts[counts > 1].index.astype(str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(SCRIPT_DIR))
    parser.add_argument("--workload-configs", default=None)
    parser.add_argument("--experiment-configs", default=None)
    parser.add_argument("--utilizations", default=None)
    parser.add_argument("--output-plan", default="flexdc_sweep_plan_v3.csv")
    parser.add_argument("--output-dir", default="flexdc_sweep_v3_plan_artifacts")
    parser.add_argument("--design-version", default="sweep_v3")

    parser.add_argument("--pbar-lower-factor", type=float, default=0.9)
    parser.add_argument("--pbar-upper-factor", type=float, default=1.0)
    parser.add_argument("--pr-upper-factor", type=float, default=1.2)
    parser.add_argument("--r-lower-kw-per-server", type=float, default=0.01)
    parser.add_argument("--r-over-p-max", type=float, default=0.6)
    parser.add_argument("--configured-weight-lower", type=float, default=0.15)
    parser.add_argument("--configured-weight-upper", type=float, default=0.45)

    parser.add_argument("--grid-p-points", type=int, default=16)
    parser.add_argument("--grid-r-points-per-pbar", type=int, default=10)
    parser.add_argument("--sobol-points", type=int, default=96)
    parser.add_argument("--boundary-points", type=int, default=32)

    parser.add_argument("--weights-per-pr", type=int, default=40)
    parser.add_argument("--weight-lattice-step", type=float, default=0.025)
    parser.add_argument("--softmax-temperatures", default=None)
    parser.add_argument("--dirichlet-alphas", default=None)
    parser.add_argument("--candidate-draws-per-setting", type=int, default=512)
    parser.add_argument("--candidate-sobol-draws", type=int, default=4096)
    parser.add_argument("--weight-candidate-seed", type=int, default=20260721)

    parser.add_argument("--base-seed", type=int, default=20)
    parser.add_argument("--replicate-fraction", type=float, default=0.05)
    parser.add_argument("--replicate-seeds", default="21,22")
    parser.add_argument("--split-seed", type=int, default=20260721)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not (repo_root / "src" / "peacsim").exists():
        raise ValueError(f"Not a FlexDC repository root: {repo_root}")

    workloads = parse_csv_items(args.workload_configs, DEFAULT_WORKLOAD_CONFIGS)
    experiments = parse_csv_items(args.experiment_configs, DEFAULT_EXPERIMENT_CONFIGS)
    utilizations = parse_float_items(args.utilizations, DEFAULT_UTILIZATIONS)
    temperatures = parse_float_items(
        args.softmax_temperatures,
        DEFAULT_SOFTMAX_TEMPERATURES,
    )
    alphas = parse_float_items(args.dirichlet_alphas, DEFAULT_DIRICHLET_ALPHAS)
    replicate_seeds = parse_int_items(args.replicate_seeds, [21, 22])

    if args.base_seed in replicate_seeds:
        raise ValueError("Base seed must not also be a replicate seed.")
    if len(set(replicate_seeds)) != len(replicate_seeds):
        raise ValueError("Replicate seeds must be unique.")
    if args.r_over_p_max <= 0:
        raise ValueError("--r-over-p-max must be positive.")
    if args.grid_p_points != 16 or args.grid_r_points_per_pbar != 10:
        print(
            "WARNING: Sweep V3 was predeclared with 16 Pbar x 10 R. "
            "You are overriding that design.",
            file=sys.stderr,
        )
    if args.boundary_points != 32:
        raise ValueError("Sweep V3 boundary design requires exactly 32 points/context.")

    contexts = build_contexts(
        repo_root,
        workloads,
        experiments,
        utilizations,
        args.pbar_lower_factor,
        args.pbar_upper_factor,
        args.pr_upper_factor,
        args.r_lower_kw_per_server,
        args.r_over_p_max,
        args.configured_weight_lower,
        args.configured_weight_upper,
    )

    bank_keys = sorted({
        (
            context.job_type_count,
            round(context.effective_weight_lower, 10),
            round(context.effective_weight_upper, 10),
        )
        for context in contexts
    })
    weight_banks: dict[tuple[int, float, float], list[WeightEntry]] = {}
    for job_type_count, lower, upper in bank_keys:
        weight_banks[(job_type_count, lower, upper)] = build_weight_bank(
            job_type_count,
            lower,
            upper,
            args.weights_per_pr,
            args.weight_lattice_step,
            temperatures,
            alphas,
            stable_seed(args.weight_candidate_seed, job_type_count, lower, upper),
            args.candidate_draws_per_setting,
            args.candidate_sobol_draws,
        )

    expected_pr_points = (
        args.grid_p_points * args.grid_r_points_per_pbar
        + args.sobol_points
        + args.boundary_points
    )

    output_plan = Path(args.output_plan)
    if not output_plan.is_absolute():
        output_plan = repo_root / output_plan
    output_plan = output_plan.resolve()
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    if output_plan.exists():
        output_plan.unlink()

    artifact_dir = Path(args.output_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = repo_root / artifact_dir
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    context_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    design_frames: list[pd.DataFrame] = []
    hard_errors: list[str] = []
    total_rows = 0
    total_base_rows = 0
    total_replicate_rows = 0
    aggregate_split_counts: dict[str, int] = {}
    aggregate_pr_strata: dict[str, int] = {}
    aggregate_weight_families: dict[str, int] = {}

    for context_index, context in enumerate(contexts):
        context_plan = make_plan(
            [context],
            args.design_version,
            weight_banks,
            args.grid_p_points,
            args.grid_r_points_per_pbar,
            args.sobol_points,
            args.boundary_points,
            args.base_seed,
            args.replicate_fraction,
            replicate_seeds,
            args.split_seed,
        )
        context_plan = context_plan.sort_values(
            ["context_id", "pr_index", "weight_index_within_pr", "simulation_seed"],
            kind="stable",
        ).reset_index(drop=True)

        context_plan.to_csv(
            output_plan,
            mode="w" if context_index == 0 else "a",
            header=context_index == 0,
            index=False,
        )

        context_summary, split_summary, design_summary, context_audit = audit_plan(
            context_plan,
            [context],
            expected_pr_points,
            args.weights_per_pr,
            args.base_seed,
            args.replicate_fraction,
            replicate_seeds,
        )
        context_frames.append(context_summary)
        split_frames.append(split_summary)
        design_frames.append(design_summary)
        hard_errors.extend(context_audit["hard_errors"])
        total_rows += int(context_audit["plan_rows"])
        total_base_rows += int(context_audit["base_rows"])
        total_replicate_rows += int(context_audit["replicate_rows"])

        for key, value in context_audit["split_counts_all_rows"].items():
            aggregate_split_counts[key] = aggregate_split_counts.get(key, 0) + int(value)
        for key, value in context_audit["pr_strata_all_rows"].items():
            aggregate_pr_strata[key] = aggregate_pr_strata.get(key, 0) + int(value)
        for key, value in context_audit["weight_families_all_rows"].items():
            aggregate_weight_families[key] = aggregate_weight_families.get(key, 0) + int(value)

        print(
            f"Prepared context {context_index + 1}/{len(contexts)}: "
            f"{context.context_id} ({len(context_plan):,} rows)"
        )
        del context_plan

    context_summary = pd.concat(context_frames, ignore_index=True)
    split_summary = pd.concat(split_frames, ignore_index=True)
    design_summary = pd.concat(design_frames, ignore_index=True)
    audit = {
        "status": "PASS" if not hard_errors else "FAIL",
        "plan_rows": int(total_rows),
        "contexts": int(len(contexts)),
        "base_rows": int(total_base_rows),
        "replicate_rows": int(total_replicate_rows),
        "hard_errors": hard_errors,
        "split_counts_all_rows": aggregate_split_counts,
        "pr_strata_all_rows": aggregate_pr_strata,
        "weight_families_all_rows": aggregate_weight_families,
    }

    context_summary_path = artifact_dir / "flexdc_sweep_plan_v3_context_summary.csv"
    split_summary_path = artifact_dir / "flexdc_sweep_plan_v3_split_summary.csv"
    design_summary_path = artifact_dir / "flexdc_sweep_plan_v3_design_summary.csv"
    weight_bank_path = artifact_dir / "flexdc_sweep_v3_weight_bank.csv"
    contexts_path = artifact_dir / "flexdc_sweep_v3_contexts.json"
    audit_path = artifact_dir / "flexdc_sweep_plan_v3_audit.json"

    context_summary.to_csv(context_summary_path, index=False)
    split_summary.to_csv(split_summary_path, index=False)
    design_summary.to_csv(design_summary_path, index=False)

    weight_rows = []
    for bank_key, bank in weight_banks.items():
        job_type_count, lower, upper = bank_key
        for entry in bank:
            weight_rows.append({
                "job_type_count": job_type_count,
                "effective_weight_lower": lower,
                "effective_weight_upper": upper,
                "weight_id": entry.weight_id,
                "weight_index": entry.weight_index,
                "weight_family": entry.weight_family,
                "weight_detail": entry.weight_detail,
                "weights": json.dumps([float(x) for x in entry.weights]),
            })
    pd.DataFrame(weight_rows).to_csv(weight_bank_path, index=False)
    contexts_path.write_text(
        json.dumps([asdict(context) for context in contexts], indent=2),
        encoding="utf-8",
    )

    audit.update({
        "generator": str(Path(__file__).resolve()),
        "repo_root": str(repo_root),
        "design_version": args.design_version,
        "arguments": vars(args),
        "expected_pr_points_per_context": expected_pr_points,
        "expected_weights_per_pr": args.weights_per_pr,
        "output_plan": str(output_plan),
        "output_plan_sha256": sha256_file(output_plan),
        "artifacts": {
            "context_summary": str(context_summary_path),
            "split_summary": str(split_summary_path),
            "design_summary": str(design_summary_path),
            "weight_bank": str(weight_bank_path),
            "contexts": str(contexts_path),
        },
    })
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("FlexDC Sweep V3 plan generation")
    print("--------------------------------")
    print(f"Status: {audit['status']}")
    print(f"Contexts: {len(contexts)}")
    print(f"P/R points per context: {expected_pr_points}")
    print(f"Weights per P/R point: {args.weights_per_pr}")
    print(f"Base rows: {audit['base_rows']:,}")
    print(f"Replicate rows: {audit['replicate_rows']:,}")
    print(f"Total rows: {audit['plan_rows']:,}")
    print(f"Plan: {output_plan}")
    print(f"Audit: {audit_path}")
    print()
    print(context_summary.to_string(index=False))

    if audit["hard_errors"]:
        print("\nHARD ERRORS:")
        for error in audit["hard_errors"]:
            print(" -", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
