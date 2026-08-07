#!/usr/bin/env python3
"""Generate a reusable, exact FlexDC sweep-plan CSV.

The default settings implement the current J=2 pairwise-profile experiment:
  * workload list supplied by am_generate_flexdc_j2_pair_workloads.py;
  * one 1000-server experiment config;
  * utilizations 0.60 and 0.80;
  * 64 unique P/R locations per workload/utilization context:
      8 x 4 structured grid = 32,
      24 scrambled-Sobol interior,
      8 physical-boundary points;
  * 16 deterministic weight vectors per P/R location;
  * no simulator-seed repeats by default.

The planner is generic: workload files, experiment files, utilization values,
P/R density, weight count, constraints, and replicate policy are all CLI
parameters. It writes the exact columns required by
am_run_flexdc_dataset_parallel_combo.py and am_data_extraction_wizard.py.

Cost constants are read only from [cost_function] and [dr_program] in the
specified simulated-annealing INI. Other SA initialization, boundaries, steps,
and annealing settings are intentionally ignored. The script copies the normal
FlexDC gradient config and overlays those canonical cost constants, producing a
runner-ready gradient config.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import math
import re
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

DEFAULT_EXPERIMENT_CONFIG = (
    "configs/experiment/new_iso/traditional_signal/generated_server_counts/"
    "exp_traditional_iso16_servers_1000.ini"
)
DEFAULT_SA_CONSTANTS_CONFIG = (
    "configs/optimization/simulated_annealing/"
    "SA_train_low_util_RSR_1000server.ini"
)
DEFAULT_BASE_GRADIENT_CONFIG = "configs/gradient_descent/gradient_descent.ini"
DEFAULT_OUTPUT_GRADIENT_CONFIG = (
    "configs/gradient_descent/gradient_descent_j2_pairwise_rsr.ini"
)
DEFAULT_UTILIZATIONS = [0.6, 0.8]
DEFAULT_SOFTMAX_TEMPERATURES = [0.5, 1.0, 2.0, 4.0]
DEFAULT_DIRICHLET_ALPHAS = [0.3, 1.0, 3.0, 10.0]


@dataclass(frozen=True)
class CostConstants:
    psi1: float
    psi2: float
    tracking_error_constraint: float
    beta: float
    rho: float
    qos_constraint: float
    program_type: str
    source_file: str


@dataclass(frozen=True)
class Context:
    context_id: str
    workload_name: str
    workload_config: str
    workload_type: str
    workload_display_name: str
    job_1_profile: str
    job_2_profile: str
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


def resolve_path(path_text: str | Path, repo_root: Path) -> Path:
    path = Path(path_text).expanduser()
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


def load_workload_list(
    *,
    workload_configs: Optional[str],
    workload_list_file: Optional[str],
    repo_root: Path,
) -> list[str]:
    if workload_configs and workload_list_file:
        raise ValueError("Use either --workload-configs or --workload-list-file, not both.")
    if workload_list_file:
        list_path = resolve_path(workload_list_file, repo_root)
        if not list_path.exists():
            raise FileNotFoundError(list_path)
        text = list_path.read_text(encoding="utf-8")
        items = [item.strip() for item in re.split(r"[,\r\n]+", text) if item.strip()]
    elif workload_configs:
        items = parse_csv_items(workload_configs, [])
    else:
        raise ValueError("Provide --workload-configs or --workload-list-file.")
    if not items:
        raise ValueError("No workload configs were supplied.")
    return items


def load_workload_manifest(path_text: Optional[str], repo_root: Path) -> dict[str, dict[str, str]]:
    if not path_text:
        return {}
    path = resolve_path(path_text, repo_root)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "workload_type",
            "workload_display_name",
            "job_1_profile",
            "job_2_profile",
            "output_file",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Workload manifest is missing columns: {sorted(missing)}")
        for row in reader:
            output_path = resolve_path(row["output_file"], repo_root)
            rows[str(output_path)] = {str(k): str(v) for k, v in row.items()}
            rows[output_path.name] = {str(k): str(v) for k, v in row.items()}
    return rows


def read_experiment_metadata(path: Path) -> dict[str, Any]:
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(path):
        raise ValueError(f"Could not read experiment config: {path}")
    required = {
        "system": ["simulation_duration", "random_seed"],
        "iso": [
            "iso_file_path",
            "iso_signal_start_hour",
            "iso_signal_granularity",
            "normalize_iso_signal",
            "randomize_iso_start",
        ],
    }
    for section, keys in required.items():
        if not parser.has_section(section):
            raise ValueError(f"Experiment config {path} lacks [{section}]")
        for key in keys:
            if not parser.has_option(section, key):
                raise ValueError(f"Experiment config {path} lacks [{section}] {key}")
    return {
        "simulation_duration": parser.getint("system", "simulation_duration"),
        "experiment_seed": parser.getint("system", "random_seed"),
        "iso_file_path": parser.get("iso", "iso_file_path"),
        "iso_signal_start_hour": parser.getint("iso", "iso_signal_start_hour"),
        "iso_signal_granularity": parser.getint("iso", "iso_signal_granularity"),
        "normalize_iso_signal": parser.getboolean("iso", "normalize_iso_signal"),
        "randomize_iso_start": parser.getboolean("iso", "randomize_iso_start"),
    }


def read_cost_constants(path: Path, repo_root: Path) -> CostConstants:
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(path):
        raise ValueError(f"Could not read SA constants config: {path}")
    if not parser.has_section("cost_function"):
        raise ValueError(f"{path} lacks [cost_function]")
    if not parser.has_section("dr_program"):
        raise ValueError(f"{path} lacks [dr_program]")
    required = [
        "psi1",
        "psi2",
        "tracking_error_constraint",
        "beta",
        "rho",
        "qos_constraint",
    ]
    for key in required:
        if not parser.has_option("cost_function", key):
            raise ValueError(f"{path} lacks [cost_function] {key}")
    if not parser.has_option("dr_program", "program_type"):
        raise ValueError(f"{path} lacks [dr_program] program_type")
    return CostConstants(
        psi1=parser.getfloat("cost_function", "psi1"),
        psi2=parser.getfloat("cost_function", "psi2"),
        tracking_error_constraint=parser.getfloat(
            "cost_function", "tracking_error_constraint"
        ),
        beta=parser.getfloat("cost_function", "beta"),
        rho=parser.getfloat("cost_function", "rho"),
        qos_constraint=parser.getfloat("cost_function", "qos_constraint"),
        program_type=parser.get("dr_program", "program_type").strip(),
        source_file=relative_or_absolute(path, repo_root),
    )


def set_case_insensitive_option(
    parser: configparser.ConfigParser,
    section: str,
    key: str,
    value: Any,
) -> None:
    if not parser.has_section(section):
        parser.add_section(section)
    existing = next(
        (option for option in parser.options(section) if option.lower() == key.lower()),
        key,
    )
    parser.set(section, existing, str(value))


def write_canonical_gradient_config(
    *,
    base_path: Path,
    output_path: Path,
    constants: CostConstants,
    overwrite: bool,
) -> None:
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if output_path.exists() and not overwrite:
        existing = configparser.ConfigParser(interpolation=None)
        existing.optionxform = str
        existing.read(output_path)
        checks = [
            ("calculate_gradient", "psi1", constants.psi1),
            ("calculate_gradient", "psi2", constants.psi2),
            (
                "calculate_gradient",
                "TRACKING_ERROR_CONSTRAINT",
                constants.tracking_error_constraint,
            ),
            ("calculate_gradient", "QOS_THRESHOLD", constants.qos_constraint),
            ("gradient_driver", "beta", constants.beta),
            ("gradient_driver", "rho", constants.rho),
            ("dr_program", "program_type", constants.program_type),
        ]
        for section, key, expected in checks:
            if not existing.has_option(section, key):
                raise FileExistsError(
                    f"{output_path} exists but lacks [{section}] {key}. Pass --overwrite."
                )
            actual = existing.get(section, key)
            if str(actual).strip().lower() != str(expected).strip().lower():
                try:
                    same = abs(float(actual) - float(expected)) <= 1e-12
                except (ValueError, TypeError):
                    same = False
                if not same:
                    raise FileExistsError(
                        f"{output_path} exists with [{section}] {key}={actual}, "
                        f"expected {expected}. Pass --overwrite."
                    )
        return

    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    with base_path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)

    set_case_insensitive_option(parser, "calculate_gradient", "psi1", constants.psi1)
    set_case_insensitive_option(parser, "calculate_gradient", "psi2", constants.psi2)
    set_case_insensitive_option(
        parser,
        "calculate_gradient",
        "TRACKING_ERROR_CONSTRAINT",
        constants.tracking_error_constraint,
    )
    set_case_insensitive_option(
        parser, "calculate_gradient", "QOS_THRESHOLD", constants.qos_constraint
    )
    set_case_insensitive_option(parser, "gradient_driver", "beta", constants.beta)
    set_case_insensitive_option(parser, "gradient_driver", "rho", constants.rho)
    set_case_insensitive_option(parser, "dr_program", "program_type", constants.program_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    from io import StringIO

    buffer = StringIO()
    parser.write(buffer, space_around_delimiters=True)
    header = (
        "# Generated by am_generate_flexdc_exact_sweep_plan.py\n"
        f"# Base config: {base_path}\n"
        f"# Cost constants source: {constants.source_file}\n"
        "# Only cost_function/dr_program constants were imported from the SA file.\n\n"
    )
    output_path.write_text(header + buffer.getvalue(), encoding="utf-8")


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
            f"N={server_count}: [{lower}, {upper}]"
        )
    return float(lower), float(upper)


def resolve_configured_weight_bounds(
    *,
    job_type_count: int,
    server_count: int,
    lower_factor_of_equal: float,
    upper_factor_of_equal: float,
    explicit_lower: Optional[float],
    explicit_upper: Optional[float],
) -> tuple[float, float, float, float]:
    equal = 1.0 / job_type_count
    requested_lower = (
        float(explicit_lower)
        if explicit_lower is not None
        else float(lower_factor_of_equal * equal)
    )
    requested_upper = (
        float(explicit_upper)
        if explicit_upper is not None
        else float(upper_factor_of_equal * equal)
    )
    if requested_lower < 0 or requested_upper > 1 or requested_lower > requested_upper:
        raise ValueError(
            f"Invalid requested weight interval [{requested_lower}, {requested_upper}]"
        )
    legacy_lower, legacy_upper = calculate_legacy_weight_bounds(
        job_type_count, server_count
    )
    effective_lower = max(legacy_lower, requested_lower)
    effective_upper = min(
        legacy_upper,
        requested_upper,
        1.0 - (job_type_count - 1) * effective_lower,
    )
    if job_type_count * effective_lower > 1.0 + WEIGHT_EPS:
        raise ValueError("Effective lower weight makes the simplex infeasible.")
    if job_type_count * effective_upper < 1.0 - WEIGHT_EPS:
        raise ValueError("Effective upper weight makes the simplex infeasible.")
    if effective_upper < effective_lower:
        raise ValueError("Effective weight interval is empty.")
    return requested_lower, requested_upper, effective_lower, effective_upper


def build_contexts(
    *,
    repo_root: Path,
    workload_configs: list[str],
    workload_manifest: dict[str, dict[str, str]],
    experiment_configs: list[str],
    utilizations: list[float],
    pbar_lower_factor: float,
    pbar_upper_factor: float,
    pr_upper_factor: float,
    r_lower: float,
    r_over_p_max: float,
    weight_lower_factor_of_equal: float,
    weight_upper_factor_of_equal: float,
    configured_weight_lower: Optional[float],
    configured_weight_upper: Optional[float],
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
            "Run from the FlexDC root or pass --repo-root correctly."
        ) from exc

    contexts: list[Context] = []
    seen_ids: set[str] = set()
    for workload_text in workload_configs:
        workload_path = resolve_path(workload_text, repo_root)
        if not workload_path.exists():
            raise FileNotFoundError(workload_path)
        job = JobProfileReader(str(workload_path))
        job_type_count = int(job.job_type_count)
        pmin = min(float(x) for x in job.all_min_job_power.values()) / 1000.0
        pmax = max(float(x) for x in job.all_max_job_power.values()) / 1000.0
        p_lower = float(pbar_lower_factor * pmin)
        p_upper = float(pbar_upper_factor * pmax)
        pr_upper = float(pr_upper_factor * pmax)
        if p_lower <= 0 or p_upper < p_lower:
            raise ValueError(
                f"Invalid Pbar range for {workload_path.name}: [{p_lower}, {p_upper}]"
            )
        metadata = workload_manifest.get(str(workload_path), workload_manifest.get(workload_path.name, {}))

        for experiment_text in experiment_configs:
            experiment_path = resolve_path(experiment_text, repo_root)
            if not experiment_path.exists():
                raise FileNotFoundError(experiment_path)
            exp = ExperimentConfigReader(str(experiment_path))
            exp_meta = read_experiment_metadata(experiment_path)
            requested_lower, requested_upper, effective_lower, effective_upper = (
                resolve_configured_weight_bounds(
                    job_type_count=job_type_count,
                    server_count=int(exp.server_count),
                    lower_factor_of_equal=weight_lower_factor_of_equal,
                    upper_factor_of_equal=weight_upper_factor_of_equal,
                    explicit_lower=configured_weight_lower,
                    explicit_upper=configured_weight_upper,
                )
            )
            legacy_lower, legacy_upper = calculate_legacy_weight_bounds(
                job_type_count, int(exp.server_count)
            )

            for utilization in utilizations:
                if utilization <= 0 or utilization > 1:
                    raise ValueError(f"Utilization must be in (0,1], got {utilization}")
                context_id = (
                    f"{workload_path.stem}__N{int(exp.server_count)}__"
                    f"U{int(round(utilization * 100)):03d}"
                )
                if context_id in seen_ids:
                    raise ValueError(f"Duplicate context_id generated: {context_id}")
                seen_ids.add(context_id)
                context = Context(
                    context_id=context_id,
                    workload_name=workload_path.stem,
                    workload_config=relative_or_absolute(workload_path, repo_root),
                    workload_type=metadata.get("workload_type", ""),
                    workload_display_name=metadata.get(
                        "workload_display_name", workload_path.stem
                    ),
                    job_1_profile=metadata.get("job_1_profile", ""),
                    job_2_profile=metadata.get("job_2_profile", ""),
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
                    configured_weight_lower=requested_lower,
                    configured_weight_upper=requested_upper,
                    legacy_weight_lower=legacy_lower,
                    legacy_weight_upper=legacy_upper,
                    effective_weight_lower=effective_lower,
                    effective_weight_upper=effective_upper,
                    experiment_seed=int(exp_meta["experiment_seed"]),
                    simulation_duration=int(exp_meta["simulation_duration"]),
                    iso_file_path=str(exp_meta["iso_file_path"]),
                    iso_signal_start_hour=int(exp_meta["iso_signal_start_hour"]),
                    iso_signal_granularity=int(exp_meta["iso_signal_granularity"]),
                    normalize_iso_signal=bool(exp_meta["normalize_iso_signal"]),
                    randomize_iso_start=bool(exp_meta["randomize_iso_start"]),
                )
                if rmax(context, context.p_lower) < context.r_lower - PR_EPS:
                    raise ValueError(
                        f"No legal R interval at Pbar lower bound for {context_id}"
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


def structured_pr_points(
    context: Context, p_points: int, r_points: int
) -> list[dict[str, Any]]:
    if p_points <= 0 or r_points <= 0:
        raise ValueError("Structured P and R counts must be positive.")
    output: list[dict[str, Any]] = []
    for p_index, pbar in enumerate(np.linspace(context.p_lower, context.p_upper, p_points)):
        pbar = float(pbar)
        max_r = rmax(context, pbar)
        if max_r < context.r_lower - PR_EPS:
            raise ValueError(f"No legal R values at Pbar={pbar} for {context.context_id}")
        for r_index, reserve in enumerate(np.linspace(context.r_lower, max_r, r_points)):
            reserve = float(reserve)
            if not legal_pr(context, pbar, reserve):
                raise AssertionError("Structured generator created an illegal P/R pair.")
            output.append(
                {
                    "Pbar_kw_per_server": pbar,
                    "R_kw_per_server": reserve,
                    "pr_stratum": "structured_grid",
                    "pr_stratum_detail": f"p{p_index:02d}_r{r_index:02d}",
                }
            )
    return output


def midpoint_values(lower: float, upper: float, count: int) -> np.ndarray:
    if count <= 0 or upper < lower - PR_EPS:
        return np.asarray([], dtype=float)
    if abs(upper - lower) <= PR_EPS:
        return np.full(count, lower, dtype=float)
    return lower + (np.arange(count, dtype=float) + 0.5) / count * (upper - lower)


def physical_boundary_points(
    context: Context,
    count: int,
    used: set[tuple[float, float]],
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    p_mid = midpoint_values(context.p_lower, context.p_upper, max(128, count * 16))
    families: list[tuple[str, list[tuple[float, float]]]] = [
        ("r_floor", [(float(p), context.r_lower) for p in p_mid]),
    ]
    cap_p_upper = min(
        context.p_upper,
        context.pr_upper / (1.0 + context.r_over_p_max),
    )
    cap_p = midpoint_values(context.p_lower, cap_p_upper, max(128, count * 16))
    families.append(
        (
            "r_over_p_cap",
            [(float(p), context.r_over_p_max * float(p)) for p in cap_p],
        )
    )
    for label, pbar in (("p_lower", context.p_lower), ("p_upper", context.p_upper)):
        values = midpoint_values(
            context.r_lower, rmax(context, pbar), max(64, count * 8)
        )
        families.append((label, [(pbar, float(r)) for r in values]))
    edge_lower = max(
        context.p_lower,
        context.pr_upper / (1.0 + context.r_over_p_max),
    )
    edge_upper = min(context.p_upper, context.pr_upper - context.r_lower)
    edge_p = midpoint_values(edge_lower, edge_upper, max(128, count * 16))
    families.append(
        (
            "pr_upper",
            [(float(p), context.pr_upper - float(p)) for p in edge_p],
        )
    )

    output: list[dict[str, Any]] = []
    cursors = [0] * len(families)
    while len(output) < count:
        made_progress = False
        for family_index, (family, candidates) in enumerate(families):
            while cursors[family_index] < len(candidates):
                pbar, reserve = candidates[cursors[family_index]]
                cursors[family_index] += 1
                key = pr_key(pbar, reserve)
                if key in used or not legal_pr(context, pbar, reserve):
                    continue
                used.add(key)
                output.append(
                    {
                        "Pbar_kw_per_server": float(pbar),
                        "R_kw_per_server": float(reserve),
                        "pr_stratum": "physical_boundary",
                        "pr_stratum_detail": family,
                    }
                )
                made_progress = True
                break
            if len(output) >= count:
                break
        if not made_progress:
            break
    if len(output) != count:
        raise RuntimeError(
            f"Could only create {len(output)}/{count} unique physical-boundary points "
            f"for {context.context_id}"
        )
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
    block_power = max(12, int(math.ceil(math.log2(max(count * 64, 2)))))
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    uv = sampler.random_base2(m=block_power)
    output: list[dict[str, Any]] = []
    local_seen: set[tuple[float, float]] = set()
    for u, v in uv:
        pbar = context.p_lower + float(u) * (context.p_upper - context.p_lower)
        reserve = context.r_lower + float(v) * (global_r_max - context.r_lower)
        key = pr_key(pbar, reserve)
        if key in used or key in local_seen or not legal_pr(context, pbar, reserve):
            continue
        local_seen.add(key)
        output.append(
            {
                "Pbar_kw_per_server": float(pbar),
                "R_kw_per_server": float(reserve),
                "pr_stratum": "sobol_interior",
                "pr_stratum_detail": "scrambled_area_rejection",
            }
        )
        if len(output) == count:
            return output
    raise RuntimeError(
        f"Sobol rejection produced {len(output)}/{count} points for {context.context_id}"
    )


def build_context_pr_points(
    *,
    context: Context,
    grid_p_points: int,
    grid_r_points_per_pbar: int,
    sobol_points: int,
    boundary_points: int,
    design_version: str,
) -> list[dict[str, Any]]:
    structured = structured_pr_points(context, grid_p_points, grid_r_points_per_pbar)
    used = {
        pr_key(row["Pbar_kw_per_server"], row["R_kw_per_server"])
        for row in structured
    }
    boundaries = physical_boundary_points(context, boundary_points, used)
    used.update(
        pr_key(row["Pbar_kw_per_server"], row["R_kw_per_server"])
        for row in boundaries
    )
    sobol = sobol_interior_points(
        context,
        sobol_points,
        stable_seed(design_version, context.context_id, "sobol_pr"),
        used,
    )
    points = structured + sobol + boundaries
    expected = grid_p_points * grid_r_points_per_pbar + sobol_points + boundary_points
    if len(points) != expected:
        raise AssertionError(f"P/R count mismatch: {len(points)} != {expected}")
    if len({pr_key(x["Pbar_kw_per_server"], x["R_kw_per_server"]) for x in points}) != expected:
        raise AssertionError("P/R points are not unique.")
    counters: dict[str, int] = {}
    for index, row in enumerate(points):
        stratum = str(row["pr_stratum"])
        local = counters.get(stratum, 0)
        counters[stratum] = local + 1
        row["pr_id"] = f"PR{index:03d}"
        row["pr_index"] = index
        row["pr_index_within_stratum"] = local
        row["R_over_P"] = float(row["R_kw_per_server"] / row["Pbar_kw_per_server"])
    return points


def validate_weight_vector(
    weights: Iterable[float],
    job_type_count: int,
    lower: float,
    upper: float,
) -> tuple[float, ...]:
    vector = tuple(float(value) for value in weights)
    if len(vector) != job_type_count:
        raise ValueError(f"Weight length {len(vector)} != J={job_type_count}")
    if not np.isclose(sum(vector), 1.0, atol=1e-9):
        raise ValueError(f"Weights do not sum to one: {vector}")
    if min(vector) < lower - WEIGHT_EPS or max(vector) > upper + WEIGHT_EPS:
        raise ValueError(f"Weights violate [{lower}, {upper}]: {vector}")
    return vector


def dedupe_weight_candidates(
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
        raise ValueError("--weight-lattice-step must be positive.")
    values = np.arange(lower, upper + step / 2.0, step)
    raw: list[tuple[str, str, Iterable[float]]] = []

    def recurse(prefix: list[float], remaining: int, remaining_sum: float) -> None:
        if remaining == 1:
            final = remaining_sum
            raw.append(("lattice", f"step_{step:g}", prefix + [final]))
            return
        min_rest = (remaining - 1) * lower
        max_rest = (remaining - 1) * upper
        for value in values:
            next_sum = remaining_sum - float(value)
            if next_sum < min_rest - WEIGHT_EPS or next_sum > max_rest + WEIGHT_EPS:
                continue
            recurse(prefix + [float(value)], remaining - 1, next_sum)

    recurse([], job_type_count, 1.0)
    return dedupe_weight_candidates(raw, job_type_count, lower, upper)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def stochastic_weight_candidates(
    *,
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
    raw: list[tuple[str, str, Iterable[float]]] = []
    for temperature in temperatures:
        if temperature <= 0:
            raise ValueError(f"Softmax temperature must be positive: {temperature}")
        for _ in range(draws_per_setting):
            raw.append(
                (
                    "softmax",
                    f"temperature_{temperature:g}",
                    softmax(rng.normal(size=job_type_count) / temperature),
                )
            )
    residual_mass = 1.0 - job_type_count * lower
    for alpha in alphas:
        if alpha <= 0:
            raise ValueError(f"Dirichlet alpha must be positive: {alpha}")
        for _ in range(draws_per_setting):
            simplex = rng.dirichlet(np.full(job_type_count, alpha))
            raw.append(
                (
                    "dirichlet",
                    f"alpha_{alpha:g}",
                    lower + residual_mass * simplex,
                )
            )
    m = int(math.ceil(math.log2(max(sobol_draws, 2))))
    sampler = qmc.Sobol(d=job_type_count, scramble=True, seed=seed + 101)
    for row in sampler.random_base2(m=m)[:sobol_draws]:
        exponentials = -np.log(np.clip(row, 1e-12, 1.0))
        simplex = exponentials / exponentials.sum()
        raw.append(
            (
                "sobol_simplex",
                "normalized_exponential",
                lower + residual_mass * simplex,
            )
        )
    return dedupe_weight_candidates(raw, job_type_count, lower, upper)


def mandatory_weight_candidates(
    job_type_count: int,
    lower: float,
    upper: float,
) -> list[tuple[str, str, tuple[float, ...]]]:
    raw: list[tuple[str, str, Iterable[float]]] = [
        ("equal", "equal_weight_baseline", [1.0 / job_type_count] * job_type_count)
    ]
    for job_index in range(job_type_count):
        low_vector = [(1.0 - lower) / (job_type_count - 1)] * job_type_count
        low_vector[job_index] = lower
        raw.append(("individual_anchor", f"job{job_index}_at_lower", low_vector))
        high_vector = [(1.0 - upper) / (job_type_count - 1)] * job_type_count
        high_vector[job_index] = upper
        raw.append(("individual_anchor", f"job{job_index}_at_upper", high_vector))
    return dedupe_weight_candidates(raw, job_type_count, lower, upper)


def build_weight_bank(
    *,
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
    mandatory = mandatory_weight_candidates(job_type_count, lower, upper)
    if len(mandatory) > total_count:
        raise ValueError(
            f"Mandatory weight vectors ({len(mandatory)}) exceed requested count {total_count}"
        )
    pool = lattice_weight_candidates(job_type_count, lower, upper, lattice_step)
    pool.extend(
        stochastic_weight_candidates(
            job_type_count=job_type_count,
            lower=lower,
            upper=upper,
            seed=candidate_seed,
            temperatures=temperatures,
            alphas=alphas,
            draws_per_setting=draws_per_setting,
            sobol_draws=sobol_draws,
        )
    )
    pool = dedupe_weight_candidates(pool, job_type_count, lower, upper)
    mandatory_keys = {tuple(round(x, 10) for x in item[2]) for item in mandatory}
    pool = [item for item in pool if tuple(round(x, 10) for x in item[2]) not in mandatory_keys]
    needed = total_count - len(mandatory)
    if len(pool) < needed:
        raise RuntimeError(f"Only {len(pool)} candidate weights available; need {needed}")

    mandatory_matrix = np.asarray([item[2] for item in mandatory], dtype=float)
    candidate_matrix = np.asarray([item[2] for item in pool], dtype=float)
    min_distance_sq = np.full(len(pool), np.inf)
    for selected in mandatory_matrix:
        min_distance_sq = np.minimum(
            min_distance_sq,
            np.square(candidate_matrix - selected).sum(axis=1),
        )
    available = np.ones(len(pool), dtype=bool)
    chosen_indices: list[int] = []
    for _ in range(needed):
        scores = np.where(available, min_distance_sq, -np.inf)
        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]):
            raise RuntimeError("Maximin weight selection exhausted the candidate pool.")
        chosen_indices.append(index)
        available[index] = False
        selected = candidate_matrix[index]
        min_distance_sq = np.minimum(
            min_distance_sq,
            np.square(candidate_matrix - selected).sum(axis=1),
        )

    selected_items = mandatory + [pool[index] for index in chosen_indices]
    output = [
        WeightEntry(
            weight_id=f"W{index:03d}",
            weight_index=index,
            weight_family=item[0] if index < len(mandatory) else "maximin_space_fill",
            weight_detail=item[1] if index < len(mandatory) else f"selected_from_{item[0]}__{item[1]}",
            weights=item[2],
        )
        for index, item in enumerate(selected_items)
    ]
    if len(output) != total_count:
        raise AssertionError("Weight-bank size mismatch.")
    return output


def choose_replicate_base_ids(
    base: pd.DataFrame,
    fraction: float,
    seed: int,
) -> set[str]:
    if fraction < 0 or fraction > 1:
        raise ValueError("--replicate-fraction must be in [0,1]")
    target = int(round(len(base) * fraction))
    if target == 0:
        return set()
    ranked = base.copy()
    ranked["__rep_hash"] = ranked["base_plan_row_id"].map(
        lambda value: stable_hex("replicate", seed, value, length=32)
    )
    return set(ranked.sort_values("__rep_hash").head(target)["base_plan_row_id"].astype(str))


def make_context_plan(
    *,
    context: Context,
    constants: CostConstants,
    design_version: str,
    weight_bank: list[WeightEntry],
    grid_p_points: int,
    grid_r_points_per_pbar: int,
    sobol_points: int,
    boundary_points: int,
    base_seed: int,
    replicate_fraction: float,
    replicate_seeds: list[int],
) -> pd.DataFrame:
    points = build_context_pr_points(
        context=context,
        grid_p_points=grid_p_points,
        grid_r_points_per_pbar=grid_r_points_per_pbar,
        sobol_points=sobol_points,
        boundary_points=boundary_points,
        design_version=design_version,
    )
    records: list[dict[str, Any]] = []
    for point in points:
        for weight in weight_bank:
            base_id = (
                f"{design_version}__{context.context_id}__"
                f"{point['pr_id']}__{weight.weight_id}"
            )
            records.append(
                {
                    "design_version": design_version,
                    "plan_row_id": f"{base_id}__seed{base_seed:03d}",
                    "base_plan_row_id": base_id,
                    "context_id": context.context_id,
                    "workload_name": context.workload_name,
                    "workload_type": context.workload_type,
                    "workload_display_name": context.workload_display_name,
                    "job_1_profile": context.job_1_profile,
                    "job_2_profile": context.job_2_profile,
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
                    "original_experiment_seed": context.experiment_seed,
                    "is_replicate": False,
                    "replicate_of_plan_row_id": "",
                    "data_split": "master_unassigned",
                    "benchmark_type": "j2_pairwise_master",
                    "cost_psi1": constants.psi1,
                    "cost_psi2": constants.psi2,
                    "tracking_error_constraint": constants.tracking_error_constraint,
                    "cost_beta": constants.beta,
                    "cost_rho": constants.rho,
                    "qos_constraint": constants.qos_constraint,
                    "dr_program_type": constants.program_type,
                    "cost_constants_source": constants.source_file,
                }
            )
    base = pd.DataFrame(records)
    repeat_ids = choose_replicate_base_ids(
        base,
        replicate_fraction,
        stable_seed(design_version, context.context_id, "replicate_selection"),
    )
    replicas: list[dict[str, Any]] = []
    if repeat_ids:
        lookup = base.set_index("base_plan_row_id", drop=False)
        for base_id in sorted(repeat_ids):
            original = lookup.loc[base_id].to_dict()
            original_plan_id = str(original["plan_row_id"])
            for repeat_seed in replicate_seeds:
                replica = dict(original)
                replica["plan_row_id"] = f"{base_id}__seed{repeat_seed:03d}"
                replica["simulation_seed"] = int(repeat_seed)
                replica["is_replicate"] = True
                replica["replicate_of_plan_row_id"] = original_plan_id
                replicas.append(replica)
    return pd.concat([base, pd.DataFrame(replicas)], ignore_index=True, sort=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(SCRIPT_DIR))
    parser.add_argument("--workload-configs", default=None)
    parser.add_argument("--workload-list-file", default=None)
    parser.add_argument("--workload-manifest", default=None)
    parser.add_argument("--experiment-configs", default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--utilizations", default="0.6,0.8")
    parser.add_argument("--output-plan", default="flexdc_sweep_plan_exact.csv")
    parser.add_argument("--output-dir", default="flexdc_exact_plan_artifacts")
    parser.add_argument("--design-version", default="exact_sweep_v1")

    parser.add_argument("--sa-constants-config", default=DEFAULT_SA_CONSTANTS_CONFIG)
    parser.add_argument("--base-gradient-config", default=DEFAULT_BASE_GRADIENT_CONFIG)
    parser.add_argument("--output-gradient-config", default=DEFAULT_OUTPUT_GRADIENT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--pbar-lower-factor", type=float, default=0.9)
    parser.add_argument("--pbar-upper-factor", type=float, default=1.0)
    parser.add_argument("--pr-upper-factor", type=float, default=1.2)
    parser.add_argument("--r-lower-kw-per-server", type=float, default=0.01)
    parser.add_argument("--r-over-p-max", type=float, default=0.6)

    parser.add_argument("--grid-p-points", type=int, default=8)
    parser.add_argument("--grid-r-points-per-pbar", type=int, default=4)
    parser.add_argument("--sobol-points", type=int, default=24)
    parser.add_argument("--boundary-points", type=int, default=8)

    parser.add_argument("--weights-per-pr", type=int, default=16)
    parser.add_argument("--weight-lower-factor-of-equal", type=float, default=0.6)
    parser.add_argument("--weight-upper-factor-of-equal", type=float, default=1.8)
    parser.add_argument("--configured-weight-lower", type=float, default=None)
    parser.add_argument("--configured-weight-upper", type=float, default=None)
    parser.add_argument("--weight-lattice-step", type=float, default=0.025)
    parser.add_argument("--softmax-temperatures", default="0.5,1,2,4")
    parser.add_argument("--dirichlet-alphas", default="0.3,1,3,10")
    parser.add_argument("--candidate-draws-per-setting", type=int, default=512)
    parser.add_argument("--candidate-sobol-draws", type=int, default=4096)
    parser.add_argument("--weight-candidate-seed", type=int, default=20260804)

    parser.add_argument("--base-seed", type=int, default=20)
    parser.add_argument("--replicate-fraction", type=float, default=0.0)
    parser.add_argument("--replicate-seeds", default="21,22")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not (repo_root / "src" / "peacsim").exists():
        raise ValueError(f"Not a FlexDC repository root: {repo_root}")

    workloads = load_workload_list(
        workload_configs=args.workload_configs,
        workload_list_file=args.workload_list_file,
        repo_root=repo_root,
    )
    workload_manifest = load_workload_manifest(args.workload_manifest, repo_root)
    experiments = parse_csv_items(args.experiment_configs, [DEFAULT_EXPERIMENT_CONFIG])
    utilizations = parse_float_items(args.utilizations, DEFAULT_UTILIZATIONS)
    temperatures = parse_float_items(
        args.softmax_temperatures, DEFAULT_SOFTMAX_TEMPERATURES
    )
    alphas = parse_float_items(args.dirichlet_alphas, DEFAULT_DIRICHLET_ALPHAS)
    replicate_seeds = parse_int_items(args.replicate_seeds, [21, 22])
    if args.base_seed in replicate_seeds:
        raise ValueError("Base seed cannot also be a replicate seed.")
    if len(set(replicate_seeds)) != len(replicate_seeds):
        raise ValueError("Replicate seeds must be unique.")
    if args.r_over_p_max <= 0:
        raise ValueError("--r-over-p-max must be positive.")

    sa_path = resolve_path(args.sa_constants_config, repo_root)
    constants = read_cost_constants(sa_path, repo_root)
    base_gradient_path = resolve_path(args.base_gradient_config, repo_root)
    output_gradient_path = resolve_path(args.output_gradient_config, repo_root)
    write_canonical_gradient_config(
        base_path=base_gradient_path,
        output_path=output_gradient_path,
        constants=constants,
        overwrite=args.overwrite,
    )

    contexts = build_contexts(
        repo_root=repo_root,
        workload_configs=workloads,
        workload_manifest=workload_manifest,
        experiment_configs=experiments,
        utilizations=utilizations,
        pbar_lower_factor=args.pbar_lower_factor,
        pbar_upper_factor=args.pbar_upper_factor,
        pr_upper_factor=args.pr_upper_factor,
        r_lower=args.r_lower_kw_per_server,
        r_over_p_max=args.r_over_p_max,
        weight_lower_factor_of_equal=args.weight_lower_factor_of_equal,
        weight_upper_factor_of_equal=args.weight_upper_factor_of_equal,
        configured_weight_lower=args.configured_weight_lower,
        configured_weight_upper=args.configured_weight_upper,
    )

    bank_keys = sorted(
        {
            (
                context.job_type_count,
                round(context.effective_weight_lower, 10),
                round(context.effective_weight_upper, 10),
            )
            for context in contexts
        }
    )
    weight_banks: dict[tuple[int, float, float], list[WeightEntry]] = {}
    for job_type_count, lower, upper in bank_keys:
        weight_banks[(job_type_count, lower, upper)] = build_weight_bank(
            job_type_count=job_type_count,
            lower=lower,
            upper=upper,
            total_count=args.weights_per_pr,
            lattice_step=args.weight_lattice_step,
            temperatures=temperatures,
            alphas=alphas,
            candidate_seed=stable_seed(
                args.weight_candidate_seed, job_type_count, lower, upper
            ),
            draws_per_setting=args.candidate_draws_per_setting,
            sobol_draws=args.candidate_sobol_draws,
        )

    output_plan = resolve_path(args.output_plan, repo_root)
    artifact_dir = resolve_path(args.output_dir, repo_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if output_plan.exists() and not args.overwrite:
        raise FileExistsError(f"{output_plan} exists. Pass --overwrite to replace it.")
    output_plan.parent.mkdir(parents=True, exist_ok=True)

    expected_pr = (
        args.grid_p_points * args.grid_r_points_per_pbar
        + args.sobol_points
        + args.boundary_points
    )
    context_summaries: list[dict[str, Any]] = []
    total_rows = 0
    total_base_rows = 0
    total_replicate_rows = 0

    for context_index, context in enumerate(contexts):
        bank_key = (
            context.job_type_count,
            round(context.effective_weight_lower, 10),
            round(context.effective_weight_upper, 10),
        )
        context_plan = make_context_plan(
            context=context,
            constants=constants,
            design_version=args.design_version,
            weight_bank=weight_banks[bank_key],
            grid_p_points=args.grid_p_points,
            grid_r_points_per_pbar=args.grid_r_points_per_pbar,
            sobol_points=args.sobol_points,
            boundary_points=args.boundary_points,
            base_seed=args.base_seed,
            replicate_fraction=args.replicate_fraction,
            replicate_seeds=replicate_seeds,
        )
        context_plan = context_plan.sort_values(
            ["pr_index", "weight_index_within_pr", "simulation_seed"],
            kind="stable",
        ).reset_index(drop=True)
        if context_plan["plan_row_id"].astype(str).duplicated().any():
            raise AssertionError(f"Duplicate plan_row_id in {context.context_id}")
        base = context_plan[~context_plan["is_replicate"].astype(bool)]
        repeats = context_plan[context_plan["is_replicate"].astype(bool)]
        expected_base = expected_pr * args.weights_per_pr
        if len(base) != expected_base:
            raise AssertionError(
                f"{context.context_id}: base rows={len(base)}, expected={expected_base}"
            )
        if base["pr_id"].nunique() != expected_pr:
            raise AssertionError(f"{context.context_id}: P/R count mismatch")
        if not (base.groupby("pr_id")["weight_id"].nunique() == args.weights_per_pr).all():
            raise AssertionError(f"{context.context_id}: not every P/R has every weight")

        context_plan.to_csv(
            output_plan,
            mode="w" if context_index == 0 else "a",
            header=context_index == 0,
            index=False,
        )
        context_summaries.append(
            {
                "context_id": context.context_id,
                "workload_name": context.workload_name,
                "workload_type": context.workload_type,
                "workload_display_name": context.workload_display_name,
                "job_1_profile": context.job_1_profile,
                "job_2_profile": context.job_2_profile,
                "server_count": context.server_count,
                "utilization": context.utilization,
                "job_type_count": context.job_type_count,
                "unique_pr_points": expected_pr,
                "weights_per_pr": args.weights_per_pr,
                "base_rows": len(base),
                "replicate_rows": len(repeats),
                "total_rows": len(context_plan),
                "effective_weight_lower": context.effective_weight_lower,
                "effective_weight_upper": context.effective_weight_upper,
            }
        )
        total_rows += len(context_plan)
        total_base_rows += len(base)
        total_replicate_rows += len(repeats)
        print(
            f"Prepared {context_index + 1}/{len(contexts)}: "
            f"{context.context_id} ({len(context_plan):,} rows)"
        )

    context_summary_path = artifact_dir / "exact_sweep_context_summary.csv"
    weight_bank_path = artifact_dir / "exact_sweep_weight_bank.csv"
    contexts_path = artifact_dir / "exact_sweep_contexts.json"
    constants_path = artifact_dir / "exact_sweep_cost_constants.json"
    audit_path = artifact_dir / "exact_sweep_audit.json"

    pd.DataFrame(context_summaries).to_csv(context_summary_path, index=False)
    weight_rows = []
    for (job_type_count, lower, upper), bank in weight_banks.items():
        for entry in bank:
            weight_rows.append(
                {
                    "job_type_count": job_type_count,
                    "effective_weight_lower": lower,
                    "effective_weight_upper": upper,
                    "weight_id": entry.weight_id,
                    "weight_index": entry.weight_index,
                    "weight_family": entry.weight_family,
                    "weight_detail": entry.weight_detail,
                    "weights": json.dumps([float(x) for x in entry.weights]),
                }
            )
    pd.DataFrame(weight_rows).to_csv(weight_bank_path, index=False)
    contexts_path.write_text(
        json.dumps([asdict(context) for context in contexts], indent=2) + "\n",
        encoding="utf-8",
    )
    constants_path.write_text(
        json.dumps(asdict(constants), indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "status": "PASS",
        "generator": str(Path(__file__).resolve()),
        "repo_root": str(repo_root),
        "design_version": args.design_version,
        "workload_count": len(workloads),
        "experiment_count": len(experiments),
        "utilizations": utilizations,
        "context_count": len(contexts),
        "pr_points_per_context": expected_pr,
        "weights_per_pr": args.weights_per_pr,
        "base_rows": total_base_rows,
        "replicate_rows": total_replicate_rows,
        "plan_rows": total_rows,
        "output_plan": str(output_plan),
        "output_plan_sha256": sha256_file(output_plan),
        "runner_gradient_config": relative_or_absolute(output_gradient_path, repo_root),
        "cost_constants": asdict(constants),
        "arguments": vars(args),
        "artifacts": {
            "context_summary": str(context_summary_path),
            "weight_bank": str(weight_bank_path),
            "contexts": str(contexts_path),
            "cost_constants": str(constants_path),
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    print("\nExact FlexDC sweep plan generated")
    print("---------------------------------")
    print("Status: PASS")
    print(f"Workloads: {len(workloads)}")
    print(f"Contexts: {len(contexts)}")
    print(f"P/R points per context: {expected_pr}")
    print(f"Weights per P/R point: {args.weights_per_pr}")
    print(f"Base rows: {total_base_rows:,}")
    print(f"Replicate rows: {total_replicate_rows:,}")
    print(f"Total rows: {total_rows:,}")
    print(f"Plan: {relative_or_absolute(output_plan, repo_root)}")
    print(f"Gradient config: {relative_or_absolute(output_gradient_path, repo_root)}")
    print(f"Audit: {relative_or_absolute(audit_path, repo_root)}")
    print("\nCanonical cost constants imported:")
    print(json.dumps(asdict(constants), indent=2))


if __name__ == "__main__":
    main()
