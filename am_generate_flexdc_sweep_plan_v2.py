#!/usr/bin/env python3
"""Generate a reproducible FlexDC v2 sweep plan.

The plan replaces the legacy 10x10 nested P/R grid and narrow fixed softmax
weight bank with:
  * area-balanced Sobol coverage of the legal P/R region;
  * explicit physical-boundary coverage;
  * pilot-guided decision-boundary / known-feasible coverage;
  * a six-vector weight block at every P/R point:
      uniform, deterministic/pilot anchor, and Dirichlet alpha 0.3/1/3/10;
  * deterministic seed replications for a small, documented subset.

The script does NOT run FlexDC. It writes one immutable CSV plan plus audit
artifacts. The edited data-extraction wizard executes this plan exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import qmc


SCRIPT_DIR = Path(__file__).resolve().parent


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
DEFAULT_DIRICHLET_ALPHAS = [0.3, 1.0, 3.0, 10.0]
DEFAULT_GAPS = [0.0, 0.005, 0.010, 0.020, 0.040]

TRACKING_THRESHOLD = 0.3
QOS_THRESHOLD = 0.1
PR_EPS = 1e-9


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
    w_lower: float
    w_upper: float
    experiment_seed: int


def parse_csv_items(value: Optional[str], default: Iterable[str]) -> list[str]:
    if value is None or str(value).strip() == "":
        return list(default)
    values = [x.strip() for x in str(value).split(",") if x.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def parse_float_items(value: Optional[str], default: Iterable[float]) -> list[float]:
    return [float(x) for x in parse_csv_items(value, [str(x) for x in default])]


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
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


def parse_json_vector(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Supports numpy-ish text and ordinary Python lists from older files.
        text = re.sub(r"np\.float64\(([^()]*)\)", r"\1", text)
        import ast
        parsed = ast.literal_eval(text)
    return [float(x) for x in parsed]


def calculate_weight_bounds(job_type_count: int, server_count: int) -> tuple[float, float]:
    equal = 1.0 / job_type_count
    relative_lower = 0.1 * equal
    relative_upper = 4.0 * equal
    server_lower = 1.0 / server_count
    lower = max(relative_lower, server_lower)
    upper_from_lower = 1.0 - (job_type_count - 1) * lower
    upper = min(relative_upper, upper_from_lower)
    if upper < lower:
        raise ValueError(
            f"Infeasible weight bounds for J={job_type_count}, servers={server_count}: "
            f"lower={lower}, upper={upper}"
        )
    return float(lower), float(upper)


def legal_pr(context: Context, pbar: float, reserve: float) -> bool:
    return (
        pbar >= context.p_lower - PR_EPS
        and pbar <= context.p_upper + PR_EPS
        and reserve >= context.r_lower - PR_EPS
        and reserve <= pbar + PR_EPS
        and pbar + reserve <= context.pr_upper + PR_EPS
    )


def rmax(context: Context, pbar: float) -> float:
    return min(float(pbar), float(context.pr_upper - pbar))


def dedupe_pr(points: Iterable[dict[str, Any]], digits: int = 9) -> list[dict[str, Any]]:
    seen: set[tuple[float, float]] = set()
    output: list[dict[str, Any]] = []
    for row in points:
        key = (round(float(row["Pbar_kw_per_server"]), digits), round(float(row["R_kw_per_server"]), digits))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def area_sobol_points(context: Context, count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    p_grid = np.linspace(context.p_lower, context.p_upper, 10001)
    global_r_max = max(rmax(context, float(p)) for p in p_grid)
    # Generate one power-of-two Sobol block so the sequence keeps its balance properties.
    # 4096 candidates is comfortably larger than the requested 120-point core.
    m = max(12, int(math.ceil(math.log2(max(count * 8, 2)))))
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    uv = sampler.random_base2(m=m)
    accepted: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for u, v in uv:
        p = context.p_lower + float(u) * (context.p_upper - context.p_lower)
        r = context.r_lower + float(v) * (global_r_max - context.r_lower)
        if not legal_pr(context, p, r):
            continue
        key = (round(p, 9), round(r, 9))
        if key in seen:
            continue
        seen.add(key)
        accepted.append({
            "Pbar_kw_per_server": p,
            "R_kw_per_server": r,
            "pr_stratum": "sobol_core",
            "pr_stratum_detail": "area_rejection_sobol",
            "pilot_anchor_weights": None,
            "pilot_anchor_type": "",
        })
        if len(accepted) >= count:
            return accepted
    raise RuntimeError(
        f"Sobol rejection produced only {len(accepted)} legal points for {context.context_id}; "
        f"requested {count}. Increase the candidate block size."
    )


def boundary_candidates(context: Context, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    groups: dict[str, list[tuple[float, float]]] = {
        "r_floor": [],
        "p_lower": [],
        "p_upper": [],
        "pr_upper": [],
        "p_equals_r_and_gaps": [],
    }

    # R floor across the full P range.
    for p in np.linspace(context.p_lower, context.p_upper, 12):
        groups["r_floor"].append((float(p), context.r_lower))

    # P lower / upper edges with multiple fractions of their valid R span.
    for label, p in [("p_lower", context.p_lower), ("p_upper", context.p_upper)]:
        rm = rmax(context, p)
        for frac in np.linspace(0.0, 1.0, 10):
            r = context.r_lower + float(frac) * max(0.0, rm - context.r_lower)
            groups[label].append((p, r))

    # P+R upper edge where it is the active upper boundary.
    p_start = max(context.p_lower, context.pr_upper / 2.0)
    for p in np.linspace(p_start, context.p_upper, 14):
        groups["pr_upper"].append((float(p), float(context.pr_upper - p)))

    # Economically attractive P≈R wall and logarithmically spaced interior gaps.
    p_eq_upper = min(context.p_upper, context.pr_upper / 2.0)
    if p_eq_upper >= context.p_lower:
        for gap in DEFAULT_GAPS:
            p_start_gap = max(context.p_lower, context.r_lower + gap)
            if p_start_gap > p_eq_upper:
                continue
            for p in np.linspace(p_start_gap, p_eq_upper, 8):
                groups["p_equals_r_and_gaps"].append((float(p), float(p - gap)))

    # Round-robin across physical edge types so one edge cannot monopolize the budget.
    cursors = {name: 0 for name in groups}
    names = list(groups)
    output: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    while len(output) < count:
        made_progress = False
        for name in names:
            items = groups[name]
            while cursors[name] < len(items):
                p, r = items[cursors[name]]
                cursors[name] += 1
                if not legal_pr(context, p, r):
                    continue
                key = (round(p, 9), round(r, 9))
                if key in seen:
                    continue
                seen.add(key)
                output.append({
                    "Pbar_kw_per_server": p,
                    "R_kw_per_server": r,
                    "pr_stratum": "physical_boundary",
                    "pr_stratum_detail": name,
                    "pilot_anchor_weights": None,
                    "pilot_anchor_type": "",
                })
                made_progress = True
                break
            if len(output) >= count:
                break
        if not made_progress:
            break
    if len(output) < count:
        # Fill any rare shortfall with additional space-filling legal points labelled boundary-fill.
        fillers = area_sobol_points(context, count - len(output), stable_seed(context.context_id, "boundary_fill"))
        for row in fillers:
            row["pr_stratum"] = "physical_boundary"
            row["pr_stratum_detail"] = "boundary_fill_sobol"
        output.extend(fillers)
    return output[:count]


def load_pilot_data(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Pilot results CSV not found: {path}")
    available = pd.read_csv(path, nrows=0).columns.tolist()
    preferred = [
        "Workload_Name", "server_count", "utilization", "Pbar_kw_per_server", "R_kw_per_server",
        "weights", "QoS_Delay_Probabilities", "QoS_Violation_Ratio", "Ctrack_Epsilon_90th",
        "Diagnostic_FullPaperObjective_Cost", "Simulator_RSR_Total_Cost",
    ]
    weight_cols = sorted(
        [c for c in available if re.fullmatch(r"Weight_\d+", str(c))],
        key=lambda c: int(c.split("_")[1]),
    )
    usecols = [c for c in preferred if c in available] + weight_cols
    pilot = pd.read_csv(path, usecols=usecols, low_memory=False)
    if "weights" not in pilot.columns:
        if not weight_cols:
            raise ValueError("Pilot CSV has neither a 'weights' column nor Weight_0...Weight_k columns.")
        pilot["weights"] = pilot[weight_cols].apply(
            lambda row: json.dumps([float(x) for x in row if not pd.isna(x)]), axis=1
        )
    if "Ctrack_Epsilon_90th" not in pilot.columns:
        raise ValueError("Pilot CSV must contain Ctrack_Epsilon_90th.")
    if "QoS_Delay_Probabilities" not in pilot.columns:
        raise ValueError("Pilot CSV must contain QoS_Delay_Probabilities.")
    pilot["max_pj"] = pilot["QoS_Delay_Probabilities"].map(
        lambda x: max(parse_json_vector(x)) if len(parse_json_vector(x)) else np.nan
    )
    if "QoS_Violation_Ratio" in pilot.columns:
        pilot["actual_feasible"] = (
            (pilot["Ctrack_Epsilon_90th"].astype(float) <= TRACKING_THRESHOLD)
            & (pilot["QoS_Violation_Ratio"].astype(float) <= QOS_THRESHOLD)
        )
    else:
        pilot["actual_feasible"] = (
            (pilot["Ctrack_Epsilon_90th"].astype(float) <= TRACKING_THRESHOLD)
            & (pilot["max_pj"].astype(float) <= QOS_THRESHOLD)
        )
    objective_col = (
        "Diagnostic_FullPaperObjective_Cost"
        if "Diagnostic_FullPaperObjective_Cost" in pilot.columns
        else "Simulator_RSR_Total_Cost"
    )
    pilot["pilot_objective"] = pilot[objective_col].astype(float)
    return pilot


def pilot_context_rows(pilot: Optional[pd.DataFrame], context: Context) -> pd.DataFrame:
    if pilot is None:
        return pd.DataFrame()
    return pilot[
        (pilot["Workload_Name"].astype(str) == context.workload_name)
        & (pilot["server_count"].astype(int) == context.server_count)
        & np.isclose(pilot["utilization"].astype(float), context.utilization)
    ].copy()


def pilot_guided_points(context: Context, pilot: Optional[pd.DataFrame], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    g = pilot_context_rows(pilot, context)
    if len(g) == 0:
        fillers = area_sobol_points(context, count, stable_seed(context.context_id, "pilot_missing"))
        for row in fillers:
            row["pr_stratum"] = "pilot_guided"
            row["pr_stratum_detail"] = "no_pilot_rows_sobol_fill"
        return fillers

    g = g.copy()
    g["pr_key"] = list(zip(g["Pbar_kw_per_server"].round(9), g["R_kw_per_server"].round(9)))

    selections: list[tuple[str, pd.DataFrame]] = []
    feasible = g[g["actual_feasible"]].sort_values("pilot_objective")
    if len(feasible):
        selections.append(("best_feasible", feasible))
    selections.append(("tracking_boundary", g.assign(_d=(g["Ctrack_Epsilon_90th"] - TRACKING_THRESHOLD).abs()).sort_values("_d")))
    selections.append(("qos_boundary", g.assign(_d=(g["max_pj"] - QOS_THRESHOLD).abs()).sort_values("_d")))
    selections.append(("low_objective", g.sort_values("pilot_objective")))

    # Allocate approximately evenly among available pilot categories.
    target_by_category = max(1, math.ceil(count / len(selections)))
    output: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for category, source in selections:
        added = 0
        # Prefer a row representative of each P/R pair; the first row carries an anchor weight.
        for _, row in source.drop_duplicates("pr_key", keep="first").iterrows():
            p = float(row["Pbar_kw_per_server"])
            r = float(row["R_kw_per_server"])
            key = (round(p, 9), round(r, 9))
            if key in seen or not legal_pr(context, p, r):
                continue
            seen.add(key)
            output.append({
                "Pbar_kw_per_server": p,
                "R_kw_per_server": r,
                "pr_stratum": "pilot_guided",
                "pr_stratum_detail": category,
                "pilot_anchor_weights": parse_json_vector(row["weights"]),
                "pilot_anchor_type": category,
                "pilot_anchor_actual_feasible": bool(row["actual_feasible"]),
                "pilot_anchor_actual_eps90": float(row["Ctrack_Epsilon_90th"]),
                "pilot_anchor_actual_max_pj": float(row["max_pj"]),
                "pilot_anchor_actual_objective": float(row["pilot_objective"]),
            })
            added += 1
            if added >= target_by_category or len(output) >= count:
                break
        if len(output) >= count:
            break

    if len(output) < count:
        # Continue through the most decision-relevant rows regardless of category.
        combined = pd.concat([x[1] for x in selections], ignore_index=False)
        for _, row in combined.drop_duplicates("pr_key", keep="first").iterrows():
            p, r = float(row["Pbar_kw_per_server"]), float(row["R_kw_per_server"])
            key = (round(p, 9), round(r, 9))
            if key in seen or not legal_pr(context, p, r):
                continue
            seen.add(key)
            output.append({
                "Pbar_kw_per_server": p,
                "R_kw_per_server": r,
                "pr_stratum": "pilot_guided",
                "pr_stratum_detail": "pilot_fill",
                "pilot_anchor_weights": parse_json_vector(row["weights"]),
                "pilot_anchor_type": "pilot_fill",
                "pilot_anchor_actual_feasible": bool(row["actual_feasible"]),
                "pilot_anchor_actual_eps90": float(row["Ctrack_Epsilon_90th"]),
                "pilot_anchor_actual_max_pj": float(row["max_pj"]),
                "pilot_anchor_actual_objective": float(row["pilot_objective"]),
            })
            if len(output) >= count:
                break

    if len(output) < count:
        fillers = area_sobol_points(context, count - len(output), stable_seed(context.context_id, "pilot_fill_sobol"))
        for row in fillers:
            row["pr_stratum"] = "pilot_guided"
            row["pr_stratum_detail"] = "pilot_shortfall_sobol_fill"
        output.extend(fillers)
    return output[:count]


def deterministic_anchor_bank(context: Context) -> list[tuple[str, list[float]]]:
    j = context.job_type_count
    lower, upper = context.w_lower, context.w_upper
    bank: list[tuple[str, list[float]]] = []
    for idx in range(j):
        floor_vec = [(1.0 - lower) / (j - 1)] * j
        floor_vec[idx] = lower
        if max(floor_vec) <= upper + 1e-9:
            bank.append((f"job{idx}_at_floor", floor_vec))

        near_cap = min(upper, 1.0 - (j - 1) * lower)
        cap_vec = [(1.0 - near_cap) / (j - 1)] * j
        cap_vec[idx] = near_cap
        if min(cap_vec) >= lower - 1e-9:
            bank.append((f"job{idx}_near_cap", cap_vec))
    return bank


def dirichlet_weight(context: Context, alpha: float, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    for _ in range(100000):
        simplex = rng.dirichlet(np.full(context.job_type_count, alpha, dtype=float))
        weights = context.w_lower + (1.0 - context.job_type_count * context.w_lower) * simplex
        if np.max(weights) <= context.w_upper + 1e-12:
            weights = weights / weights.sum()
            return [float(x) for x in weights]
    raise RuntimeError(
        f"Could not draw bounded Dirichlet weights for {context.context_id}, alpha={alpha}."
    )


def validate_weights(context: Context, weights: list[float]) -> None:
    if len(weights) != context.job_type_count:
        raise ValueError(f"Weight length mismatch for {context.context_id}: {weights}")
    if not np.isclose(sum(weights), 1.0, atol=1e-8):
        raise ValueError(f"Weights do not sum to one: {weights}")
    if min(weights) < context.w_lower - 1e-8 or max(weights) > context.w_upper + 1e-8:
        raise ValueError(
            f"Weights outside bounds [{context.w_lower}, {context.w_upper}]: {weights}"
        )


def build_contexts(
    repo_root: Path,
    workload_configs: list[str],
    experiment_configs: list[str],
    utilizations: list[float],
    pbar_lower_factor: float,
    pbar_upper_factor: float,
    pr_upper_factor: float,
    r_lower: float,
) -> list[Context]:
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        from peacsim.parsing.job_profile_reader import JobProfileReader
        from peacsim.parsing.experiment_config_reader import ExperimentConfigReader
    except Exception as exc:
        raise RuntimeError(
            f"Could not import FlexDC parsers from {src_dir}. "
            "Verify --repo-root points to the FlexDC repository root."
        ) from exc

    contexts: list[Context] = []
    for workload_text in workload_configs:
        workload_path = resolve_path(workload_text, repo_root)
        if not workload_path.exists():
            raise FileNotFoundError(workload_path)
        job = JobProfileReader(str(workload_path))
        pmin = min(float(x) for x in job.all_min_job_power.values()) / 1000.0
        pmax = max(float(x) for x in job.all_max_job_power.values()) / 1000.0
        p_lower = pbar_lower_factor * pmin
        p_upper = pbar_upper_factor * pmax
        pr_upper = pr_upper_factor * pmax

        for exp_text in experiment_configs:
            exp_path = resolve_path(exp_text, repo_root)
            if not exp_path.exists():
                raise FileNotFoundError(exp_path)
            exp = ExperimentConfigReader(str(exp_path))
            w_lower, w_upper = calculate_weight_bounds(job.job_type_count, exp.server_count)
            for utilization in utilizations:
                workload_name = workload_path.stem
                context_id = f"{workload_name}__N{exp.server_count}__U{int(round(utilization * 100)):03d}"
                contexts.append(Context(
                    context_id=context_id,
                    workload_name=workload_name,
                    workload_config=relative_or_absolute(workload_path, repo_root),
                    experiment_config=relative_or_absolute(exp_path, repo_root),
                    server_count=int(exp.server_count),
                    utilization=float(utilization),
                    job_type_count=int(job.job_type_count),
                    pmin=pmin,
                    pmax=pmax,
                    p_lower=p_lower,
                    p_upper=p_upper,
                    pr_upper=pr_upper,
                    r_lower=float(r_lower),
                    w_lower=w_lower,
                    w_upper=w_upper,
                    experiment_seed=int(exp.random_seed),
                ))
    return contexts


def make_plan_rows(
    contexts: list[Context],
    pilot: Optional[pd.DataFrame],
    design_version: str,
    core_count: int,
    boundary_count: int,
    pilot_count: int,
    dirichlet_alphas: list[float],
    base_seed: int,
    replicate_fraction: float,
    replicate_seeds: list[int],
) -> pd.DataFrame:
    base_rows: list[dict[str, Any]] = []

    for context in contexts:
        pr_points = []
        pr_points.extend(area_sobol_points(context, core_count, stable_seed(design_version, context.context_id, "core")))
        pr_points.extend(boundary_candidates(context, boundary_count))
        pr_points.extend(pilot_guided_points(context, pilot, pilot_count))
        pr_points = dedupe_pr(pr_points)
        expected_pr = core_count + boundary_count + pilot_count
        # Different strata can intentionally touch the same physical boundary. Fill any
        # cross-stratum duplicate shortfall with additional legal Sobol points so each
        # context still has the exact planned budget.
        fill_attempt = 0
        while len(pr_points) < expected_pr:
            shortfall = expected_pr - len(pr_points)
            fillers = area_sobol_points(
                context,
                shortfall + 16,
                stable_seed(design_version, context.context_id, "cross_stratum_fill", fill_attempt),
            )
            for row in fillers:
                row["pr_stratum"] = "sobol_core"
                row["pr_stratum_detail"] = "cross_stratum_dedupe_fill"
            pr_points = dedupe_pr(pr_points + fillers)[:expected_pr]
            fill_attempt += 1
            if fill_attempt > 20:
                raise RuntimeError(f"Could not fill unique P/R budget for {context.context_id}")
        if len(pr_points) != expected_pr:
            raise RuntimeError(
                f"{context.context_id}: expected {expected_pr} unique P/R points, got {len(pr_points)}"
            )

        anchor_bank = deterministic_anchor_bank(context)
        if not anchor_bank:
            raise RuntimeError(f"No deterministic anchors for {context.context_id}")

        for pr_index, pr in enumerate(pr_points):
            p = float(pr["Pbar_kw_per_server"])
            r = float(pr["R_kw_per_server"])
            if not legal_pr(context, p, r):
                raise RuntimeError(f"Generated illegal P/R for {context.context_id}: {(p, r)}")

            weight_specs: list[tuple[str, Optional[float], str, list[float]]] = []
            equal = [1.0 / context.job_type_count] * context.job_type_count
            weight_specs.append(("equal", None, "uniform", equal))

            pilot_anchor = pr.get("pilot_anchor_weights")
            if pilot_anchor is not None:
                validate_weights(context, pilot_anchor)
                weight_specs.append(("pilot_anchor", None, pr.get("pilot_anchor_type", "pilot_anchor"), pilot_anchor))
            else:
                anchor_name, anchor = anchor_bank[pr_index % len(anchor_bank)]
                weight_specs.append(("deterministic_anchor", None, anchor_name, anchor))

            for alpha in dirichlet_alphas:
                weights = dirichlet_weight(
                    context,
                    alpha,
                    stable_seed(design_version, context.context_id, pr_index, "dirichlet", alpha),
                )
                weight_specs.append(("dirichlet", float(alpha), "", weights))

            for weight_index, (weight_stratum, alpha, anchor_type, weights) in enumerate(weight_specs):
                validate_weights(context, weights)
                row_id_payload = (
                    design_version, context.context_id, pr_index, weight_index,
                    round(p, 12), round(r, 12), json.dumps(weights, sort_keys=True), base_seed,
                )
                row_hash = hashlib.sha256("|".join(map(str, row_id_payload)).encode()).hexdigest()[:16]
                plan_row_id = f"{context.context_id}__pr{pr_index:03d}__w{weight_index:02d}__{row_hash}"
                base_rows.append({
                    "design_version": design_version,
                    "plan_row_id": plan_row_id,
                    "base_plan_row_id": plan_row_id,
                    "context_id": context.context_id,
                    "workload_name": context.workload_name,
                    "workload_config": context.workload_config,
                    "experiment_config": context.experiment_config,
                    "server_count": context.server_count,
                    "utilization": context.utilization,
                    "job_type_count": context.job_type_count,
                    "Pmin_kw_per_server": context.pmin,
                    "Pmax_kw_per_server": context.pmax,
                    "Pbar_lower_bound_kw_per_server": context.p_lower,
                    "Pbar_upper_bound_kw_per_server": context.p_upper,
                    "PR_upper_bound_kw_per_server": context.pr_upper,
                    "R_lower_bound_kw_per_server": context.r_lower,
                    "Pbar_kw_per_server": p,
                    "R_kw_per_server": r,
                    "P_plus_R_kw_per_server": p + r,
                    "P_minus_R_kw_per_server": p - r,
                    "R_max_for_this_Pbar_kw_per_server": rmax(context, p),
                    "R_fraction_of_max": (r - context.r_lower) / max(rmax(context, p) - context.r_lower, 1e-12),
                    "pr_stratum": pr["pr_stratum"],
                    "pr_stratum_detail": pr["pr_stratum_detail"],
                    "weight_index_within_pr": weight_index,
                    "weight_stratum": weight_stratum,
                    "dirichlet_alpha": alpha if alpha is not None else "",
                    "anchor_type": anchor_type,
                    "Weight_Final_Lower_Bound": context.w_lower,
                    "Weight_Final_Upper_Bound": context.w_upper,
                    "weights": json.dumps([float(x) for x in weights]),
                    "simulation_seed": int(base_seed),
                    "original_experiment_seed": context.experiment_seed,
                    "is_replicate": False,
                    "pilot_anchor_actual_feasible": pr.get("pilot_anchor_actual_feasible", "") if weight_stratum == "pilot_anchor" else "",
                    "pilot_anchor_actual_eps90": pr.get("pilot_anchor_actual_eps90", "") if weight_stratum == "pilot_anchor" else "",
                    "pilot_anchor_actual_max_pj": pr.get("pilot_anchor_actual_max_pj", "") if weight_stratum == "pilot_anchor" else "",
                    "pilot_anchor_actual_objective": pr.get("pilot_anchor_actual_objective", "") if weight_stratum == "pilot_anchor" else "",
                })

    base_df = pd.DataFrame(base_rows)
    if base_df["plan_row_id"].duplicated().any():
        raise RuntimeError("Duplicate plan_row_id values in base plan.")

    # Choose a reproducible replicate subset. Prioritize pilot-guided and physical-boundary rows,
    # then fill from the full plan. Duplicating each selected row for each extra seed means a 5%
    # selected base subset adds 10% more runs when two replicate seeds are used.
    base_df["_rep_priority"] = base_df["pr_stratum"].map({
        "pilot_guided": 0,
        "physical_boundary": 1,
        "sobol_core": 2,
    }).fillna(3)
    base_df["_rep_hash"] = base_df["plan_row_id"].map(
        lambda x: stable_seed(design_version, x, "replicate_select")
    )
    # Select the same fraction within every context so noise coverage is not
    # accidentally concentrated in only a few workloads/server counts.
    chosen_parts = []
    for _, group in base_df.groupby("context_id", sort=True):
        count = int(round(len(group) * replicate_fraction))
        chosen_parts.append(group.sort_values(["_rep_priority", "_rep_hash"]).head(count))
    chosen = pd.concat(chosen_parts, ignore_index=False) if chosen_parts else base_df.iloc[0:0]

    replicate_rows: list[dict[str, Any]] = []
    for _, base in chosen.iterrows():
        for seed in replicate_seeds:
            row = base.drop(labels=["_rep_priority", "_rep_hash"]).to_dict()
            row["simulation_seed"] = int(seed)
            row["is_replicate"] = True
            row["plan_row_id"] = f"{base['plan_row_id']}__seed{seed}"
            replicate_rows.append(row)

    base_df = base_df.drop(columns=["_rep_priority", "_rep_hash"])
    final = pd.concat([base_df, pd.DataFrame(replicate_rows)], ignore_index=True, sort=False)
    if final["plan_row_id"].duplicated().any():
        raise RuntimeError("Duplicate plan_row_id values after replicate expansion.")
    return final


def audit_plan(plan: pd.DataFrame, contexts: list[Context]) -> tuple[pd.DataFrame, dict[str, Any]]:
    context_summary = plan.groupby("context_id").agg(
        total_rows=("plan_row_id", "size"),
        base_rows=("is_replicate", lambda s: int((~s.astype(bool)).sum())),
        replicate_rows=("is_replicate", lambda s: int(s.astype(bool).sum())),
        unique_p=("Pbar_kw_per_server", "nunique"),
        unique_pr=("Pbar_kw_per_server", lambda s: 0),  # replaced below
        min_weight=("weights", lambda s: min(min(parse_json_vector(x)) for x in s)),
        max_weight=("weights", lambda s: max(max(parse_json_vector(x)) for x in s)),
        pilot_anchor_rows=("weight_stratum", lambda s: int((s == "pilot_anchor").sum())),
    ).reset_index()
    known_counts = plan[
        plan["weight_stratum"].astype(str).eq("pilot_anchor")
        & plan["pilot_anchor_actual_feasible"].astype(str).str.lower().eq("true")
    ].groupby("context_id").size()
    context_summary["known_feasible_pilot_anchors"] = context_summary["context_id"].map(known_counts).fillna(0).astype(int)
    unique_pr = plan.groupby("context_id").apply(
        lambda g: g[["Pbar_kw_per_server", "R_kw_per_server"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    context_summary["unique_pr"] = context_summary["context_id"].map(unique_pr)

    hard_errors: list[str] = []
    for context in contexts:
        g = plan[plan["context_id"] == context.context_id]
        if len(g) == 0:
            hard_errors.append(f"Missing context {context.context_id}")
            continue
        for _, row in g.iterrows():
            if not legal_pr(context, float(row["Pbar_kw_per_server"]), float(row["R_kw_per_server"])):
                hard_errors.append(f"Illegal P/R in {row['plan_row_id']}")
            try:
                validate_weights(context, parse_json_vector(row["weights"]))
            except Exception as exc:
                hard_errors.append(f"Bad weights in {row['plan_row_id']}: {exc}")

    audit = {
        "rows": int(len(plan)),
        "base_rows": int((~plan["is_replicate"].astype(bool)).sum()),
        "replicate_rows": int(plan["is_replicate"].astype(bool).sum()),
        "contexts": int(plan["context_id"].nunique()),
        "workloads": sorted(plan["workload_name"].unique().tolist()),
        "server_counts": sorted(int(x) for x in plan["server_count"].unique()),
        "utilizations": sorted(float(x) for x in plan["utilization"].unique()),
        "job_type_counts": sorted(int(x) for x in plan["job_type_count"].unique()),
        "pr_strata": {str(k): int(v) for k, v in plan["pr_stratum"].value_counts().to_dict().items()},
        "weight_strata": {str(k): int(v) for k, v in plan["weight_stratum"].value_counts().to_dict().items()},
        "simulation_seeds": {str(k): int(v) for k, v in plan["simulation_seed"].value_counts().to_dict().items()},
        "known_feasible_pilot_anchor_rows": int((
            plan["weight_stratum"].astype(str).eq("pilot_anchor")
            & plan["pilot_anchor_actual_feasible"].astype(str).str.lower().eq("true")
        ).sum()),
        "hard_errors": hard_errors,
    }
    if hard_errors:
        raise RuntimeError("Plan audit failed:\n" + "\n".join(hard_errors[:20]))
    return context_summary, audit


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=str(SCRIPT_DIR), help="FlexDC repository root.")
    p.add_argument("--workload-configs", default=None, help="Comma-separated workload config paths.")
    p.add_argument("--experiment-configs", default=None, help="Comma-separated experiment config paths.")
    p.add_argument("--utilization-values", default="0.6,0.8")
    p.add_argument("--pilot-results-csv", default=None, help="Optional existing results CSV for pilot-guided regions and known-feasible anchors.")
    p.add_argument("--output-plan", default="flexdc_sweep_plan_v2.csv")
    p.add_argument("--output-audit", default="flexdc_sweep_plan_v2_audit.json")
    p.add_argument("--output-context-summary", default="flexdc_sweep_plan_v2_context_summary.csv")
    p.add_argument("--design-version", default="flexdc_sweep_v2")
    p.add_argument("--core-points", type=int, default=120)
    p.add_argument("--boundary-points", type=int, default=30)
    p.add_argument("--pilot-guided-points", type=int, default=30)
    p.add_argument("--dirichlet-alphas", default="0.3,1,3,10")
    p.add_argument("--base-seed", type=int, default=20)
    p.add_argument("--replicate-fraction", type=float, default=0.05)
    p.add_argument("--replicate-seeds", default="21,22")
    p.add_argument("--pbar-lower-factor", type=float, default=0.9)
    p.add_argument("--pbar-upper-factor", type=float, default=1.0)
    p.add_argument("--pr-upper-factor", type=float, default=1.2)
    p.add_argument("--r-lower-kw-per-server", type=float, default=0.01)
    return p


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    workload_configs = parse_csv_items(args.workload_configs, DEFAULT_WORKLOAD_CONFIGS)
    experiment_configs = parse_csv_items(args.experiment_configs, DEFAULT_EXPERIMENT_CONFIGS)
    utilizations = parse_float_items(args.utilization_values, DEFAULT_UTILIZATIONS)
    alphas = parse_float_items(args.dirichlet_alphas, DEFAULT_DIRICHLET_ALPHAS)
    replicate_seeds = [int(x) for x in parse_csv_items(args.replicate_seeds, ["21", "22"])]
    if len(alphas) != 4:
        raise ValueError("The production design expects exactly four Dirichlet alpha values, making six weights per P/R point total.")
    if not 0 <= args.replicate_fraction <= 1:
        raise ValueError("--replicate-fraction must be in [0,1].")

    contexts = build_contexts(
        repo_root=repo_root,
        workload_configs=workload_configs,
        experiment_configs=experiment_configs,
        utilizations=utilizations,
        pbar_lower_factor=args.pbar_lower_factor,
        pbar_upper_factor=args.pbar_upper_factor,
        pr_upper_factor=args.pr_upper_factor,
        r_lower=args.r_lower_kw_per_server,
    )
    pilot_path = resolve_path(args.pilot_results_csv, repo_root) if args.pilot_results_csv else None
    pilot = load_pilot_data(pilot_path)
    plan = make_plan_rows(
        contexts=contexts,
        pilot=pilot,
        design_version=args.design_version,
        core_count=args.core_points,
        boundary_count=args.boundary_points,
        pilot_count=args.pilot_guided_points,
        dirichlet_alphas=alphas,
        base_seed=args.base_seed,
        replicate_fraction=args.replicate_fraction,
        replicate_seeds=replicate_seeds,
    )
    summary, audit = audit_plan(plan, contexts)
    audit["repo_root"] = str(repo_root)
    audit["pilot_results_csv"] = str(pilot_path) if pilot_path else None
    audit["design_version"] = args.design_version
    audit["points_per_context"] = {
        "sobol_core": args.core_points,
        "physical_boundary": args.boundary_points,
        "pilot_guided": args.pilot_guided_points,
        "weights_per_pr": 2 + len(alphas),
    }

    output_plan = resolve_path(args.output_plan, repo_root)
    output_audit = resolve_path(args.output_audit, repo_root)
    output_summary = resolve_path(args.output_context_summary, repo_root)
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output_plan, index=False)
    summary.to_csv(output_summary, index=False)
    output_audit.write_text(json.dumps(audit, indent=2))

    print("FlexDC sweep plan generated")
    print("---------------------------")
    print(f"Plan: {output_plan}")
    print(f"Audit: {output_audit}")
    print(f"Context summary: {output_summary}")
    print(f"Rows: {len(plan):,}")
    print(f"Base rows: {(~plan['is_replicate'].astype(bool)).sum():,}")
    print(f"Replicate rows: {plan['is_replicate'].astype(bool).sum():,}")
    print(f"Contexts: {plan['context_id'].nunique()}")
    print(f"Known-feasible pilot-anchor rows included: {audit['known_feasible_pilot_anchor_rows']}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
