"""Space-filling and measured-only sampling over the legal exact-plan domain."""

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.stats import qmc

from am_generate_flexdc_exact_sweep_plan import (
    resolve_configured_weight_bounds,
    stable_seed,
)

from .design import A_COUNT, B_COUNT, DESIGN_SEED, POWER
from .io import ROOT, VERSION, digest, encode


@dataclass(frozen=True)
class Domain:
    pmin: float
    pmax: float
    j: int

    @cached_property
    def weights(self):
        return resolve_configured_weight_bounds(
            job_type_count=self.j,
            server_count=1000,
            lower_factor_of_equal=0.6,
            upper_factor_of_equal=1.8,
            explicit_lower=None,
            explicit_upper=None,
        )

    @property
    def p_lower(self):
        return 0.9 * self.pmin

    def reserve_upper(self, p):
        return min(0.6 * p, 1.2 * self.pmax - p)

    def validate(self, coordinates):
        p, r, *weights = coordinates
        _, _, low, high = self.weights
        tolerance = 2e-10
        if (
            len(weights) != self.j
            or not all(math.isfinite(x) for x in coordinates)
            or not self.p_lower - tolerance <= p <= self.pmax + tolerance
            or not 0.01 - tolerance <= r <= self.reserve_upper(p) + tolerance
            or any(not low - tolerance <= w <= high + tolerance for w in weights)
            or abs(sum(weights) - 1) > tolerance
        ):
            raise ValueError(f"Illegal bid {coordinates}")
        return coordinates

    def encode(self, coordinates):
        p, r, *weights = coordinates
        _, _, low, high = self.weights
        return np.array(
            [
                (p - self.p_lower) / (self.pmax - self.p_lower),
                (r - 0.01) / (self.reserve_upper(p) - 0.01),
                *((w - low) / (high - low) for w in weights),
            ]
        )

    def decode(self, z):
        return self.decode_many(np.atleast_2d(z))[0]

    def decode_many(self, z):
        _, _, low, high = self.weights
        z = np.clip(z, 0, 1)
        p = self.p_lower + (self.pmax - self.p_lower) * z[:, 0]
        r = 0.01 + (np.minimum(0.6 * p, 1.2 * self.pmax - p) - 0.01) * z[:, 1]
        # Deterministic bounded-simplex projection: w = clip(x - lambda, lo, hi).
        x = low + (high - low) * z[:, 2:]
        a, b = x.min(axis=1, keepdims=True) - high, x.max(axis=1, keepdims=True) - low
        for _ in range(39):
            midpoint = (a + b) / 2
            above = np.clip(x - midpoint, low, high).sum(axis=1, keepdims=True) > 1
            a, b = np.where(above, midpoint, a), np.where(above, b, midpoint)
        weights = np.clip(x - (a + b) / 2, low, high)
        # Correct floating-point residue without leaving the bounds.
        slots = np.argmax(np.minimum(weights - low, high - weights), axis=1)
        weights[np.arange(len(weights)), slots] += 1 - weights.sum(axis=1)
        # The wizard quantizes aggregate P and R to whole watts. Stay inside the
        # domain after quantization and use the applied grid for deduplication.
        grid = 1_000_000
        p = (
            np.clip(
                np.rint(p * grid),
                math.ceil(self.p_lower * grid - 1e-8),
                math.floor(self.pmax * grid + 1e-8),
            )
            / grid
        )
        upper = np.floor(np.minimum(0.6 * p, 1.2 * self.pmax - p) * grid + 1e-8)
        r = np.minimum(upper, np.maximum(10_000, np.rint(r * grid))) / grid
        return [
            self.validate(tuple(row))
            for row in np.column_stack((p, r, weights)).tolist()
        ]


def domain_for(manifest, context):
    workload = manifest["workloads"][context["workload"]]
    profiles = [manifest["profiles"][p]["physical"] for p in workload["jobs"]]
    return Domain(
        min(float(p[POWER[0]]) for p in profiles) / 1000,
        max(float(p[POWER[1]]) for p in profiles) / 1000,
        context["j"],
    )


def point_key(point):
    return tuple(round(x, 10) for x in point)


def coords(row):
    import json

    weights = json.loads(row["weights"])
    return (float(row["Pbar_kw_per_server"]), float(row["R_kw_per_server"]), *weights)


def sobol(domain, context_id, salt, bits):
    return qmc.Sobol(
        domain.j + 2, scramble=True, seed=stable_seed(DESIGN_SEED, context_id, salt)
    ).random_base2(bits)


def broad(domain, context_id):
    points = sobol(domain, context_id, "A", 8)
    for index in range(16):
        points[index, :2] = [(index // 2) % 2, index % 2]
    seen = set()
    answer = []
    for index, point in enumerate(domain.decode_many(points)):
        key = point_key(point)
        if key not in seen:
            seen.add(key)
            answer.append((point, "physical_edge" if index < 16 else "sobol", ""))
        if len(answer) == A_COUNT:
            return answer
    raise ValueError(f"Stage A could not produce {A_COUNT} distinct bids: {context_id}")


def margins(p90, probabilities):
    tracking = p90 / 0.30 - 1
    qos = max(probabilities) / 0.10 - 1
    return max(tracking, qos), math.hypot(max(0, tracking), max(0, qos))


def adaptive(domain, context_id, a_rows, measured):
    if (
        len(a_rows) != A_COUNT
        or len({r["plan_row_id"] for r in a_rows}) != A_COUNT
        or {r["plan_row_id"] for r in a_rows} != set(measured)
    ):
        raise ValueError("Stage B requires exactly 150 audited Stage A observations")
    ranked = []
    for row in a_rows:
        result = measured[row["plan_row_id"]]
        p90, probabilities = result["p90"], result["qos_probabilities"]
        if (
            len(probabilities) != domain.j
            or not math.isfinite(p90)
            or p90 < 0
            or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
        ):
            raise ValueError("Invalid measured constraint evidence")
        ranked.append(
            (
                margins(p90, probabilities)[0],
                row["plan_row_id"],
                domain.validate(coords(row)),
            )
        )
    ranked.sort(key=lambda entry: (abs(entry[0]), entry[1]))
    feasible = [row for row in ranked if row[0] <= 0]
    infeasible = [row for row in ranked if row[0] > 0]
    near = [row for row in ranked if abs(row[0]) <= 0.25]
    anchors = near[:12] if near else ranked[:8]
    # Reserve a straddling interpolation pool when both sides were observed.
    normalized = [domain.encode(row[2]) for row in ranked]
    occupied = {point_key(row[2]) for row in ranked}
    candidates = []
    rejected = 0

    def add(z, family, anchor):
        nonlocal rejected
        point = domain.decode(z)
        key = point_key(point)
        if key in occupied:
            rejected += 1
        else:
            occupied.add(key)
            candidates.append((point, family, anchor))

    if feasible and infeasible:
        for left in feasible[:4]:
            for right in infeasible[:4]:
                t = abs(left[0]) / (abs(left[0]) + abs(right[0]))
                add(
                    (1 - t) * domain.encode(left[2]) + t * domain.encode(right[2]),
                    "measured_straddle",
                    left[1] + "|" + right[1],
                )
    for i, z in enumerate(sobol(domain, context_id, "B local", 8)):
        anchor = anchors[i % len(anchors)]
        add(
            domain.encode(anchor[2]) + (0.045 if near else 0.09) * (2 * z - 1),
            "boundary_local" if near else "closest_anchor_fallback",
            anchor[1],
        )

    def greedy(pool, quota):
        chosen, distances = [], []
        if not pool:
            return chosen, distances
        encoded = np.array([domain.encode(point) for point, _, _ in pool])
        refs = np.array(normalized)
        distance = np.min(
            np.linalg.norm(encoded[:, None, :] - refs[None, :, :], axis=2), axis=1
        )
        for _ in range(min(quota, len(pool))):
            i = int(np.argmax(distance))
            chosen.append(pool[i])
            distances.append(float(distance[i]))
            normalized.append(encoded[i])
            distance = np.minimum(
                distance, np.linalg.norm(encoded - encoded[i], axis=1)
            )
            distance[i] = -1
        return chosen, distances

    chosen, distances = greedy(
        candidates, 40 if near or (feasible and infeasible) else 20
    )
    candidates = []
    for z in sobol(domain, context_id, "B broad", 8):
        add(z, "broad_fallback", "")
    extra, extra_distances = greedy(candidates, B_COUNT - len(chosen))
    chosen += extra
    distances += extra_distances
    if len(chosen) != B_COUNT:
        raise ValueError(f"Stage B could not produce {B_COUNT} distinct bids")
    directed = sum(
        family in ("boundary_local", "measured_straddle") for _, family, _ in chosen
    )
    return chosen, dict(
        feasible=len(feasible),
        infeasible=len(infeasible),
        closest_signed_margin=ranked[0][0],
        near_boundary_initial=len(near),
        boundary_directed=directed,
        fallback=len(chosen) - directed,
        rejected=rejected,
        distance_min=min(distances),
        distance_median=float(np.median(distances)),
    )


def plan_rows(root, manifest, context_id, stage, samples):
    context = manifest["contexts"][context_id]
    workload = manifest["workloads"][context["workload"]]
    domain = domain_for(manifest, context)
    low, high, _, _ = domain.weights
    for i, (point, family, anchor) in enumerate(samples):
        row_id = f"P100D_{stage}_{digest([context_id, i, point_key(point)])[:24]}"
        yield dict(
            plan_row_id=row_id,
            base_plan_row_id=row_id,
            design_version=VERSION,
            context_id=context_id,
            workload_config=(root / workload["path"]).relative_to(ROOT).as_posix(),
            experiment_config=(root / context["experiment"])
            .relative_to(ROOT)
            .as_posix(),
            server_count=context["n"],
            utilization=context["u"],
            job_type_count=context["j"],
            Pbar_kw_per_server=point[0],
            R_kw_per_server=point[1],
            weights=encode(point[2:]),
            simulation_seed=context["seed"],
            configured_r_over_p_max=0.6,
            configured_weight_lower=low,
            configured_weight_upper=high,
            pr_stratum=stage,
            weight_family=family,
            anchor_type=anchor,
            normalized_coordinates=encode(domain.encode(point).tolist()),
            weight_index_within_pr=i,
            is_replicate=False,
        )


FIELDS = (
    "plan_row_id",
    "base_plan_row_id",
    "design_version",
    "context_id",
    "workload_config",
    "experiment_config",
    "server_count",
    "utilization",
    "job_type_count",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "weights",
    "simulation_seed",
    "configured_r_over_p_max",
    "configured_weight_lower",
    "configured_weight_upper",
    "pr_stratum",
    "weight_family",
    "anchor_type",
    "normalized_coordinates",
    "weight_index_within_pr",
    "is_replicate",
)
