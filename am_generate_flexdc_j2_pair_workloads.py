#!/usr/bin/env python3
"""Generate all 28 named J=2 FlexDC workload INI files for the pairwise-profile experiment.

Run this file from the FlexDC repository root. The default design uses the
physical training profiles from W1-train-qos4444.ini and the physical inference
profiles from W2-short-qos5555.ini. The W1 qos3333 and W2 mixed-QoS files are
used only as lineage/audit sources: power, runtime, and job-size fields must
match their canonical counterpart; their alternate QoS values are not copied.

The generated master workload set contains:
  * 6 inference-only pairs;
  * 6 training-only pairs;
  * 16 inference/training pairs;
  * 28 total J=2 workload definitions.

Outputs under --output-dir:
  * 28 .ini workload files;
  * j2_pairwise_workloads_manifest.csv;
  * j2_pairwise_workloads_manifest.json;
  * j2_pairwise_planner_workload_list.txt.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_TRAIN_SOURCE = "configs/workload/W1-train-qos4444.ini"
DEFAULT_TRAIN_AUDIT_SOURCE = "configs/workload/W1-train-qos3333.ini"
DEFAULT_INFER_SOURCE = "configs/workload/W2-short-qos5555.ini"
DEFAULT_INFER_AUDIT_SOURCE = "configs/workload/W2-short-qos5_4.5_4_3.5.ini"
DEFAULT_OUTPUT_DIR = "configs/workload/j2_pairwise"

REQUIRED_FIELD_ALIASES = {
    "min_job_power_watts": {"minjobpowerwatts"},
    "max_job_power_watts": {"maxjobpowerwatts"},
    "min_time_seconds": {
        "mintimeseconds",
        "minexecutiontimeseconds",
        "minexecutiontime",
    },
    "max_time_seconds": {
        "maxtimeseconds",
        "maxexecutiontimeseconds",
        "maxexecutiontime",
    },
    "qos_constraint": {"qosconstraint"},
    "job_size": {"jobsize"},
}

PHYSICAL_FIELD_NAMES = (
    "min_job_power_watts",
    "max_job_power_watts",
    "min_time_seconds",
    "max_time_seconds",
    "job_size",
)

APPLICATION_ALIASES = {
    "resnet": ("resnet", "resnet50"),
    "gpt2": ("gpt2", "gpt-2", "gpt_2"),
    "llama": ("llama", "llama7b", "llama-7b"),
    "bloom": ("bloom", "bloom560m", "bloom-560m"),
}

APPLICATION_DISPLAY = {
    "resnet": "ResNet",
    "gpt2": "GPT-2",
    "llama": "Llama",
    "bloom": "Bloom",
}

APPLICATION_FILE = {
    "resnet": "ResNet",
    "gpt2": "GPT2",
    "llama": "Llama",
    "bloom": "Bloom",
}

ROLE_DISPLAY = {
    "infer": "Inference",
    "train": "Training",
}

ROLE_FILE = {
    "infer": "Inf",
    "train": "Train",
}

APPLICATION_ORDER = ("resnet", "gpt2", "llama", "bloom")


@dataclass(frozen=True)
class ProfileSpec:
    role: str
    application: str

    @property
    def display_name(self) -> str:
        return f"{APPLICATION_DISPLAY[self.application]} {ROLE_DISPLAY[self.role]}"

    @property
    def short_name(self) -> str:
        return f"{APPLICATION_FILE[self.application]}{ROLE_FILE[self.role]}"


@dataclass(frozen=True)
class WorkloadSpec:
    workload_id: str
    workload_type: str
    first: ProfileSpec
    second: ProfileSpec

    @property
    def display_name(self) -> str:
        return f"{self.first.display_name} + {self.second.display_name}"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def read_ini(path: Path) -> configparser.ConfigParser:
    if not path.is_file():
        raise FileNotFoundError(path)
    parser = configparser.ConfigParser(
        interpolation=None,
        strict=True,
        delimiters=("=",),
        comment_prefixes=("#", ";"),
        inline_comment_prefixes=("#", ";"),
        empty_lines_in_values=False,
    )
    parser.optionxform = str
    with path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    if not parser.sections():
        raise ValueError(f"No workload sections found in {path}")
    return parser


def section_values_with_defaults(
    parser: configparser.ConfigParser,
    section: str,
) -> dict[str, str]:
    return {str(key): str(value) for key, value in parser.items(section, raw=True)}


def semantic_key_map(values: dict[str, str]) -> dict[str, str]:
    output: dict[str, str] = {}
    normalized = {normalize(key): key for key in values}
    for semantic_name, aliases in REQUIRED_FIELD_ALIASES.items():
        matches = [normalized[alias] for alias in aliases if alias in normalized]
        if not matches:
            raise ValueError(
                f"Missing semantic field {semantic_name}; available keys={sorted(values)}"
            )
        output[semantic_name] = matches[0]
    return output


def match_application_sections(
    parser: configparser.ConfigParser,
    *,
    source_path: Path,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for application, aliases in APPLICATION_ALIASES.items():
        normalized_aliases = tuple(normalize(alias) for alias in aliases)
        matches = []
        for section in parser.sections():
            section_norm = normalize(section)
            if any(alias in section_norm for alias in normalized_aliases):
                matches.append(section)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {application} section in {source_path}; "
                f"found {len(matches)}: {matches}. All sections={parser.sections()}"
            )
        semantic_key_map(section_values_with_defaults(parser, matches[0]))
        result[application] = matches[0]
    return result


def comparable_number(value: str) -> float | str:
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value).strip()


def audit_physical_profile_consistency(
    *,
    canonical_parser: configparser.ConfigParser,
    canonical_sections: dict[str, str],
    canonical_path: Path,
    audit_parser: configparser.ConfigParser,
    audit_sections: dict[str, str],
    audit_path: Path,
    role: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for application in APPLICATION_ORDER:
        canonical_values = section_values_with_defaults(
            canonical_parser, canonical_sections[application]
        )
        audit_values = section_values_with_defaults(
            audit_parser, audit_sections[application]
        )
        canonical_keys = semantic_key_map(canonical_values)
        audit_keys = semantic_key_map(audit_values)
        for field in PHYSICAL_FIELD_NAMES:
            left = comparable_number(canonical_values[canonical_keys[field]])
            right = comparable_number(audit_values[audit_keys[field]])
            if isinstance(left, float) and isinstance(right, float):
                same = abs(left - right) <= 1e-12
            else:
                same = left == right
            rows.append(
                {
                    "role": role,
                    "application": application,
                    "field": field,
                    "canonical_source": relative_or_absolute(canonical_path, canonical_path.parent.parent.parent),
                    "canonical_value": left,
                    "audit_source": relative_or_absolute(audit_path, audit_path.parent.parent.parent),
                    "audit_value": right,
                    "match": bool(same),
                }
            )
            if not same:
                raise ValueError(
                    f"Physical-profile mismatch for {role}/{application}/{field}: "
                    f"{canonical_path}={left!r}, {audit_path}={right!r}. "
                    "Only QoS is allowed to differ between the canonical and audit source files."
                )
    return rows


def build_workload_specs() -> tuple[WorkloadSpec, ...]:
    inference = [ProfileSpec("infer", app) for app in APPLICATION_ORDER]
    training = [ProfileSpec("train", app) for app in APPLICATION_ORDER]
    specs: list[WorkloadSpec] = []

    for first, second in itertools.combinations(inference, 2):
        specs.append(
            WorkloadSpec(
                workload_id=f"J2-II-{first.short_name}-{second.short_name}",
                workload_type="Inference-Only",
                first=first,
                second=second,
            )
        )
    for first, second in itertools.combinations(training, 2):
        specs.append(
            WorkloadSpec(
                workload_id=f"J2-TT-{first.short_name}-{second.short_name}",
                workload_type="Training-Only",
                first=first,
                second=second,
            )
        )
    for first in inference:
        for second in training:
            specs.append(
                WorkloadSpec(
                    workload_id=f"J2-IT-{first.short_name}-{second.short_name}",
                    workload_type="Mixed",
                    first=first,
                    second=second,
                )
            )

    if len(specs) != 28:
        raise AssertionError(f"Expected 28 J=2 workloads, built {len(specs)}")
    if len({spec.workload_id for spec in specs}) != 28:
        raise AssertionError("Duplicate workload IDs were generated.")
    return tuple(specs)


def unique_output_section_name(
    source_section: str,
    profile: ProfileSpec,
    used: set[str],
) -> str:
    candidate = source_section
    if candidate not in used:
        used.add(candidate)
        return candidate
    base = f"{source_section}__{profile.short_name}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def profile_source(
    profile: ProfileSpec,
    *,
    train_parser: configparser.ConfigParser,
    infer_parser: configparser.ConfigParser,
    train_sections: dict[str, str],
    infer_sections: dict[str, str],
    train_source: Path,
    infer_source: Path,
) -> tuple[configparser.ConfigParser, str, Path]:
    if profile.role == "train":
        return train_parser, train_sections[profile.application], train_source
    if profile.role == "infer":
        return infer_parser, infer_sections[profile.application], infer_source
    raise ValueError(f"Unsupported profile role: {profile.role}")


def write_workload(
    *,
    spec: WorkloadSpec,
    output_path: Path,
    train_parser: configparser.ConfigParser,
    infer_parser: configparser.ConfigParser,
    train_sections: dict[str, str],
    infer_sections: dict[str, str],
    train_source: Path,
    infer_source: Path,
    repo_root: Path,
    overwrite: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace generated files."
        )

    output = configparser.ConfigParser(
        interpolation=None,
        strict=True,
        delimiters=("=",),
        empty_lines_in_values=False,
    )
    output.optionxform = str
    used_sections: set[str] = set()
    job_rows: list[dict[str, object]] = []

    for job_index, profile in enumerate((spec.first, spec.second)):
        source_parser, source_section, source_path = profile_source(
            profile,
            train_parser=train_parser,
            infer_parser=infer_parser,
            train_sections=train_sections,
            infer_sections=infer_sections,
            train_source=train_source,
            infer_source=infer_source,
        )
        values = section_values_with_defaults(source_parser, source_section)
        keys = semantic_key_map(values)
        output_section = unique_output_section_name(source_section, profile, used_sections)
        output.add_section(output_section)
        for key, value in values.items():
            output.set(output_section, key, value)

        job_rows.append(
            {
                "workload_id": spec.workload_id,
                "workload_type": spec.workload_type,
                "workload_display_name": spec.display_name,
                "output_file": relative_or_absolute(output_path, repo_root),
                "job_count": 2,
                "job_index": job_index,
                "profile_name": profile.display_name,
                "role": profile.role,
                "application": profile.application,
                "output_section": output_section,
                "source_file": relative_or_absolute(source_path, repo_root),
                "source_section": source_section,
                "qos_constraint": float(values[keys["qos_constraint"]]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    from io import StringIO

    buffer = StringIO()
    output.write(buffer, space_around_delimiters=True)
    header = (
        "# Generated by am_generate_flexdc_j2_pair_workloads.py\n"
        f"# Workload ID: {spec.workload_id}\n"
        f"# Workload type: {spec.workload_type}\n"
        f"# Profiles: {spec.display_name}\n"
        "# Each source section is copied completely and unchanged.\n"
        "# Training profiles use W1-train-qos4444.ini; inference profiles use W2-short-qos5555.ini.\n\n"
    )
    output_path.write_text(header + buffer.getvalue(), encoding="utf-8")

    generated = read_ini(output_path)
    if len(generated.sections()) != 2:
        raise RuntimeError(f"Generated {output_path} has J={len(generated.sections())}; expected J=2")

    summary_row = {
        "workload_id": spec.workload_id,
        "workload_type": spec.workload_type,
        "workload_display_name": spec.display_name,
        "job_1_profile": spec.first.display_name,
        "job_2_profile": spec.second.display_name,
        "job_1_role": spec.first.role,
        "job_2_role": spec.second.role,
        "job_1_application": spec.first.application,
        "job_2_application": spec.second.application,
        "job_count": 2,
        "output_file": relative_or_absolute(output_path, repo_root),
    }
    return [summary_row], job_rows


def verify_with_flexdc_parser(
    repo_root: Path,
    generated_paths: Iterable[Path],
) -> list[str]:
    messages: list[str] = []
    src_dir = repo_root / "src"
    if not src_dir.is_dir():
        return ["FlexDC parser verification skipped: repo_root/src was not found."]
    sys.path.insert(0, str(src_dir))
    try:
        from peacsim.parsing.job_profile_reader import JobProfileReader
    except Exception as exc:
        return [
            "FlexDC parser verification skipped because JobProfileReader could not be imported: "
            f"{type(exc).__name__}: {exc}"
        ]
    for path in generated_paths:
        parsed = JobProfileReader(str(path))
        actual = int(parsed.job_type_count)
        if actual != 2:
            raise RuntimeError(f"FlexDC parser read J={actual} from {path}; expected J=2")
        messages.append(f"FlexDC parser verified {path.name}: J=2")
    return messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(SCRIPT_DIR))
    parser.add_argument("--train-source", default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--train-audit-source", default=DEFAULT_TRAIN_AUDIT_SOURCE)
    parser.add_argument("--infer-source", default=DEFAULT_INFER_SOURCE)
    parser.add_argument("--infer-audit-source", default=DEFAULT_INFER_AUDIT_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    train_source = resolve(repo_root, args.train_source)
    train_audit_source = resolve(repo_root, args.train_audit_source)
    infer_source = resolve(repo_root, args.infer_source)
    infer_audit_source = resolve(repo_root, args.infer_audit_source)
    output_dir = resolve(repo_root, args.output_dir)

    train_parser = read_ini(train_source)
    train_audit_parser = read_ini(train_audit_source)
    infer_parser = read_ini(infer_source)
    infer_audit_parser = read_ini(infer_audit_source)

    train_sections = match_application_sections(train_parser, source_path=train_source)
    train_audit_sections = match_application_sections(
        train_audit_parser, source_path=train_audit_source
    )
    infer_sections = match_application_sections(infer_parser, source_path=infer_source)
    infer_audit_sections = match_application_sections(
        infer_audit_parser, source_path=infer_audit_source
    )

    source_audit_rows = []
    source_audit_rows.extend(
        audit_physical_profile_consistency(
            canonical_parser=train_parser,
            canonical_sections=train_sections,
            canonical_path=train_source,
            audit_parser=train_audit_parser,
            audit_sections=train_audit_sections,
            audit_path=train_audit_source,
            role="train",
        )
    )
    source_audit_rows.extend(
        audit_physical_profile_consistency(
            canonical_parser=infer_parser,
            canonical_sections=infer_sections,
            canonical_path=infer_source,
            audit_parser=infer_audit_parser,
            audit_sections=infer_audit_sections,
            audit_path=infer_audit_source,
            role="infer",
        )
    )

    print("Canonical training sections:")
    for app in APPLICATION_ORDER:
        print(f"  {APPLICATION_DISPLAY[app]:7s} -> [{train_sections[app]}]")
    print("Canonical inference sections:")
    for app in APPLICATION_ORDER:
        print(f"  {APPLICATION_DISPLAY[app]:7s} -> [{infer_sections[app]}]")
    print("Physical-profile consistency audit: PASS")

    summary_rows: list[dict[str, object]] = []
    job_rows: list[dict[str, object]] = []
    generated_paths: list[Path] = []

    for spec in build_workload_specs():
        output_path = output_dir / f"{spec.workload_id}.ini"
        summaries, jobs = write_workload(
            spec=spec,
            output_path=output_path,
            train_parser=train_parser,
            infer_parser=infer_parser,
            train_sections=train_sections,
            infer_sections=infer_sections,
            train_source=train_source,
            infer_source=infer_source,
            repo_root=repo_root,
            overwrite=args.overwrite,
        )
        summary_rows.extend(summaries)
        job_rows.extend(jobs)
        generated_paths.append(output_path)
        print(f"Generated {relative_or_absolute(output_path, repo_root)}")

    manifest_csv = output_dir / "j2_pairwise_workloads_manifest.csv"
    jobs_csv = output_dir / "j2_pairwise_workload_jobs_manifest.csv"
    manifest_json = output_dir / "j2_pairwise_workloads_manifest.json"
    source_audit_csv = output_dir / "j2_pairwise_source_profile_audit.csv"
    planner_list = output_dir / "j2_pairwise_planner_workload_list.txt"

    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with jobs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(job_rows[0]))
        writer.writeheader()
        writer.writerows(job_rows)
    with source_audit_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_audit_rows[0]))
        writer.writeheader()
        writer.writerows(source_audit_rows)

    manifest_json.write_text(
        json.dumps(
            {
                "design_version": "j2_pairwise_workloads_v1",
                "job_count": 2,
                "canonical_training_source": relative_or_absolute(train_source, repo_root),
                "training_audit_source": relative_or_absolute(train_audit_source, repo_root),
                "canonical_inference_source": relative_or_absolute(infer_source, repo_root),
                "inference_audit_source": relative_or_absolute(infer_audit_source, repo_root),
                "workload_count": len(summary_rows),
                "workloads": summary_rows,
                "jobs": job_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    planner_list.write_text(
        ",".join(relative_or_absolute(path, repo_root) for path in generated_paths) + "\n",
        encoding="utf-8",
    )

    print("\nParser verification:")
    for message in verify_with_flexdc_parser(repo_root, generated_paths):
        print(f"  {message}")

    print("\nSummary")
    print("-------")
    print("Inference-only workloads: 6")
    print("Training-only workloads: 6")
    print("Mixed workloads: 16")
    print("Total workloads: 28")
    print(f"Manifest: {relative_or_absolute(manifest_csv, repo_root)}")
    print(f"Planner list: {relative_or_absolute(planner_list, repo_root)}")


if __name__ == "__main__":
    main()
