#!/usr/bin/env python3
"""
Generate server-count-specific FlexDC experiment configs from one canonical base INI.

python -u ./am_generate_paper_iso_experiment_configs.py \
  --base-experiment-config "configs/experiment/new_iso/traditional_signal/exp_inf_low_util=0.6.ini" \
  --output-dir "configs/experiment/new_iso/traditional_signal/generated_server_counts" \
  --server-counts "250,1000" \
  --name-template "exp_traditional_iso16_servers_{server_count}.ini"

Default base:
    configs/experiment/new_iso/traditional_signal/exp_inf_low_util=0.6.ini

The generator preserves comments, ordering, and all base settings. By default it
changes only [system] server_count. Optional --set section.key=value overrides
allow deliberate, auditable changes when needed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_BASE_CONFIG = Path(
    "configs/experiment/new_iso/traditional_signal/exp_inf_low_util=0.6.ini"
)
DEFAULT_OUTPUT_DIR = Path(
    "configs/experiment/new_iso/traditional_signal/generated_server_counts"
)
DEFAULT_SERVER_COUNTS = [250, 1000]
DEFAULT_NAME_TEMPLATE = "exp_traditional_iso16_servers_{server_count}.ini"


def parse_server_counts(value: str) -> list[int]:
    counts: list[int] = []

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue

        try:
            count = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid server count {item!r}; expected comma-separated positive integers."
            ) from exc

        if count <= 0:
            raise argparse.ArgumentTypeError(
                f"Server counts must be positive. Got {count}."
            )

        if count not in counts:
            counts.append(count)

    if not counts:
        raise argparse.ArgumentTypeError(
            "Provide at least one server count, for example: 250,1000"
        )

    return counts


def parse_override(value: str) -> tuple[str, str, str]:
    """
    Parse: section.key=value
    Example: iso.iso_signal_start_hour=16
    """
    if "=" not in value or "." not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError(
            "Overrides must use section.key=value, for example "
            "iso.iso_signal_start_hour=16"
        )

    left, replacement = value.split("=", 1)
    section, key = left.split(".", 1)

    section = section.strip()
    key = key.strip()
    replacement = replacement.strip()

    if not section or not key:
        raise argparse.ArgumentTypeError(
            "Overrides must use a non-empty section and key."
        )

    return section, key, replacement


def replace_ini_values(
    text: str,
    replacements: dict[tuple[str, str], str],
    source_path: Path,
) -> str:
    """
    Replace exact active key assignments while preserving comments and layout.

    Each requested section/key must occur exactly once. This avoids silently
    editing the wrong setting when an INI later gains duplicate keys.
    """
    section_pattern = re.compile(r"^\s*\[([^\]]+)\]\s*$")
    key_pattern = re.compile(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)(.*?)(\r?\n)?$")

    current_section: str | None = None
    seen = {target: 0 for target in replacements}
    output_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        section_match = section_pattern.match(line.strip())
        if section_match:
            current_section = section_match.group(1).strip()
            output_lines.append(line)
            continue

        stripped = line.lstrip()
        if stripped.startswith(";") or stripped.startswith("#"):
            output_lines.append(line)
            continue

        key_match = key_pattern.match(line)
        if key_match and current_section is not None:
            indent, key, separator, old_value, newline = key_match.groups()
            target = (current_section, key)

            if target in replacements:
                seen[target] += 1
                output_lines.append(
                    f"{indent}{key}{separator}{replacements[target]}"
                    f"{newline if newline is not None else ''}"
                )
                continue

        output_lines.append(line)

    missing = [f"[{section}] {key}" for (section, key), count in seen.items() if count == 0]
    duplicate = [f"[{section}] {key}" for (section, key), count in seen.items() if count > 1]

    if missing:
        raise ValueError(
            f"Could not find requested setting(s) in {source_path}: {', '.join(missing)}"
        )

    if duplicate:
        raise ValueError(
            f"Found duplicate active setting(s) in {source_path}: {', '.join(duplicate)}"
        )

    return "".join(output_lines)


def output_filename(template: str, server_count: int) -> str:
    try:
        filename = template.format(server_count=server_count)
    except KeyError as exc:
        raise ValueError(
            "Name template may use only {server_count}, for example "
            "'exp_traditional_iso16_servers_{server_count}.ini'."
        ) from exc

    candidate = Path(filename)

    if candidate.name != filename or filename in {".", ".."}:
        raise ValueError("--name-template must produce a filename, not a path.")

    if candidate.suffix.lower() != ".ini":
        raise ValueError("--name-template must produce a .ini filename.")

    return filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate server-count-specific experiment INIs from the canonical "
            "Fatih traditional-signal experiment config."
        )
    )

    parser.add_argument(
        "--base-experiment-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help=f"Canonical base INI. Default: {DEFAULT_BASE_CONFIG}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated INIs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--server-counts",
        type=parse_server_counts,
        default=DEFAULT_SERVER_COUNTS,
        help="Comma-separated server counts. Default: 250,1000",
    )
    parser.add_argument(
        "--name-template",
        type=str,
        default=DEFAULT_NAME_TEMPLATE,
        help=(
            "Output filename template using {server_count}. "
            f"Default: {DEFAULT_NAME_TEMPLATE}"
        ),
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        type=parse_override,
        default=[],
        metavar="SECTION.KEY=VALUE",
        help=(
            "Optional repeatable section-aware override. Example: "
            "--set iso.iso_signal_start_hour=16"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing generated files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs and effective settings without writing files.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    base_path = args.base_experiment_config
    output_dir = args.output_dir

    if not base_path.is_file():
        raise FileNotFoundError(f"Base experiment config not found: {base_path}")

    base_text = base_path.read_text(encoding="utf-8")

    shared_replacements: dict[tuple[str, str], str] = {}
    for section, key, value in args.overrides:
        target = (section, key)

        if target == ("system", "server_count"):
            raise ValueError(
                "Do not override system.server_count through --set. "
                "Use --server-counts instead."
            )

        if target in shared_replacements:
            raise ValueError(
                f"Duplicate override provided for [{section}] {key}."
            )

        shared_replacements[target] = value

    planned: list[tuple[int, Path, str]] = []

    for server_count in args.server_counts:
        filename = output_filename(args.name_template, server_count)
        destination = output_dir / filename

        replacements = dict(shared_replacements)
        replacements[("system", "server_count")] = str(server_count)

        generated_text = replace_ini_values(
            text=base_text,
            replacements=replacements,
            source_path=base_path,
        )

        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing file: {destination}. "
                "Use --overwrite if this is intentional."
            )

        planned.append((server_count, destination, generated_text))

    print(f"Base config: {base_path}")
    print(f"Output directory: {output_dir}")
    print(f"Server counts: {args.server_counts}")
    print(f"Shared overrides: {args.overrides if args.overrides else '[none]'}")
    print()

    for server_count, destination, _ in planned:
        print(f"{'[dry-run] ' if args.dry_run else ''}server_count={server_count} -> {destination}")

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for _, destination, generated_text in planned:
        destination.write_text(generated_text, encoding="utf-8")

    print(f"\nWrote {len(planned)} config file(s).")


if __name__ == "__main__":
    main()
