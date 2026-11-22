"""Stable serialization for isolated DOE artifacts."""

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "P100_V3_DOE_V1"
SEED = 20
PROTECTED = {
    "am_run_flexdc_dataset_parallel_combo.py": "094094fe3aa5af6c6a2f2dc287a9a89e5ec9b326cfddb9f28569702293f3194b",
    "src/peacsim/am_data_extraction_wizard.py": "1628c6ab100aa705f790585365226a6c13ba78365e3d91f078a736ebddc34db7",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else digest_stream(stream)
        )


def digest_stream(stream):
    hasher = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        hasher.update(block)
    return hasher.hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def atomic(path, payload, immutable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Frozen artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if immutable:
            try:
                if os.name == "nt":
                    os.rename(temporary, path)
                else:
                    os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_write(path, value, immutable=True):
    atomic(path, (encode(value) + "\n").encode(), immutable)


def load(path):
    with Path(path).open(encoding="utf8") as stream:
        return json.load(stream)


def csv_write(path, rows, fields, immutable=True):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, buffer.getvalue().encode(), immutable)


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf8") as stream:
        yield from csv.DictReader(stream)


def protected():
    actual = {name: sha(ROOT / name) for name in PROTECTED}
    if actual != PROTECTED:
        raise ValueError("Protected runner or wizard differs from reviewed baseline")
    return actual
