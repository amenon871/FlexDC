"""Resume ledger populated by independently reconciled simulator artifacts."""

import json
import sqlite3
from contextlib import contextmanager

from .audit import audit_observation, require
from .io import csv_rows, digest, encode


@contextmanager
def ledger(root):
    connection = sqlite3.connect(root / "observations.sqlite", timeout=20)
    try:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS observations (
              row_id TEXT PRIMARY KEY, context_id TEXT NOT NULL, stage TEXT NOT NULL,
              body TEXT NOT NULL, body_sha256 TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS context_stage ON observations(context_id, stage, row_id);
            CREATE TABLE IF NOT EXISTS arrivals(context_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
        """)
        yield connection
    finally:
        connection.close()


def measured(connection, context_id, stage):
    observations = {}
    for rid, body, checksum in connection.execute(
        "SELECT row_id,body,body_sha256 FROM observations WHERE context_id=? AND stage=? ORDER BY row_id",
        (context_id, stage),
    ):
        decoded = json.loads(body)
        require(
            digest(decoded) == checksum
            and decoded["plan_row_id"] == rid
            and decoded["context_id"] == context_id
            and decoded["stage"] == stage,
            f"Ledger observation integrity mismatch {rid}",
        )
        observations[rid] = decoded
    return observations


def accept(connection, observation):
    cid, rid = observation["context_id"], observation["plan_row_id"]
    arrival = observation["evidence"]["arrival_sha256"]
    stored = connection.execute(
        "SELECT sha256 FROM arrivals WHERE context_id=?", (cid,)
    ).fetchone()
    require(
        stored is None or stored[0] == arrival,
        f"Common-arrival contract violated for {cid}",
    )
    old = connection.execute(
        "SELECT body FROM observations WHERE row_id=?", (rid,)
    ).fetchone()
    body = encode(observation)
    require(
        old is None or old[0] == body,
        f"Multiple distinct successful observations for {rid}",
    )
    connection.execute("INSERT OR IGNORE INTO arrivals VALUES (?,?)", (cid, arrival))
    connection.execute(
        "INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?)",
        (rid, cid, observation["stage"], body, digest(observation)),
    )


def reconcile(connection, output, planned, manifest, provenance):
    def records(name):
        path = output / name
        if not path.exists():
            return {}
        found = {}
        for row in csv_rows(path):
            rid = row.get("Plan_Row_ID")
            if rid in (None, ""):
                continue
            require(
                rid in planned and rid not in found,
                f"Foreign or duplicate result ID in {path}: {rid}",
            )
            found[rid] = row
        return found

    results = records("grid_search_results.csv")
    diagnostics = records("grid_search_diagnostics.csv")
    evidence_path = output / "p100_doe_evidence.jsonl"
    evidence = []
    if evidence_path.exists():
        with evidence_path.open(encoding="utf8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                evidence.append(json.loads(line))
    errors = []
    accepted = 0
    result_iterations = {row["Iteration"] for row in results.values()}
    expected_order = sorted(planned)
    require(
        len(result_iterations) == len(results), f"Repeated worker iteration: {output}"
    )
    with connection:
        for rid in results.keys() & diagnostics.keys():
            try:
                result = results[rid]
                iteration = int(result["Iteration"])
                require(
                    0 < iteration <= len(evidence),
                    "Missing terminal simulator evidence",
                )
                require(
                    rid == expected_order[iteration - 1],
                    "Result does not match worker chunk iteration",
                )
                observation = audit_observation(
                    planned[rid],
                    result,
                    diagnostics[rid],
                    evidence[iteration - 1],
                    manifest,
                )
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                errors.append(dict(row_id=rid, error=str(exc)))
                continue
            observation["provenance"] = {**provenance, "output": output.name}
            accept(connection, observation)
            accepted += 1
    return dict(
        committed=accepted,
        result_rows=len(results),
        diagnostic_rows=len(diagnostics),
        terminal_simulator_rows=len(evidence),
        invalid=errors,
    )
