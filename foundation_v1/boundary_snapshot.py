"""Create a verified read-only-source SQLite snapshot for local boundary planning."""
import argparse
import json
import os
import sqlite3
from pathlib import Path

from am_flexdc_foundation_v1_common import file_hash,write_json
from foundation_v1.audit import readonly_connection,verify_planning_database


def snapshot(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if not source.exists():raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_name(destination.name+'.tmp')
    temporary.unlink(missing_ok=True)
    src=readonly_connection(source);dst=sqlite3.connect(temporary)
    try:src.backup(dst);dst.commit()
    finally:dst.close();src.close()
    before=verify_planning_database(source);after=verify_planning_database(temporary)
    if before['database_content_sha256']!=after['database_content_sha256'] or before['rows']!=after['rows']:
        temporary.unlink(missing_ok=True);raise ValueError('SQLite snapshot scientific inventory mismatch')
    if destination.exists():
        existing=verify_planning_database(destination)
        if existing['database_content_sha256']!=before['database_content_sha256']:raise ValueError('Existing snapshot differs; refusing overwrite')
        temporary.unlink()
    else:os.replace(temporary,destination)
    receipt={'status':'VERIFIED_READ_ONLY_SOURCE_SNAPSHOT','source_path':str(source),'destination_path':str(destination),
        'source_bytes':source.stat().st_size,'destination_bytes':destination.stat().st_size,
        'source_inventory':before,'destination_inventory':verify_planning_database(destination),
        'destination_file_sha256':file_hash(destination),'source_was_opened_read_only':True}
    write_json(destination.with_suffix(destination.suffix+'.snapshot_receipt.json'),receipt,False)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True,type=Path);parser.add_argument('--destination',required=True,type=Path)
    print(json.dumps(snapshot(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':main()
