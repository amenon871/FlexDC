"""Disposable per-stage resume cache. Atomic receipts remain completion authority."""
import argparse
import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path

from am_flexdc_foundation_v1_common import atomic_bytes, file_hash, iter_csv, read_json
from foundation_v1 import collector_receipts as receipts

SCHEMA = 'FOUNDATION_RECOVERY_INDEX_V1'
BATCH_SIZE = 100
FLUSH_SECONDS = 5


def invalidate(area, key):
    """A strong audit revoked this cache entry; never change the receipt here."""
    import os
    from foundation_v1.stage_certification import invalidate as invalidate_certificate
    invalidate_certificate(area,'receipt invalidation')
    path = Path(area)/'events'/'invalidations.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8') as f:
        f.write(json.dumps({'id': key, 'time': time.time(), 'invalidated': True})+'\n')
        f.flush()
        os.fsync(f.fileno())


class RecoveryIndex:
    """Single parent writer; index rows and consumed journal offsets commit together."""
    def __init__(self, plan, area, wanted, force_rebuild=False):
        started = time.perf_counter()
        self.area = Path(area)
        self.area.mkdir(parents=True, exist_ok=True)
        self.path = self.area/'recovery_index.sqlite'
        self.wanted = set(wanted)
        self.identity = dict(schema=SCHEMA, partition=self.area.parent.parent.name,
                             stage=self.area.name, plan_sha256=file_hash(plan),
                             receipt_schema=receipts.SCHEMA)
        self.done = set()
        self.added = {}
        self.removed = set()
        self.offsets = {}
        self.db = None
        self.disabled = False
        self.last_flush = time.monotonic()
        self.stats = dict(recovery_mode='indexed', recovery_index_rows=0,
                          replayed_journal_events=0, verified_new_receipts=0,
                          full_historical_receipt_opens=0, pending_recovered=0)
        reason = 'explicit maintenance rebuild' if force_rebuild else None
        if reason is None:
            try:
                if not self.path.exists():raise ValueError('recovery index missing')
                self.db = sqlite3.connect(self.path)
                self.db.execute('PRAGMA synchronous=NORMAL')
                if self.db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise ValueError('SQLite quick_check failed')
                meta=dict(self.db.execute('SELECT key,value FROM metadata'))
                self.generation=meta.pop('generation',None)
                if meta != self.identity:
                    raise ValueError('schema/partition/stage/plan/receipt identity mismatch')
                if self.generation is None:
                    self.generation=uuid.uuid4().hex
                    with self.db:self.db.execute('INSERT INTO metadata VALUES(?,?)',('generation',self.generation))
                self.done = {r[0] for r in self.db.execute('SELECT id FROM completed')}
                if not self.done <= self.wanted:raise ValueError('index contains IDs outside this plan')
                self.stats['recovery_index_rows'] = len(self.done)
                for name, pos, dev, ino, head, tail, size, mtime in self.db.execute('SELECT * FROM journal_offsets'):
                    if Path(name).name != name or not name.endswith('.jsonl'):
                        raise ValueError('invalid journal index path')
                    old = (pos, dev, ino, head, tail, size, mtime)
                    path = self.area/'events'/name
                    now = self._stamp(path, pos)
                    if now[:5] != old[:5] or (now[5] == old[5] and now[6] != old[6]):
                        raise ValueError('journal truncated/replaced/rewritten: '+name)
                    self.offsets[name] = old
                self._replay()
            except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as error:
                reason = str(error)
        if reason is not None:
            self._rebuild(reason)
        self._pending()
        self._persist()
        self.stats['recovery_seconds'] = time.perf_counter()-started
        self.stats['completed_after_recovery'] = len(self.done)
        print('RECOVERY '+json.dumps(self.stats, sort_keys=True), flush=True)

    @staticmethod
    def _stamp(path, pos):
        stat = path.stat()
        if pos < 0 or pos > stat.st_size:raise ValueError('journal offset outside file: '+path.name)
        with path.open('rb') as f:
            head = f.read(min(pos, 256))
            f.seek(max(0, pos-256))
            tail = f.read(min(pos, 256))
        return (pos, str(stat.st_dev), str(stat.st_ino), hashlib.sha256(head).hexdigest(),
                hashlib.sha256(tail).hexdigest(), stat.st_size, stat.st_mtime_ns)

    def _verify(self, key, historical=False):
        if key not in self.wanted:return False
        metric = 'full_historical_receipt_opens' if historical else 'verified_new_receipts'
        self.stats[metric] += 1
        try:
            body = receipts.load(self.area, key)
        except (ValueError, OSError, KeyError, TypeError):
            return False
        self.accept(body)
        return True

    def _rebuild(self, reason):
        if self.db is not None:self.db.close()
        self.db = None
        # Only these disposable index files are removable; never receipt trees.
        for suffix in ('', '-journal', '-wal', '-shm'):
            self.path.with_name(self.path.name+suffix).unlink(missing_ok=True)
        self.done.clear()
        self.added.clear()
        self.removed.clear()
        self.offsets.clear()
        self.stats.update(recovery_mode='authoritative_rebuild', rebuild_reason=reason,
                          recovery_index_rows=0)
        for path in (self.area/'receipts').glob('*/F1_*.json'):
            if path.stem in self.wanted:self._verify(path.stem, historical=True)
        # Preserve the existing live-status event view, once per full rebuild.
        events = self.area/'events'
        events.mkdir(exist_ok=True)
        snapshot = ''.join(json.dumps({'id': k, 'time': t})+'\n' for k,t in sorted(self.added.items()))
        atomic_bytes(events/'recovered.jsonl', snapshot.encode(), False)
        for path in events.glob('*.jsonl'):
            self.offsets[path.name] = self._stamp(path, path.stat().st_size)
        self.db = sqlite3.connect(self.path)
        self.db.executescript('''
            PRAGMA synchronous=NORMAL;
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE completed(id TEXT PRIMARY KEY,timestamp REAL NOT NULL);
            CREATE TABLE journal_offsets(name TEXT PRIMARY KEY,pos INTEGER,dev TEXT,ino TEXT,
                head TEXT,tail TEXT,size INTEGER,mtime INTEGER);
        ''')
        self.generation=uuid.uuid4().hex
        with self.db:self.db.executemany('INSERT INTO metadata VALUES(?,?)', {**self.identity,'generation':self.generation}.items())

    def _replay(self):
        for path in sorted((self.area/'events').glob('*.jsonl')):
            pos = self.offsets.get(path.name, (0,))[0]
            with path.open('rb') as f:
                f.seek(pos)
                while True:
                    line = f.readline()
                    if not line or not line.endswith(b'\n'):break
                    event = json.loads(line)
                    key = event['id']
                    self.stats['replayed_journal_events'] += 1
                    if event.get('invalidated'):
                        self.done.discard(key)
                        self.added.pop(key, None)
                        self.removed.add(key)
                    elif key not in self.done:self._verify(key)
                    pos = f.tell()
            self.offsets[path.name] = self._stamp(path, pos)

    def _pending(self):
        for path in (self.area/'events').glob('*.pending.json'):
            try:
                key = read_json(path)['Plan_Row_ID']
                if key not in self.done and self._verify(key):self.stats['pending_recovered'] += 1
            except (ValueError, OSError, KeyError, TypeError):pass

    def accept(self, body):
        """Only callers that already validated a receipt may supply this body."""
        key = body['Plan_Row_ID']
        if key in self.wanted:
            self.done.add(key)
            self.added[key] = body['timestamp']
            self.removed.discard(key)

    def _persist(self):
        if self.disabled:return
        try:
            with self.db:
                self.db.executemany('DELETE FROM completed WHERE id=?', [(k,) for k in self.removed])
                self.db.executemany('INSERT OR REPLACE INTO completed VALUES(?,?)', self.added.items())
                self.db.executemany('INSERT OR REPLACE INTO journal_offsets VALUES(?,?,?,?,?,?,?,?)',
                                    [(name,*value) for name,value in self.offsets.items()])
            self.added.clear()
            self.removed.clear()
            self.last_flush = time.monotonic()
        except sqlite3.Error as error:
            self.disabled = True
            print('RECOVERY_INDEX_WRITE_DEBT: '+str(error)+'; receipts/journals remain authoritative', flush=True)

    def flush(self, force=False):
        if self.disabled:return
        if not force and len(self.added)<BATCH_SIZE and time.monotonic()-self.last_flush<FLUSH_SECONDS:return
        try:
            self._replay()
            self._pending()
            self._persist()
        except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as error:
            self.disabled = True
            # Invalidate only disposable cache metadata; next resume rebuilds.
            try:
                with self.db:self.db.execute('DELETE FROM metadata')
            except sqlite3.Error:pass
            print('RECOVERY_INDEX_REBUILD_REQUIRED: '+str(error), flush=True)

    def close(self):
        self.flush(force=True)
        if self.db is not None:self.db.close()


def main():
    p=argparse.ArgumentParser(description='Rebuild disposable recovery index; no collection or simulations.')
    p.add_argument('--plan',required=True)
    p.add_argument('--area',required=True)
    p.add_argument('--rebuild',action='store_true',required=True)
    a=p.parse_args()
    from am_flexdc_foundation_v1_common import ROOT
    from foundation_v1.campaign import CampaignLock
    with CampaignLock(ROOT/'foundation_v1_campaign.lock'):
        index=RecoveryIndex(a.plan,a.area,{r['plan_row_id'] for r in iter_csv(a.plan)},force_rebuild=True)
        index.close()


if __name__=='__main__':main()
