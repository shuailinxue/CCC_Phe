#!/usr/bin/env python3
"""Read-only backup inventory; never follow symlinks or read file contents."""
import csv
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def measure():
    seen = set()
    results, errors, links = [], [], []
    with (HERE / 'nas3_backup_roots.tsv').open() as handle:
        roots = list(csv.DictReader(handle, delimiter='\t'))
    for row in roots:
        result = dict(id=row['id'], path=row['path'], logical_bytes=0,
                      allocated_bytes=0, files=0, errors=0)
        pending = [row['path']]
        while pending:
            path = pending.pop()
            try:
                info = os.lstat(path)
                key = (info.st_dev, info.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                result['logical_bytes'] += info.st_size
                result['allocated_bytes'] += info.st_blocks * 512
                if stat.S_ISLNK(info.st_mode):
                    links.append(dict(path=path, target=os.readlink(path)))
                elif stat.S_ISDIR(info.st_mode):
                    with os.scandir(path) as entries:
                        pending.extend(entry.path for entry in entries)
                elif stat.S_ISREG(info.st_mode):
                    result['files'] += 1
            except OSError as exc:
                errors.append(dict(path=path, error=str(exc)))
                result['errors'] += 1
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = dict(complete=not errors, roots=results, errors=errors,
                   symlinks=links, symlink_targets_included=False,
                   logical_bytes=sum(r['logical_bytes'] for r in results),
                   allocated_bytes=sum(r['allocated_bytes'] for r in results))
    (HERE / 'nas3_sizes.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('COMPLETE' if not errors else 'INCOMPLETE: totals are lower bounds', flush=True)
    print('Logical GiB:', summary['logical_bytes'] / 2**30, flush=True)
    print('Allocated GiB:', summary['allocated_bytes'] / 2**30, flush=True)
    print('Symlinks requiring review:', len(links), flush=True)
    return 0 if not errors else 1


if __name__ == '__main__':
    if '--worker' in sys.argv:
        sys.exit(measure())
    try:
        with socket.create_connection(('10.254.29.187', 2049), timeout=4):
            pass
    except OSError as exc:
        print(f'NAS unavailable; size UNKNOWN: {exc}', file=sys.stderr)
        sys.exit(2)
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'])
    try:
        sys.exit(child.wait(timeout=1800))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        child.kill()
        print('Scan interrupted; no complete total. An NFS worker may remain blocked.', file=sys.stderr)
        sys.exit(3)
