"""Download the published AIN Xenium Seurat object with resumable ranges."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import requests

ROOT = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/xenium_ain_kidney')
URL = 'https://zenodo.org/api/records/21793755/files/xenium_AIN_for_deposit.rds/content'
SIZE = 687096972
MD5 = '3a30e20b123ed0a45dbaec0d32f5ed62'
CHUNK = 4 * 1024 * 1024
ROOT.mkdir(parents=True, exist_ok=True)
PART = ROOT / 'xenium_AIN_for_deposit.rds.part'
STATE = ROOT / 'download_parts.json'
FINAL = ROOT / 'xenium_AIN_for_deposit.rds'

def fetch(i):
    start = i * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    for attempt in range(5):
        try:
            r = requests.get(URL, headers={'Range': f'bytes={start}-{end}'}, timeout=120)
            if r.status_code == 403:
                raise PermissionError('Zenodo returned HTTP 403')
            r.raise_for_status()
            if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}' or len(r.content) != end-start+1:
                raise ValueError('Invalid range response')
            with open(PART, 'r+b', buffering=0) as out:
                os.pwrite(out.fileno(), r.content, start)
            return i
        except PermissionError:
            raise
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

if __name__ == '__main__':
    if FINAL.exists():
        print('Existing completed file:', FINAL, flush=True)
        raise SystemExit(0)
    if not PART.exists():
        with open(PART, 'wb') as out:
            out.truncate(SIZE)
    done = set(json.loads(STATE.read_text())) if STATE.exists() else set()
    total = (SIZE + CHUNK - 1) // CHUNK
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, i) for i in range(total) if i not in done]
        for future in concurrent.futures.as_completed(futures):
            done.add(future.result())
            tmp = STATE.with_suffix('.tmp')
            tmp.write_text(json.dumps(sorted(done)))
            tmp.replace(STATE)
            print(f'Downloaded {len(done)}/{total} chunks', flush=True)
    digest = hashlib.md5()
    with open(PART, 'rb') as inp:
        for block in iter(lambda: inp.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != MD5:
        raise ValueError('MD5 mismatch; partial file retained for investigation')
    PART.replace(FINAL)
    print('MD5 verified:', FINAL, flush=True)
