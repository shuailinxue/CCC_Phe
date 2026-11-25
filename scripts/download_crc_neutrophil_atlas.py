"""Fetch the author-annotated CRC Xenium object to NAS, with resumable ranges.

After startup, a lightweight author-metadata reader can prepare the spatial atlas
while the full expression object downloads. Only metadata are read remotely for
this preview; no cell labels are inferred and no niche analysis is launched.
"""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from phenoniche.cohorts.crc_neutrophil_atlas import ROOT, DATA_URL, DATA_BYTES

BLOCK = 64 << 20
LOCAL = threading.local()


def client():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))))
    return LOCAL.session


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def fetch_metadata():
    directory = ROOT / 'metadata'
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in {
        'study.json': 'https://www.ebi.ac.uk/biostudies/api/v1/studies/S-BIAD2208',
        'file_inventory.json': 'https://www.ebi.ac.uk/biostudies/files/S-BIAD2208/xenium.json',
        'samplesheet.csv': 'https://www.ebi.ac.uk/biostudies/files/S-BIAD2208/xenium/samplesheet.csv',
    }.items():
        path = directory / name
        if not path.exists():
            reply = client().get(url, timeout=90)
            reply.raise_for_status()
            path.write_bytes(reply.content)
    rows = json.loads((directory / 'file_inventory.json').read_text())
    entry = next(x for x in rows if x['path'] == 'xenium/processed/crca_xenium.h5ad')
    if entry['size'] != DATA_BYTES:
        raise ValueError('The published file size changed; inspect the new release before resuming')


def download(workers=8):
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / 'crca_xenium.h5ad'
    partial = ROOT / 'crca_xenium.h5ad.partial'
    state_path = ROOT / 'metadata/download_chunks.json'
    manifest = ROOT / 'metadata/download_complete.json'
    progress = ROOT / 'metadata/download_progress.json'
    fetch_metadata()
    if target.exists() and manifest.exists():
        record = json.loads(manifest.read_text())
        if target.stat().st_size != DATA_BYTES:
            raise ValueError('Completed H5AD size no longer matches its receipt')
        return target
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    completed = state.get('completed', {}) if partial.exists() else {}
    if state and (state['source'] != DATA_URL or state['total_bytes'] != DATA_BYTES):
        raise ValueError('Download state belongs to a different source')
    started = time.monotonic()
    count = (DATA_BYTES + BLOCK - 1) // BLOCK
    descriptor = os.open(partial, os.O_CREAT | os.O_RDWR, 0o664)
    os.ftruncate(descriptor, DATA_BYTES)
    try:
        # A resumed segment is trusted only after its own local SHA256 matches.
        for key, receipt in list(completed.items()):
            start = int(key) * BLOCK
            block = os.pread(descriptor, receipt['bytes'], start)
            if hashlib.sha256(block).hexdigest() != receipt['sha256']:
                del completed[key]

        def segment(index):
            start = index * BLOCK
            end = min(start + BLOCK, DATA_BYTES) - 1
            for attempt in range(7):
                try:
                    with client().get(DATA_URL, headers={'Range': f'bytes={start}-{end}'},
                                      stream=True, timeout=(30, 150)) as reply:
                        reply.raise_for_status()
                        if reply.status_code != 206 or reply.headers.get('Content-Range') != f'bytes {start}-{end}/{DATA_BYTES}':
                            raise IOError('Server returned an unexpected byte range')
                        digest = hashlib.sha256()
                        position = start
                        for block in reply.iter_content(2 << 20):
                            # pwrite does not share the seek position between workers.
                            view = memoryview(block)
                            while view:
                                written = os.pwrite(descriptor, view, position)
                                if written <= 0:
                                    raise IOError('Unable to write download segment')
                                view = view[written:]
                                position += written
                            digest.update(block)
                        if position != end + 1:
                            raise IOError('Incomplete byte range')
                    return str(index), dict(bytes=end - start + 1, sha256=digest.hexdigest())
                except Exception:
                    if attempt == 6:
                        raise
                    time.sleep(min(2 ** attempt, 20))
            raise AssertionError('Unreachable')

        pending = [i for i in range(count) if str(i) not in completed]
        initial = sum(x['bytes'] for x in completed.values())
        print(f'Download H5AD: {DATA_BYTES / 1e9:.2f} GB; resume {initial / 1e9:.2f} GB; {workers} connections', flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(segment, i) for i in pending]
            last_log = 0.
            for future in concurrent.futures.as_completed(futures):
                key, receipt = future.result()
                completed[key] = receipt
                done = sum(x['bytes'] for x in completed.values())
                write_json(state_path, dict(source=DATA_URL, total_bytes=DATA_BYTES, completed=completed))
                elapsed = time.monotonic() - started
                speed = (done - initial) / max(elapsed, .01)
                status = dict(stage='downloading', pid=os.getpid(), bytes=done,
                              total_bytes=DATA_BYTES, fraction=done / DATA_BYTES,
                              elapsed_seconds=elapsed, megabytes_per_second=speed / 1e6,
                              eta_seconds=(DATA_BYTES - done) / max(speed, 1))
                write_json(progress, status)
                if time.monotonic() - last_log > 15 or done == DATA_BYTES:
                    print(f"H5AD {done / DATA_BYTES:.1%}: {done / 1e9:.2f} GB; {speed / 1e6:.1f} MB/s; ETA {status['eta_seconds'] / 60:.0f} min", flush=True)
                    last_log = time.monotonic()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    write_json(progress, dict(stage='validating', bytes=DATA_BYTES, total_bytes=DATA_BYTES))
    digest = hashlib.sha256()
    with partial.open('rb') as handle:
        for block in iter(lambda: handle.read(16 << 20), b''):
            digest.update(block)
    import h5py
    with h5py.File(partial, 'r') as h:
        if 'celltype' not in h['obs'] or 'spatial' not in h['obsm'] or 'counts' not in h['layers']:
            raise ValueError('Missing required author labels, coordinates or native counts')
        cell_count = len(h['obs/celltype/codes'])
    partial.replace(target)
    record = dict(source=DATA_URL, bytes=DATA_BYTES, sha256=digest.hexdigest(),
                  cells=cell_count, validated_fields=['obs/celltype', 'obsm/spatial', 'layers/counts'],
                  finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write_json(manifest, record)
    write_json(progress, dict(stage='complete', **record))
    print('Author H5AD downloaded and verified:', target, flush=True)
    return target


def execute_notebook():
    import nbformat
    from nbclient import NotebookClient
    path = PROJECT / 'Xenium_CRC_Neutrophil_Atlas.ipynb'
    book = nbformat.read(path, as_version=4)
    # Download/preview must not launch the subsequently appended joint fit.
    client = NotebookClient(book, timeout=1200, kernel_name='cccphe',
                            resources={'metadata': {'path': str(PROJECT)}})
    with client.setup_kernel():
        for index, cell in enumerate(book.cells):
            if cell.cell_type == 'code' and not cell.metadata.get('crc_selected_five'):
                client.execute_cell(cell, index)
    live = nbformat.read(path, as_version=4)
    for cell in live.cells:
        if cell.cell_type == 'code' and not cell.metadata.get('crc_selected_five'):
            matches = [c for c in book.cells if c.cell_type == 'code' and c.source == cell.source]
            if len(matches) == 1:
                cell.outputs, cell.execution_count = matches[0].outputs, matches[0].execution_count
    temp = path.with_suffix('.ipynb.tmp')
    nbformat.write(live, temp)
    temp.replace(path)
    print('Executed notebook:', path, flush=True)


def run():
    from phenoniche.cohorts.crc_neutrophil_atlas import prepare_display_table
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A display uses only author labels and positions, so it need not wait for
        # the wide expression/layer objects. The full H5AD is still downloaded.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            transfer = executor.submit(download)
            try:
                prepare_display_table(allow_remote=True)
                execute_notebook()
            except Exception as error:
                print('Early metadata display deferred:', repr(error), flush=True)
                transfer.result()
                prepare_display_table(allow_remote=False)
                execute_notebook()
            transfer.result()
        print('All requested download and visualization work completed.', flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        directory = ROOT / 'metadata'
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'controller_error.json', dict(error=repr(error), pid=os.getpid()))
        raise
