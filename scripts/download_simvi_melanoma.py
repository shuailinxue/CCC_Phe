"""Download and verify the public SIMVI melanoma object; no model fitting."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/simvi_melanoma_cosmx')
RECORD = 'https://zenodo.org/api/records/14708000'
NAME = 'Melanoma_5612.h5ad'
BLOCK = 32 << 20


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def download():
    metadata = ROOT / 'metadata'
    metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / 'download.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reply = requests.get(RECORD, timeout=45)
        reply.raise_for_status()
        record = reply.json()
        write_json(metadata / 'zenodo_record.json', record)
        entry = next(f for f in record['files'] if f['key'] == NAME)
        total, url = entry['size'], entry['links']['self']
        target = ROOT / NAME
        parts = ROOT / '.download_parts'
        parts.mkdir(exist_ok=True)

        def segment(index):
            start, stop = index * BLOCK, min((index + 1) * BLOCK, total) - 1
            part = parts / f'{index:04d}.bin'
            if part.exists() and part.stat().st_size == stop - start + 1:
                return part
            for attempt in range(5):
                temporary = part.with_suffix('.partial')
                try:
                    with requests.get(url + f'?range_start={start}&attempt={attempt}',
                                      headers={'Range': f'bytes={start}-{stop}'},
                                      timeout=(25, 100), stream=True) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{stop}/{total}':
                            raise ValueError('Server did not return the requested byte range')
                        with temporary.open('wb') as stream:
                            for chunk in response.iter_content(1 << 20):
                                stream.write(chunk)
                    if temporary.stat().st_size != stop - start + 1:
                        raise ValueError('Truncated public file range')
                    temporary.replace(part)
                    return part
                except (requests.RequestException, ValueError):
                    if attempt == 4:
                        raise

        count = (total + BLOCK - 1) // BLOCK
        if not target.exists():
            completed = 0
            with ThreadPoolExecutor(max_workers=4) as pool:
                for future in as_completed([pool.submit(segment, i) for i in range(count)]):
                    part = future.result()
                    completed += part.stat().st_size
                    write_json(metadata / 'download_progress.json', dict(stage='downloading', completed_bytes=completed, total_bytes=total))
                    print(f'Download {completed:,}/{total:,} bytes ({completed / total:.1%})', flush=True)
            temporary = ROOT / (NAME + '.partial')
            with temporary.open('wb') as output:
                for i in range(count):
                    with (parts / f'{i:04d}.bin').open('rb') as source:
                        for chunk in iter(lambda: source.read(1 << 20), b''):
                            output.write(chunk)
            temporary.replace(target)
        md5, sha = hashlib.md5(), hashlib.sha256()
        with target.open('rb') as source:
            for chunk in iter(lambda: source.read(8 << 20), b''):
                md5.update(chunk)
                sha.update(chunk)
        if target.stat().st_size != total or 'md5:' + md5.hexdigest() != entry['checksum']:
            raise ValueError('Source size or Zenodo-published MD5 mismatch; file not accepted')
        print('Zenodo MD5 verified:', md5.hexdigest(), flush=True)
        audit = inspect(target)
        write_json(metadata / 'download_complete.json', dict(source=url, bytes=total, md5=md5.hexdigest(), sha256=sha.hexdigest(),
                   completed_at=datetime.now().astimezone().isoformat(), **audit))
        write_json(metadata / 'download_progress.json', dict(stage='complete', completed_bytes=total, total_bytes=total))
        for part in parts.glob('*.bin'):
            part.unlink()
        if not any(parts.iterdir()):
            parts.rmdir()
        print('Complete:', target, audit, flush=True)


def inspect(path):
    import anndata as ad
    import numpy as np
    from scipy import sparse
    a = ad.read_h5ad(path)
    if a.shape != (56761, 960) or a.obs['CPID_str'].nunique() != 25:
        raise ValueError('Unexpected cohort dimensions')
    xy = np.asarray(a.obsm['spatial'])
    if xy.shape != (len(a), 2) or not np.isfinite(xy).all():
        raise ValueError('Invalid spatial coordinates')
    clinical = ['SEX', 'RACE', 'AGE_AT_DX', 'DX_STAGE', 'BEST_RESPONSE_BY_SCAN', 'PFS_BY_SCAN',
                'PROG_STATUS_BY_SCAN', 'OS_FROM_START_OF_ITX', 'VITAL', 'SPEC_CATEGORY',
                'Prior_Checkpoint_Blockade', 'TREATMENT', 'MUTATION']
    for column in clinical:
        if (a.obs.groupby('CPID_str', observed=True)[column].nunique(dropna=False) > 1).any():
            raise ValueError('Inconsistent patient-level metadata: ' + column)
    a.obs[['CPID_str', *clinical]].drop_duplicates().to_csv(ROOT / 'metadata/patients.csv', index=False)
    a.obs['celltype'].value_counts().rename_axis('author_cell_type').to_csv(ROOT / 'metadata/cell_type_counts.csv')
    matrices = {}
    candidates = {'X': a.X, **{'layers/' + key: value for key, value in a.layers.items()}}
    if a.raw is not None:
        candidates['raw/X'] = a.raw.X
    for name, matrix in candidates.items():
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
        matrices[name] = dict(shape=list(matrix.shape), finite=bool(np.isfinite(values).all()),
                              nonnegative=bool((values >= 0).all()), integer_values=bool(np.allclose(values, np.rint(values), atol=1e-6, rtol=0)))
    audit = dict(cells=a.n_obs, genes=a.n_vars, patients=25, author_cell_types=int(a.obs.celltype.nunique()),
                 obs_columns=list(a.obs.columns), obsm_keys=list(a.obsm), expression_matrices=matrices,
                 clinical_units_and_event_coding='Retained as published; verify definitions before survival analysis',
                 model_training=False, source_object_modified=False)
    write_json(ROOT / 'metadata/data_audit.json', audit)
    return audit


if __name__ == '__main__':
    download()
