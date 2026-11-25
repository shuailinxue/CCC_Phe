"""Small, backend-independent receipts; generated artifacts always stay on NAS."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import numpy as np
from ..storage import STORAGE_ROOT


def output_path(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(STORAGE_ROOT.resolve()):
        raise ValueError('Generated model outputs must stay under the NAS workspace')
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_identity(value):
    """Bounded hashing for arrays, file identity for read-only matrix memmaps."""
    result = dict(shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, np.memmap) and value.filename:
        path = Path(value.filename).resolve()
        stat = path.stat()
        return dict(result, path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    h = hashlib.sha256()
    for start in range(0, len(value), 2048):
        h.update(np.ascontiguousarray(value[start:start + 2048]).tobytes())
    return dict(result, sha256=h.hexdigest())


def bind_backend(folder, family):
    """Reject output-directory reuse across independently packaged backends."""
    families = {'gnn'}
    if family not in families:
        raise ValueError('Unrecognized model backend')
    path = Path(folder) / 'model_backend.json'
    if path.exists():
        if json.loads(path.read_text())['backend'] != family:
            raise ValueError('This output directory belongs to the other model backend')
    elif any((Path(folder) / other).exists() for other in families - {family}):
        raise ValueError('Use a separate output directory for each model backend')
    else:
        write_json(path, dict(backend=family, format_version=1))


@contextmanager
def fit_lock(folder):
    with (Path(folder) / 'fit.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This model output is already in use') from error
        yield
