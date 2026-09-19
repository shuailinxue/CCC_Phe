from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def get_logger(name: str = "cccphe") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def md5sum(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.md5()  # noqa: S324 - GDC publishes MD5 checksums for transfer QC.
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: Any, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)


def write_provenance(destination: str | Path, cfg: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cfg["project"]["code_root"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_commit = None
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.get("_config_path"),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit,
        **(extra or {}),
    }
    atomic_json(payload, destination)


def geometric_mean(values, axis: int = -1, epsilon: float = 0.0):
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if np.any(array < 0):
        raise ValueError("Geometric mean requires non-negative expression")
    if epsilon:
        return np.exp(np.mean(np.log(array + epsilon), axis=axis)) - epsilon
    zero = np.any(array == 0, axis=axis)
    out = np.exp(np.mean(np.log(np.where(array > 0, array, 1.0)), axis=axis))
    return np.where(zero, 0.0, out)


def require_columns(frame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
