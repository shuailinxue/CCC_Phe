#!/usr/bin/env python3
"""Run disjoint slices on GPUs 1 and 2 after the authorized GPU0 migration."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/phenomap_tnbc_batch.json'
PYTHON = '/home/xueshuailin/miniconda3/envs/cell2st/bin/python'


def legacy_workers():
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            args = (path/'cmdline').read_bytes().decode().strip('\0').split('\0')
            cwd = (path/'cwd').resolve()
            scripts = [a for a in args if a.endswith('/run_phenomap_tnbc.py') or a == 'run_phenomap_tnbc.py']
            if not scripts or (cwd / scripts[0]).resolve() != ROOT/'scripts/run_phenomap_tnbc.py':
                continue
            if '--worker-count' in args or '--smoke' in args or '--preflight' in args:
                continue
            if '--config' in args and Path(args[args.index('--config')+1]).resolve() != CONFIG:
                continue
            found.append(int(path.name))
        except (OSError, ValueError):
            continue
    return found


def main():
    root = Path(json.loads(CONFIG.read_text())['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'dual_launcher.lock').open('a') as launch_lock, (root/'batch.lock').open('a') as batch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pids = legacy_workers()
            if len(pids) != 1:
                raise RuntimeError(f'Cannot safely identify the existing single-GPU run: {pids}. No process stopped.')
            print(f'Graceful handover: interrupting only existing PhenoMap worker PID {pids[0]}; completed caches/checkpoints retained.', flush=True)
            os.kill(pids[0], signal.SIGINT)
            deadline = time.monotonic() + 90
            while True:
                try:
                    fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Existing worker did not exit within 90s; no force kill performed. Retry after it exits.')
                    time.sleep(1)
        children = []
        logs = []
        stop_signal = [None]
        def stop_children(signum, frame):
            stop_signal[0] = signum
            for process in children:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
            # Never wait inside a signal handler: Popen.wait may already hold
            # its internal wait lock in the interrupted main thread.
        signal.signal(signal.SIGTERM, stop_children)
        signal.signal(signal.SIGINT, stop_children)
        try:
            for index, gpu in enumerate((1, 2)):
                log = (root/f'batch_gpu{gpu}.log').open('a', buffering=1)
                logs.append(log)
                env = dict(os.environ, PHENOMAP_GPU=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1')
                process = subprocess.Popen(['bash', str(ROOT/'scripts/run_phenomap_tnbc.sh'), '--gpu', str(gpu), '--worker-index', str(index), '--worker-count', '2'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                children.append(process)
                print(f'GPU{gpu}: PID {process.pid}; log {log.name}', flush=True)
            codes = [process.wait() for process in children]
            print(f'Dual-GPU batch finished: GPU1 exit={codes[0]}, GPU2 exit={codes[1]}', flush=True)
            return 128 + stop_signal[0] if stop_signal[0] else int(any(codes))
        finally:
            for log in logs:
                log.close()


if __name__ == '__main__':
    raise SystemExit(main())
