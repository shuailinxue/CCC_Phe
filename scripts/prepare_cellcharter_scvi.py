"""Create the isolated compatible scVI overlay without changing cccphe."""
from pathlib import Path
import os
import shutil
import subprocess
import venv
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT/'baselines/config.json').read_text())
    python = Path(cfg['cellcharter_scvi_python']); target = python.parent.parent
    venv.EnvBuilder(system_site_packages=True, with_pip=True, symlinks=False).create(target)
    # venv may create a lib64 directory alias even with symlinks=False.
    for p in target.rglob('*'):
        if p.is_symlink():
            source = p.resolve(); p.unlink()
            if source.is_file():shutil.copy2(source,p)
            elif p.name!='lib64':shutil.copytree(source,p)
    workspace = target.parents[2]
    env = dict(os.environ,TMPDIR=str(workspace/'tmp'),PIP_CACHE_DIR=str(workspace/'artifacts/pip_cache'))
    subprocess.run([str(python),'-m','pip','install','--no-deps','-r',str(ROOT/'baselines/scvi-requirements.txt')],check=True,env=env)
    subprocess.run([str(python),'-c','import scvi; print(scvi.__version__)'],check=True,env=dict(env,JAX_PLATFORMS='cpu'))


if __name__=='__main__':main()
